#!/usr/bin/env python3
"""Production knowledge ingestion package runtime (v1, bounded substrate).

This helper introduces a fail-closed ingestion package layer that sits in front of
Wave 5's promotion queue/review/promote stack.

Commands:
- create   : build a bounded ingestion package around an existing promotion candidate
- validate : schema + provenance/fixity checks for an ingestion package
- enqueue  : validate package then enqueue candidate into knowledge_promotion_queue
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_PACKAGE_SCHEMA_PATH = Path("docs/ops/schemas/knowledge_ingestion_package.schema.json")
DEFAULT_PROMOTION_SCHEMA_PATH = Path("docs/ops/schemas/promotion_candidate.schema.json")
DEFAULT_EVENTS_LOG_PATH = Path("state/continuity/knowledge_ingestion/events.jsonl")
DEFAULT_QUEUE_RUNNER = SCRIPT_PATH.parent / "knowledge_promotion_queue.py"
DEFAULT_QUEUE_STATE_PATH = Path("state/continuity/knowledge_promotion_queue/state.json")
DEFAULT_QUEUE_EVENTS_LOG_PATH = Path("state/continuity/knowledge_promotion_queue/events.jsonl")

PACKAGE_SCHEMA_VERSION = "clawd.knowledge_ingestion.package.v1"
EVIDENCE_SCHEMA_VERSION = "clawd.knowledge_ingestion.evidence_bundle.v1"
INGESTION_STATE_READY = "READY_FOR_QUEUE"
INGESTION_STATE_QUEUED = "QUEUED_REVIEW"
QUEUE_STATUSES = {"QUEUED_REVIEW", "APPROVED", "PROMOTED", "REJECTED", "BLOCKED"}
PROMOTION_ID_RE = re.compile(r"^prom_[a-z0-9._-]+$")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(stable_json_dumps(payload) + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def normalize_sha256(raw: str) -> str:
    token = str(raw or "").strip().lower()
    if token.startswith("sha256:"):
        token = token.split(":", 1)[1]
    return token


def rel_posix(repo_root: Path, target: Path) -> str:
    return target.relative_to(repo_root).as_posix()


def _validate_schema(payload: Any, schema_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "schema_validator_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists():
        return False, "schema_missing", {"error": "schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "schema_unreadable", {"error": "schema_unreadable", "detail": str(exc)}

    if not isinstance(schema_doc, dict):
        return False, "schema_not_object", {"error": "schema_not_object"}

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, "", {}

    err = errors[0]
    return (
        False,
        "schema_invalid",
        {
            "error": "schema_validation_failed",
            "data_path": "$" if not err.absolute_path else "$/" + "/".join(str(x) for x in err.absolute_path),
            "message": str(err.message),
        },
    )


def _candidate_from_path(repo_root: Path, candidate_path: Path) -> Dict[str, Any]:
    if not is_within(repo_root, candidate_path):
        raise RuntimeError("candidate_path_outside_repo")
    if not candidate_path.exists() or not candidate_path.is_file():
        raise RuntimeError("candidate_path_unresolved")
    payload = load_json_file(candidate_path)
    if not isinstance(payload, dict):
        raise RuntimeError("candidate_not_object")
    return payload


def _derive_package_id(promotion_id: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "_", promotion_id.lower()).strip("._-")
    if slug.startswith("prom_"):
        slug = slug[len("prom_") :]
    if not slug:
        slug = "candidate"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dt%H%M%S").lower()
    return f"kip_{slug}_{stamp}"


def _collect_candidate_source_refs(candidate: Mapping[str, Any]) -> List[Dict[str, str]]:
    source_refs = candidate.get("source_refs") if isinstance(candidate.get("source_refs"), list) else []
    out: List[Dict[str, str]] = []
    for idx, row in enumerate(source_refs):
        if not isinstance(row, Mapping):
            continue
        raw_path = str(row.get("path") or "").strip()
        if not raw_path:
            continue
        ref_id = str(row.get("ref_id") or f"src_ref_{idx + 1}").strip()
        out.append({"source_ref_id": ref_id, "path": raw_path})
    return out


def _collect_explicit_refs(raw_refs: Sequence[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for idx, raw in enumerate(raw_refs):
        path_token = str(raw or "").strip()
        if not path_token:
            continue
        out.append({"source_ref_id": f"src_explicit_{idx + 1}", "path": path_token})
    return out


def _build_preserved_evidence_items(
    *,
    repo_root: Path,
    source_rows: Sequence[Mapping[str, str]],
    captured_at: str,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for idx, row in enumerate(source_rows):
        raw_path = str(row.get("path") or "").strip()
        if not raw_path:
            continue
        resolved = resolve_repo_path(repo_root, raw_path)
        if not is_within(repo_root, resolved):
            raise RuntimeError(f"evidence_path_outside_repo:{raw_path}")
        if not resolved.exists() or not resolved.is_file():
            raise RuntimeError(f"evidence_path_unresolved:{raw_path}")

        rel = rel_posix(repo_root, resolved)
        if rel in seen:
            continue
        seen.add(rel)

        digest = file_sha256(resolved)
        stat = resolved.stat()
        source_ref_id = str(row.get("source_ref_id") or f"src_ref_{idx + 1}")

        items.append(
            {
                "evidence_id": f"ev_{idx + 1:03d}",
                "source_ref_id": source_ref_id,
                "path": rel,
                "content_hash": f"sha256:{digest}",
                "bytes": int(stat.st_size),
                "captured_at": captured_at,
                "media_type": "text/plain" if resolved.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml"} else "application/octet-stream",
            }
        )

    if not items:
        raise RuntimeError("evidence_refs_missing")
    return items


def _default_package_out(repo_root: Path, package_id: str) -> Path:
    return (repo_root / "state" / "continuity" / "knowledge_ingestion" / "packages" / f"{package_id}.json").resolve()


def _validate_package_runtime(
    *,
    repo_root: Path,
    package: Mapping[str, Any],
) -> Tuple[bool, List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []

    candidate_ref = package.get("promotion_candidate_ref") if isinstance(package.get("promotion_candidate_ref"), Mapping) else {}
    candidate_path_raw = str(candidate_ref.get("path") or "").strip()
    candidate_hash_raw = str(candidate_ref.get("content_hash") or "").strip()
    if not candidate_path_raw:
        issues.append({"code": "candidate_path_missing"})
    else:
        cpath = resolve_repo_path(repo_root, candidate_path_raw)
        if not is_within(repo_root, cpath):
            issues.append({"code": "candidate_path_outside_repo", "path": candidate_path_raw})
        elif not cpath.exists() or not cpath.is_file():
            issues.append({"code": "candidate_path_unresolved", "path": candidate_path_raw})
        else:
            actual = file_sha256(cpath)
            declared = normalize_sha256(candidate_hash_raw)
            if declared != actual:
                issues.append(
                    {
                        "code": "candidate_fixity_mismatch",
                        "path": candidate_path_raw,
                        "declared": declared,
                        "actual": actual,
                    }
                )

    preserved = package.get("preserved_evidence") if isinstance(package.get("preserved_evidence"), Mapping) else {}
    items = preserved.get("items") if isinstance(preserved.get("items"), list) else []
    if not items:
        issues.append({"code": "preserved_evidence_items_missing"})

    declared_item_count = preserved.get("item_count")
    if isinstance(declared_item_count, int) and declared_item_count != len(items):
        issues.append(
            {
                "code": "preserved_evidence_item_count_mismatch",
                "declared": declared_item_count,
                "observed": len(items),
            }
        )

    mismatch_count = 0
    checked_count = 0
    for idx, row in enumerate(items):
        if not isinstance(row, Mapping):
            issues.append({"code": "evidence_item_not_object", "index": idx})
            continue
        raw_path = str(row.get("path") or "").strip()
        raw_hash = str(row.get("content_hash") or "").strip()
        if not raw_path:
            issues.append({"code": "evidence_path_missing", "index": idx})
            continue
        resolved = resolve_repo_path(repo_root, raw_path)
        if not is_within(repo_root, resolved):
            issues.append({"code": "evidence_path_outside_repo", "index": idx, "path": raw_path})
            continue
        if not resolved.exists() or not resolved.is_file():
            issues.append({"code": "evidence_path_unresolved", "index": idx, "path": raw_path})
            continue

        checked_count += 1
        declared = normalize_sha256(raw_hash)
        actual = file_sha256(resolved)
        if declared != actual:
            mismatch_count += 1
            issues.append(
                {
                    "code": "evidence_fixity_mismatch",
                    "index": idx,
                    "path": raw_path,
                    "declared": declared,
                    "actual": actual,
                }
            )

    provenance = package.get("provenance") if isinstance(package.get("provenance"), Mapping) else {}
    declared_source_ref_ids = provenance.get("source_ref_ids") if isinstance(provenance.get("source_ref_ids"), list) else []
    observed_source_ref_ids = [str((row or {}).get("source_ref_id") or "") for row in items if isinstance(row, Mapping)]
    if declared_source_ref_ids:
        if sorted(str(x) for x in declared_source_ref_ids) != sorted(observed_source_ref_ids):
            issues.append(
                {
                    "code": "provenance_source_ref_ids_mismatch",
                    "declared": declared_source_ref_ids,
                    "observed": observed_source_ref_ids,
                }
            )

    handoff = package.get("handoff") if isinstance(package.get("handoff"), Mapping) else {}
    queue_entry_id = handoff.get("queue_entry_id")
    queue_status = handoff.get("queue_status")
    if queue_entry_id not in {None, ""}:
        if not isinstance(queue_entry_id, str) or not queue_entry_id.startswith("kpq_"):
            issues.append({"code": "queue_entry_id_invalid", "value": queue_entry_id})
        if queue_status not in QUEUE_STATUSES:
            issues.append({"code": "queue_status_invalid", "value": queue_status})

    fixity = package.get("fixity") if isinstance(package.get("fixity"), Mapping) else {}
    declared_status = str(fixity.get("verification_status") or "")
    if declared_status == "verified" and mismatch_count > 0:
        issues.append({"code": "fixity_status_claim_invalid", "status": declared_status, "mismatch_count": mismatch_count})

    declared_mismatch = fixity.get("mismatch_count")
    if isinstance(declared_mismatch, int) and declared_mismatch != mismatch_count:
        issues.append(
            {
                "code": "fixity_mismatch_count_incorrect",
                "declared": declared_mismatch,
                "observed": mismatch_count,
            }
        )

    declared_checked = fixity.get("checked_item_count")
    if isinstance(declared_checked, int) and declared_checked != checked_count:
        issues.append(
            {
                "code": "fixity_checked_item_count_incorrect",
                "declared": declared_checked,
                "observed": checked_count,
            }
        )

    return (len(issues) == 0), issues


def cmd_create(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    promotion_schema = resolve_repo_path(repo_root, args.promotion_schema_path)

    candidate_path = resolve_repo_path(repo_root, args.candidate)
    try:
        candidate = _candidate_from_path(repo_root, candidate_path)
    except Exception as exc:
        return 2, {"schema": "clawd.knowledge_ingestion.result.v1", "action": "create", "ok": False, "error": str(exc)}

    ok, reason, details = _validate_schema(candidate, promotion_schema)
    if not ok:
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "create",
            "ok": False,
            "error": f"promotion_candidate_{reason}",
            "details": details,
        }

    promotion_id = str(candidate.get("promotion_id") or "").strip()
    if not PROMOTION_ID_RE.fullmatch(promotion_id):
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "create",
            "ok": False,
            "error": "promotion_id_invalid",
        }

    source_rows = _collect_explicit_refs(args.evidence_ref or [])
    if not source_rows:
        source_rows = _collect_candidate_source_refs(candidate)

    captured_at = now_iso()
    try:
        evidence_items = _build_preserved_evidence_items(repo_root=repo_root, source_rows=source_rows, captured_at=captured_at)
    except Exception as exc:
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "create",
            "ok": False,
            "error": str(exc),
        }

    package_id = str(args.package_id or "").strip() or _derive_package_id(promotion_id)
    if not package_id.startswith("kip_"):
        package_id = f"kip_{package_id}"

    package_out = resolve_repo_path(repo_root, args.package_out) if args.package_out else _default_package_out(repo_root, package_id)
    if not is_within(repo_root, package_out):
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "create",
            "ok": False,
            "error": "unsafe_package_out_path",
            "package_path": str(package_out),
        }

    timestamp = now_iso()
    candidate_rel = rel_posix(repo_root, candidate_path)
    candidate_hash = file_sha256(candidate_path)

    source_lane = candidate.get("source_lane") if isinstance(candidate.get("source_lane"), Mapping) else {}
    provenance = candidate.get("provenance") if isinstance(candidate.get("provenance"), Mapping) else {}

    package_payload: Dict[str, Any] = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "package_id": package_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "ingestion_state": INGESTION_STATE_READY,
        "source_lane": {
            "lane_id": str(source_lane.get("lane_id") or "unknown_lane"),
            "work_item_id": str(source_lane.get("work_item_id") or "unknown_work_item"),
            "producer_role": str(source_lane.get("producer_role") or "EXECUTOR"),
            "session_key": source_lane.get("session_key"),
        },
        "promotion_candidate_ref": {
            "promotion_id": promotion_id,
            "path": candidate_rel,
            "content_hash": f"sha256:{candidate_hash}",
        },
        "preserved_evidence": {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "bundle_role": "promotion_candidate_supporting_evidence",
            "item_count": len(evidence_items),
            "items": evidence_items,
        },
        "provenance": {
            "capture_method": str(provenance.get("capture_method") or "direct_synthesis"),
            "captured_at": str(provenance.get("captured_at") or timestamp),
            "collector_role": source_lane.get("producer_role"),
            "tool_trace_refs": list(provenance.get("tool_trace_refs") or []),
            "source_ref_ids": [str(item.get("source_ref_id") or "") for item in evidence_items],
        },
        "fixity": {
            "algorithm": "sha256",
            "verification_status": "verified",
            "verified_at": timestamp,
            "mismatch_count": 0,
            "checked_item_count": len(evidence_items),
        },
        "handoff": {
            "queue_runtime": "scripts/knowledge_promotion_queue.py",
            "queue_entry_id": None,
            "queue_status": None,
            "enqueued_at": None,
            "last_event_ref": None,
        },
    }

    package_schema = resolve_repo_path(repo_root, args.package_schema_path)
    ok, reason, details = _validate_schema(package_payload, package_schema)
    if not ok:
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "create",
            "ok": False,
            "error": f"package_{reason}",
            "details": details,
        }

    atomic_write_json(package_out, package_payload)

    return 0, {
        "schema": "clawd.knowledge_ingestion.result.v1",
        "action": "create",
        "ok": True,
        "package": package_payload,
        "package_path": str(package_out),
    }


def cmd_validate(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    package_schema = resolve_repo_path(repo_root, args.package_schema_path)
    package_path = resolve_repo_path(repo_root, args.package)

    if not is_within(repo_root, package_path):
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "validate",
            "ok": False,
            "error": "unsafe_package_path",
            "package_path": str(package_path),
        }

    if not package_path.exists() or not package_path.is_file():
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "validate",
            "ok": False,
            "error": "package_path_unresolved",
            "package_path": str(package_path),
        }

    try:
        package = load_json_file(package_path)
    except Exception as exc:
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "validate",
            "ok": False,
            "error": "package_json_unreadable",
            "detail": str(exc),
        }

    ok_schema, reason, details = _validate_schema(package, package_schema)
    if not ok_schema:
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "validate",
            "ok": False,
            "error": f"package_{reason}",
            "details": details,
            "package_path": str(package_path),
        }

    runtime_ok, issues = _validate_package_runtime(repo_root=repo_root, package=package if isinstance(package, Mapping) else {})

    payload = {
        "schema": "clawd.knowledge_ingestion.result.v1",
        "action": "validate",
        "ok": runtime_ok,
        "package_path": str(package_path),
        "issue_count": len(issues),
        "issues": issues,
    }
    return (0 if runtime_ok else 2), payload


def _queue_runner_path(repo_root: Path, raw: str) -> Path:
    candidate = resolve_repo_path(repo_root, raw)
    if candidate.exists() and candidate.is_file():
        return candidate
    return DEFAULT_QUEUE_RUNNER


def cmd_enqueue(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()

    rc_validate, validate_payload = cmd_validate(args)
    if rc_validate != 0:
        out = dict(validate_payload)
        out["action"] = "enqueue"
        out["error"] = out.get("error") or "package_invalid"
        out["ok"] = False
        return 2, out

    package_path = resolve_repo_path(repo_root, args.package)
    package = load_json_file(package_path)
    if not isinstance(package, dict):
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "enqueue",
            "ok": False,
            "error": "package_not_object",
        }

    handoff = package.get("handoff") if isinstance(package.get("handoff"), dict) else {}
    if handoff.get("queue_entry_id"):
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "enqueue",
            "ok": False,
            "error": "package_already_queued",
            "queue_entry_id": handoff.get("queue_entry_id"),
            "queue_status": handoff.get("queue_status"),
        }

    candidate_ref = package.get("promotion_candidate_ref") if isinstance(package.get("promotion_candidate_ref"), dict) else {}
    candidate_path = resolve_repo_path(repo_root, str(candidate_ref.get("path") or ""))
    if not is_within(repo_root, candidate_path):
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "enqueue",
            "ok": False,
            "error": "candidate_path_outside_repo",
        }

    queue_runner = _queue_runner_path(repo_root, args.queue_runner)
    promotion_schema = resolve_repo_path(repo_root, args.promotion_schema_path)

    command = [
        sys.executable,
        str(queue_runner),
        "--repo-root",
        str(repo_root),
        "--state-path",
        args.queue_state_path,
        "--events-log",
        args.queue_events_log,
        "--promotion-schema-path",
        str(promotion_schema),
        "--json",
        "enqueue",
        "--candidate",
        str(candidate_path),
    ]

    queue_env = os.environ.copy()
    queue_env["OPENCLAW_INTERNAL_MUTATION"] = "1"
    queue_env["OPENCLAW_INTERNAL_MUTATION_CALLSITE"] = "knowledge_ingestion_package.py:enqueue"

    cp = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        cwd=str(DEFAULT_REPO_ROOT),
        env=queue_env,
    )
    try:
        queue_payload = json.loads(cp.stdout.strip()) if cp.stdout.strip() else {}
    except Exception as exc:
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "enqueue",
            "ok": False,
            "error": "queue_output_not_json",
            "detail": str(exc),
            "queue_stdout": cp.stdout,
            "queue_stderr": cp.stderr,
            "queue_rc": cp.returncode,
        }

    if cp.returncode != 0 or not isinstance(queue_payload, dict) or not bool(queue_payload.get("ok")):
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "enqueue",
            "ok": False,
            "error": "queue_enqueue_failed",
            "queue_rc": cp.returncode,
            "queue_payload": queue_payload,
            "queue_stderr": cp.stderr,
        }

    entry = queue_payload.get("entry") if isinstance(queue_payload.get("entry"), dict) else {}
    queue_entry_id = str(entry.get("queue_entry_id") or "").strip()
    queue_status = str(entry.get("status") or "").strip() or None
    if not queue_entry_id.startswith("kpq_"):
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "enqueue",
            "ok": False,
            "error": "queue_entry_id_missing",
            "queue_payload": queue_payload,
        }

    ts = now_iso()
    package["updated_at"] = ts
    package["ingestion_state"] = INGESTION_STATE_QUEUED

    handoff_out = handoff.copy()
    handoff_out["queue_runtime"] = rel_posix(repo_root, queue_runner) if is_within(repo_root, queue_runner) else str(queue_runner)
    handoff_out["queue_entry_id"] = queue_entry_id
    handoff_out["queue_status"] = queue_status
    handoff_out["enqueued_at"] = ts

    events_log = resolve_repo_path(repo_root, args.events_log)
    if not is_within(repo_root, events_log):
        return 2, {
            "schema": "clawd.knowledge_ingestion.result.v1",
            "action": "enqueue",
            "ok": False,
            "error": "unsafe_events_log_path",
            "events_log": str(events_log),
        }

    event_row = {
        "ts": ts,
        "schema": "clawd.knowledge_ingestion.event.v1",
        "event": "enqueue",
        "package_id": package.get("package_id"),
        "promotion_id": candidate_ref.get("promotion_id"),
        "queue_entry_id": queue_entry_id,
        "queue_status": queue_status,
    }
    append_jsonl(events_log, event_row)
    handoff_out["last_event_ref"] = rel_posix(repo_root, events_log)

    package["handoff"] = handoff_out
    atomic_write_json(package_path, package)

    return 0, {
        "schema": "clawd.knowledge_ingestion.result.v1",
        "action": "enqueue",
        "ok": True,
        "package_path": str(package_path),
        "package": package,
        "queue_result": queue_payload,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Production knowledge ingestion package runtime (v1)")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--package-schema-path", default=str(DEFAULT_PACKAGE_SCHEMA_PATH), help="Ingestion package schema path")
    ap.add_argument("--promotion-schema-path", default=str(DEFAULT_PROMOTION_SCHEMA_PATH), help="Promotion candidate schema path")
    ap.add_argument("--events-log", default=str(DEFAULT_EVENTS_LOG_PATH), help="Ingestion package events jsonl path")
    ap.add_argument("--queue-runner", default=str(DEFAULT_QUEUE_RUNNER), help="knowledge_promotion_queue runner path")
    ap.add_argument("--queue-state-path", default=str(DEFAULT_QUEUE_STATE_PATH), help="Queue state path")
    ap.add_argument("--queue-events-log", default=str(DEFAULT_QUEUE_EVENTS_LOG_PATH), help="Queue events log path")
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON")

    sub = ap.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create ingestion package from promotion candidate")
    p_create.add_argument("--candidate", required=True, help="Promotion candidate path")
    p_create.add_argument("--package-out", default="", help="Output package path")
    p_create.add_argument("--package-id", default="", help="Optional explicit package id")
    p_create.add_argument("--evidence-ref", action="append", default=[], help="Optional explicit evidence path (repeatable)")
    p_create.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_validate = sub.add_parser("validate", help="Validate package schema + runtime fixity")
    p_validate.add_argument("--package", required=True, help="Ingestion package path")
    p_validate.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_enqueue = sub.add_parser("enqueue", help="Validate then enqueue package candidate to promotion queue")
    p_enqueue.add_argument("--package", required=True, help="Ingestion package path")
    p_enqueue.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.command == "create":
        rc, payload = cmd_create(args)
    elif args.command == "validate":
        rc, payload = cmd_validate(args)
    elif args.command == "enqueue":
        rc, payload = cmd_enqueue(args)
    else:
        rc, payload = 2, {"schema": "clawd.knowledge_ingestion.result.v1", "ok": False, "error": f"unknown_command:{args.command}"}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(payload if isinstance(payload, dict) else {"payload": payload}))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
