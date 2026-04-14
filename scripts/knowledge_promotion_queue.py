#!/usr/bin/env python3
"""Knowledge review/approval/promotion queue runtime (v1, canonical path).

Wrapper-only enforcement: mutating commands (enqueue, review, promote) require OPENCLAW_INTERNAL_MUTATION=1
and OPENCLAW_INTERNAL_MUTATION_CALLSITE allowlisted (default: continuity.sh:promotion-queue).
Direct token‑path calls are not supported; use continuity.sh promotion‑queue.
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
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_STATE_PATH = Path("state/continuity/knowledge_promotion_queue/state.json")
DEFAULT_EVENTS_LOG = Path("state/continuity/knowledge_promotion_queue/events.jsonl")
DEFAULT_PROMOTION_SCHEMA_PATH = Path("docs/ops/schemas/promotion_candidate.schema.json")
DEFAULT_PROMOTION_TRACE_SCHEMA_PATH = Path("docs/ops/schemas/promotion_trace_manifest.schema.json")
PROMOTION_WORKFLOW_RUNNER = SCRIPT_PATH.parent / "promotion_review_entrypoint.py"
DEFAULT_SHARED_MEMORY_RUNNER = SCRIPT_PATH.parent / "shared_memory_fabric.py"
WRAPPER_REQUIRED_SCHEMA = "clawd.knowledge_promotion_queue.wrapper_contract.v1"
DEFAULT_MUTATING_COMMANDS = {"enqueue", "review", "promote"}
DEFAULT_ALLOWED_MUTATION_CALLSITES = {
    "continuity.sh:promotion-queue",
    "knowledge_ingestion_package.py:enqueue",
}

QUEUE_SCHEMA = "clawd.knowledge_promotion_queue.state.v1"
ENTRY_SCHEMA = "clawd.knowledge_promotion_queue.entry.v1"
QUEUE_STATUSES = {"QUEUED_REVIEW", "APPROVED", "PROMOTED", "REJECTED", "BLOCKED"}
PROMOTION_ID_RE = re.compile(r"^prom_[a-z0-9._-]+$")

REVIEW_TO_STATUS = {
    "approved": "APPROVED",
    "rejected": "REJECTED",
    "needs_changes": "BLOCKED",
}

REVIEW_TO_PROMOTION_STATE = {
    "approved": "APPROVED",
    "rejected": "REJECTED",
    "needs_changes": "BLOCKED",
}


def _resolve_allowed_mutation_callsites() -> List[str]:
    allowed = set(DEFAULT_ALLOWED_MUTATION_CALLSITES)
    raw = str(os.environ.get("OPENCLAW_KNOWLEDGE_PROMOTION_QUEUE_ALLOWED_CALLSITES") or "").strip()
    if raw:
        for token in raw.split(","):
            value = token.strip()
            if value:
                allowed.add(value)
    return sorted(allowed)


def _enforce_wrapper_only_contract(command: str) -> Optional[Dict[str, Any]]:
    if command not in DEFAULT_MUTATING_COMMANDS:
        return None

    internal_mutation = str(os.environ.get("OPENCLAW_INTERNAL_MUTATION") or "").strip()
    callsite = str(os.environ.get("OPENCLAW_INTERNAL_MUTATION_CALLSITE") or "").strip()
    allowed_callsites = _resolve_allowed_mutation_callsites()

    if internal_mutation != "1":
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_env_missing",
            "required_env": ["OPENCLAW_INTERNAL_MUTATION=1", "OPENCLAW_INTERNAL_MUTATION_CALLSITE=<allowlisted>"],
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh --action-token <...> promotion-queue <enqueue|review|promote> ... --json",
        }

    if not callsite:
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_callsite_missing",
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh --action-token <...> promotion-queue <enqueue|review|promote> ... --json",
        }

    if callsite not in allowed_callsites:
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_callsite_not_allowlisted",
            "callsite": callsite,
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh --action-token <...> promotion-queue <enqueue|review|promote> ... --json",
        }

    return None


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_sha256(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text


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


def _load_or_init_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema": QUEUE_SCHEMA,
            "updated_at": now_iso(),
            "entries": [],
        }

    payload = load_json_file(path)
    if not isinstance(payload, dict):
        raise RuntimeError("queue_state_not_object")
    if payload.get("schema") != QUEUE_SCHEMA:
        raise RuntimeError("queue_state_schema_mismatch")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("queue_state_entries_not_list")

    normalized_entries: List[Dict[str, Any]] = []
    for row in entries:
        if not isinstance(row, dict):
            raise RuntimeError("queue_state_entry_not_object")
        status = str(row.get("status") or "")
        if status not in QUEUE_STATUSES:
            raise RuntimeError("queue_state_entry_invalid_status")
        normalized_entries.append(dict(row))

    out = dict(payload)
    out["entries"] = normalized_entries
    return out


def _append_event(repo_root: Path, events_log_path: Path, event: Dict[str, Any]) -> Dict[str, Any]:
    path = events_log_path if events_log_path.is_absolute() else (repo_root / events_log_path).resolve()
    if not is_within(repo_root, path):
        return {"written": False, "reason": "unsafe_events_log_path", "path": str(path)}

    try:
        if path.exists() and not path.is_file():
            return {"written": False, "reason": "events_log_not_file", "path": str(path)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(stable_json_dumps(event) + "\n")
        return {"written": True, "path": str(path)}
    except Exception as exc:
        return {"written": False, "reason": "events_log_append_failed", "error": str(exc), "path": str(path)}


def _validate_schema_payload(payload: Any, schema_path: Path, *, schema_missing_error: str) -> Tuple[bool, str, Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "gate_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists():
        return False, "gate_unavailable", {"error": schema_missing_error, "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "gate_unavailable", {"error": "schema_unreadable", "detail": str(exc)}

    if not isinstance(schema_doc, dict):
        return False, "gate_unavailable", {"error": "schema_not_object"}

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


def _validate_promotion_candidate(candidate: Any, schema_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    ok, reason, details = _validate_schema_payload(candidate, schema_path, schema_missing_error="promotion_schema_missing")
    if ok:
        return True, "", {}
    if reason == "schema_invalid":
        return False, "candidate_schema_invalid", details
    return False, reason, details


def _entry_id_from_promotion_id(promotion_id: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "_", promotion_id.lower()).strip("._-")
    if not slug:
        slug = "candidate"
    return f"kpq_{slug}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dt%H%M%S').lower()}"


def _read_candidate(candidate_path: Path) -> Dict[str, Any]:
    payload = load_json_file(candidate_path)
    if not isinstance(payload, dict):
        raise RuntimeError("candidate_not_object")
    return payload


def _entry_from_candidate(repo_root: Path, candidate_path: Path, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    promotion_id = str(candidate.get("promotion_id") or "").strip()
    if not PROMOTION_ID_RE.fullmatch(promotion_id):
        raise RuntimeError("candidate_promotion_id_invalid")

    target = candidate.get("target") if isinstance(candidate.get("target"), Mapping) else {}
    target_surface = str(target.get("surface") or "").strip()
    if target_surface not in {"doctrine", "memory", "playbook"}:
        raise RuntimeError("candidate_target_surface_invalid")

    review = candidate.get("review") if isinstance(candidate.get("review"), Mapping) else {}
    review_state = str(review.get("state") or "").strip()
    status = "QUEUED_REVIEW"
    if review_state == "approved":
        status = "APPROVED"
    elif review_state == "rejected":
        status = "REJECTED"
    elif review_state == "needs_changes":
        status = "BLOCKED"

    rel_candidate_path = candidate_path.relative_to(repo_root).as_posix()

    return {
        "schema_version": ENTRY_SCHEMA,
        "queue_entry_id": _entry_id_from_promotion_id(promotion_id),
        "enqueued_at": now_iso(),
        "updated_at": None,
        "status": status,
        "promotion_id": promotion_id,
        "candidate_path": rel_candidate_path,
        "candidate_sha256": file_sha256(candidate_path),
        "target_surface": target_surface,
        "review": {
            "state": review_state if review_state in {"pending", "approved", "rejected", "needs_changes"} else "pending",
            "reviewer_role": review.get("reviewer_role"),
            "reviewer_id": review.get("reviewer_id"),
            "reviewed_at": review.get("reviewed_at"),
            "rationale": str(review.get("rationale") or "Awaiting review"),
        },
        "workflow_decision_path": None,
        "promotion_manifest_path": None,
        "shared_memory_object_path": None,
        "last_block_reason": None,
    }


def _find_entry(entries: List[Dict[str, Any]], entry_id: str) -> Optional[Dict[str, Any]]:
    for row in entries:
        if str(row.get("queue_entry_id") or "") == entry_id:
            return row
    return None


def _run_promotion_workflow(
    *,
    repo_root: Path,
    candidate_path: Path,
    doctrine_object: Optional[str],
    publish_note_path: Optional[str],
) -> Tuple[bool, Dict[str, Any], str]:
    command = [
        sys.executable,
        str(PROMOTION_WORKFLOW_RUNNER),
        "--candidate",
        str(candidate_path),
        "--repo-root",
        str(repo_root),
        "--promotion-schema-path",
        str((repo_root / "docs" / "ops" / "schemas" / "promotion_candidate.schema.json").resolve()),
        "--doctrine-schema-path",
        str((repo_root / "docs" / "ops" / "schemas" / "doctrine_object.schema.json").resolve()),
        "--json",
    ]

    if doctrine_object:
        command.extend(["--doctrine-object", doctrine_object])
    if publish_note_path:
        command.extend(["--publish-note-path", publish_note_path])

    cp = subprocess.run(command, text=True, capture_output=True, check=False, cwd=str(DEFAULT_REPO_ROOT))

    try:
        payload = json.loads(cp.stdout.strip()) if cp.stdout.strip() else {}
    except Exception as exc:
        return False, {}, f"workflow_output_not_json:{exc}"

    if not isinstance(payload, dict):
        return False, {}, "workflow_output_not_object"

    decision = str(payload.get("decision") or "")
    if cp.returncode not in {0, 2}:
        return False, payload, f"workflow_exit_unexpected:{cp.returncode}"
    if decision not in {"PASS", "BLOCK"}:
        return False, payload, "workflow_decision_invalid"

    return True, payload, ""


def _run_shared_memory_promote(
    *,
    repo_root: Path,
    shared_memory_runner: Path,
    candidate_path: Path,
    queue_entry_id: str,
    workflow_decision_path: Path,
) -> Tuple[bool, Dict[str, Any], str]:
    command = [
        sys.executable,
        str(shared_memory_runner),
        "--repo-root",
        str(repo_root),
        "--json",
        "promote",
        "--candidate",
        str(candidate_path),
        "--queue-entry-id",
        queue_entry_id,
        "--workflow-decision-path",
        str(workflow_decision_path),
    ]

    shared_memory_env = os.environ.copy()
    shared_memory_env["OPENCLAW_INTERNAL_MUTATION"] = "1"
    shared_memory_env["OPENCLAW_INTERNAL_MUTATION_CALLSITE"] = "continuity.sh:shared-memory"

    cp = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        cwd=str(DEFAULT_REPO_ROOT),
        env=shared_memory_env,
    )

    try:
        payload = json.loads(cp.stdout.strip()) if cp.stdout.strip() else {}
    except Exception as exc:
        return False, {}, f"shared_memory_output_not_json:{exc}"

    if not isinstance(payload, dict):
        return False, {}, "shared_memory_output_not_object"

    if cp.returncode != 0 or not bool(payload.get("ok")):
        err = str(payload.get("error") or f"shared_memory_exit:{cp.returncode}")
        return False, payload, err

    return True, payload, ""


def _emit_promotion_trace_manifest(
    *,
    repo_root: Path,
    entry: Mapping[str, Any],
    candidate_path: Path,
    candidate: Mapping[str, Any],
    workflow_payload: Mapping[str, Any],
    workflow_path: Path,
    shared_memory_object_rel: Optional[str],
    trace_schema_path: Path,
) -> Tuple[bool, Optional[str], str]:
    try:
        source_rows = candidate.get("source_refs") if isinstance(candidate.get("source_refs"), list) else []
        if not source_rows:
            return False, None, "manifest_source_refs_missing"

        source_refs: List[Dict[str, Any]] = []
        for row in source_rows:
            if not isinstance(row, Mapping):
                return False, None, "manifest_source_ref_not_object"
            ref_id = str(row.get("ref_id") or "").strip()
            raw_path = str(row.get("path") or "").strip()
            declared_hash = str(row.get("content_hash") or "").strip()
            if not ref_id or not raw_path or not declared_hash:
                return False, None, "manifest_source_ref_incomplete"

            resolved = resolve_repo_path(repo_root, raw_path)
            if not is_within(repo_root, resolved):
                return False, None, "manifest_source_ref_outside_repo"
            if not resolved.exists() or not resolved.is_file():
                return False, None, "manifest_source_ref_unresolved"

            observed_hash = file_sha256(resolved)
            hash_match = normalize_sha256(declared_hash) == observed_hash
            source_refs.append(
                {
                    "ref_id": ref_id,
                    "path": resolved.relative_to(repo_root).as_posix(),
                    "declared_content_hash": declared_hash,
                    "observed_content_hash": f"sha256:{observed_hash}",
                    "hash_match": hash_match,
                }
            )

        if any(not bool(row.get("hash_match")) for row in source_refs):
            return False, None, "manifest_source_hash_mismatch"

        shared_memory_obj: Optional[Dict[str, Any]] = None
        if shared_memory_object_rel:
            sm_path = resolve_repo_path(repo_root, shared_memory_object_rel)
            if not is_within(repo_root, sm_path):
                return False, None, "manifest_shared_memory_path_outside_repo"
            if not sm_path.exists() or not sm_path.is_file():
                return False, None, "manifest_shared_memory_path_unresolved"
            shared_memory_obj = {
                "path": sm_path.relative_to(repo_root).as_posix(),
                "content_hash": f"sha256:{file_sha256(sm_path)}",
            }

        manifest_dir = repo_root / "state" / "continuity" / "knowledge_promotion_queue" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{entry['queue_entry_id']}.json"

        manifest = {
            "schema_version": "clawd.promotion_trace_manifest.v1",
            "manifest_id": f"ptm_{entry['queue_entry_id']}",
            "generated_at": now_iso(),
            "promotion_id": entry.get("promotion_id"),
            "queue_entry_id": entry.get("queue_entry_id"),
            "candidate": {
                "path": candidate_path.relative_to(repo_root).as_posix(),
                "content_hash": f"sha256:{file_sha256(candidate_path)}",
            },
            "workflow_decision": {
                "path": workflow_path.relative_to(repo_root).as_posix(),
                "content_hash": f"sha256:{file_sha256(workflow_path)}",
                "decision": str(workflow_payload.get("decision") or ""),
                "final_state": str(workflow_payload.get("final_state") or ""),
            },
            "source_refs": source_refs,
            "decision_refs": list(candidate.get("decision_refs") or []),
            "shared_memory_object": shared_memory_obj,
            "implementation_refs": list(candidate.get("implementation_refs") or []),
        }

        ok_schema, reason_schema, details_schema = _validate_schema_payload(
            manifest,
            trace_schema_path,
            schema_missing_error="promotion_trace_schema_missing",
        )
        if not ok_schema:
            return False, None, f"manifest_{reason_schema}:{details_schema}"

        atomic_write_json(manifest_path, manifest)
        return True, manifest_path.relative_to(repo_root).as_posix(), ""
    except Exception as exc:
        return False, None, f"manifest_error:{exc}"


def cmd_enqueue(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    state_path = Path(args.state_path).expanduser()
    if not state_path.is_absolute():
        state_path = (repo_root / state_path).resolve()

    events_log_path = Path(args.events_log).expanduser()
    promotion_schema_path = Path(args.promotion_schema_path).expanduser()
    if not promotion_schema_path.is_absolute():
        promotion_schema_path = (repo_root / promotion_schema_path).resolve()

    if not is_within(repo_root, state_path):
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "enqueue",
            "ok": False,
            "error": "unsafe_state_path",
            "state_path": str(state_path),
        }
    if events_log_path.is_absolute() and not is_within(repo_root, events_log_path.resolve()):
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "enqueue",
            "ok": False,
            "error": "unsafe_events_log_path",
            "state_path": str(state_path),
        }
    if not is_within(repo_root, promotion_schema_path):
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "enqueue",
            "ok": False,
            "error": "unsafe_promotion_schema_path",
            "state_path": str(state_path),
        }

    try:
        candidate_path = resolve_repo_path(repo_root, args.candidate)
        if not is_within(repo_root, candidate_path):
            raise RuntimeError("candidate_path_outside_repo")
        if not candidate_path.exists() or not candidate_path.is_file():
            raise RuntimeError("candidate_path_unresolved")

        candidate = _read_candidate(candidate_path)
        valid, reason, details = _validate_promotion_candidate(candidate, promotion_schema_path)
        if not valid:
            raise RuntimeError(reason + (f":{details}" if details else ""))

        state = _load_or_init_state(state_path)
        entries = state.get("entries") if isinstance(state.get("entries"), list) else []

        rel_candidate = candidate_path.relative_to(repo_root).as_posix()
        for row in entries:
            if str(row.get("candidate_path") or "") == rel_candidate and str(row.get("status") or "") in {
                "QUEUED_REVIEW",
                "APPROVED",
            }:
                raise RuntimeError("candidate_already_queued")

        entry = _entry_from_candidate(repo_root, candidate_path, candidate)
        entries.append(entry)
        state["entries"] = entries
        state["updated_at"] = now_iso()
        atomic_write_json(state_path, state)

        event = {
            "ts": now_iso(),
            "schema": "clawd.knowledge_promotion_queue.event.v1",
            "event": "enqueue",
            "entry_id": entry["queue_entry_id"],
            "promotion_id": entry["promotion_id"],
            "candidate_path": entry["candidate_path"],
        }
        event_record = _append_event(repo_root, events_log_path, event)

        return 0, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "enqueue",
            "ok": True,
            "entry": entry,
            "state_path": str(state_path),
            "event_record": event_record,
        }
    except Exception as exc:
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "enqueue",
            "ok": False,
            "error": str(exc),
            "state_path": str(state_path),
        }


def cmd_review(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    state_path = Path(args.state_path).expanduser()
    if not state_path.is_absolute():
        state_path = (repo_root / state_path).resolve()
    events_log_path = Path(args.events_log).expanduser()

    if not is_within(repo_root, state_path):
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "review",
            "ok": False,
            "error": "unsafe_state_path",
            "state_path": str(state_path),
        }
    if events_log_path.is_absolute() and not is_within(repo_root, events_log_path.resolve()):
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "review",
            "ok": False,
            "error": "unsafe_events_log_path",
            "state_path": str(state_path),
        }

    decision = str(args.decision or "").strip()
    if decision not in REVIEW_TO_STATUS:
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "review",
            "ok": False,
            "error": f"invalid_review_decision:{decision}",
            "state_path": str(state_path),
        }

    try:
        state = _load_or_init_state(state_path)
        entries = state.get("entries") if isinstance(state.get("entries"), list) else []
        entry = _find_entry(entries, args.entry_id)
        if entry is None:
            raise RuntimeError("entry_not_found")

        current_status = str(entry.get("status") or "")
        if current_status not in {"QUEUED_REVIEW", "APPROVED"}:
            raise RuntimeError(f"entry_not_reviewable:{current_status}")

        candidate_rel = str(entry.get("candidate_path") or "")
        candidate_path = resolve_repo_path(repo_root, candidate_rel)
        if not is_within(repo_root, candidate_path):
            raise RuntimeError("candidate_path_outside_repo")
        candidate = _read_candidate(candidate_path)

        reviewed_at = args.reviewed_at or now_iso()
        reviewer_role = args.reviewer_role
        reviewer_id = args.reviewer_id
        rationale = args.rationale or f"Queue review decision: {decision}"

        review_obj = candidate.get("review") if isinstance(candidate.get("review"), dict) else {}
        review_obj["state"] = decision
        review_obj["reviewer_role"] = reviewer_role
        review_obj["reviewer_id"] = reviewer_id
        review_obj["reviewed_at"] = reviewed_at
        review_obj["rationale"] = rationale
        candidate["review"] = review_obj
        candidate["promotion_state"] = REVIEW_TO_PROMOTION_STATE[decision]

        atomic_write_json(candidate_path, candidate)

        entry["status"] = REVIEW_TO_STATUS[decision]
        entry["updated_at"] = now_iso()
        entry["candidate_sha256"] = file_sha256(candidate_path)
        entry["review"] = {
            "state": decision,
            "reviewer_role": reviewer_role,
            "reviewer_id": reviewer_id,
            "reviewed_at": reviewed_at,
            "rationale": rationale,
        }
        if decision == "approved":
            entry["last_block_reason"] = None

        state["updated_at"] = now_iso()
        atomic_write_json(state_path, state)

        event = {
            "ts": now_iso(),
            "schema": "clawd.knowledge_promotion_queue.event.v1",
            "event": "review",
            "entry_id": entry["queue_entry_id"],
            "promotion_id": entry.get("promotion_id"),
            "decision": decision,
            "status": entry["status"],
        }
        event_record = _append_event(repo_root, events_log_path, event)

        return 0, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "review",
            "ok": True,
            "entry": entry,
            "state_path": str(state_path),
            "event_record": event_record,
        }
    except Exception as exc:
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "review",
            "ok": False,
            "error": str(exc),
            "state_path": str(state_path),
        }


def cmd_promote(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    state_path = Path(args.state_path).expanduser()
    if not state_path.is_absolute():
        state_path = (repo_root / state_path).resolve()
    events_log_path = Path(args.events_log).expanduser()

    if not is_within(repo_root, state_path):
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "promote",
            "ok": False,
            "error": "unsafe_state_path",
            "state_path": str(state_path),
        }
    if events_log_path.is_absolute() and not is_within(repo_root, events_log_path.resolve()):
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "promote",
            "ok": False,
            "error": "unsafe_events_log_path",
            "state_path": str(state_path),
        }

    trace_schema_path = Path(args.promotion_trace_schema_path).expanduser()
    if not trace_schema_path.is_absolute():
        trace_schema_path = (repo_root / trace_schema_path).resolve()
    if not is_within(repo_root, trace_schema_path):
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "promote",
            "ok": False,
            "error": "unsafe_promotion_trace_schema_path",
            "state_path": str(state_path),
        }

    try:
        state = _load_or_init_state(state_path)
        entries = state.get("entries") if isinstance(state.get("entries"), list) else []
        entry = _find_entry(entries, args.entry_id)
        if entry is None:
            raise RuntimeError("entry_not_found")

        current_status = str(entry.get("status") or "")
        if current_status != "APPROVED":
            raise RuntimeError(f"entry_not_promotable:{current_status}")

        candidate_rel = str(entry.get("candidate_path") or "")
        candidate_path = resolve_repo_path(repo_root, candidate_rel)
        if not is_within(repo_root, candidate_path):
            raise RuntimeError("candidate_path_outside_repo")

        candidate = _read_candidate(candidate_path)

        ok, workflow_payload, workflow_err = _run_promotion_workflow(
            repo_root=repo_root,
            candidate_path=candidate_path,
            doctrine_object=args.doctrine_object,
            publish_note_path=args.publish_note_path,
        )
        if not ok:
            raise RuntimeError(workflow_err)

        workflow_dir = repo_root / "state" / "continuity" / "knowledge_promotion_queue" / "workflow"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        workflow_path = workflow_dir / f"{entry['queue_entry_id']}.json"
        atomic_write_json(workflow_path, workflow_payload)

        decision = str(workflow_payload.get("decision") or "")
        shared_memory_rel: Optional[str] = None
        if decision == "PASS" and str(entry.get("target_surface") or "") == "memory":
            shared_memory_runner = Path(args.shared_memory_runner).expanduser()
            if not shared_memory_runner.is_absolute():
                shared_memory_runner = (repo_root / shared_memory_runner).resolve()
            if not shared_memory_runner.exists() or not shared_memory_runner.is_file():
                shared_memory_runner = DEFAULT_SHARED_MEMORY_RUNNER

            sm_ok, sm_payload, sm_err = _run_shared_memory_promote(
                repo_root=repo_root,
                shared_memory_runner=shared_memory_runner,
                candidate_path=candidate_path,
                queue_entry_id=str(entry.get("queue_entry_id") or ""),
                workflow_decision_path=workflow_path,
            )
            if not sm_ok:
                decision = "BLOCK"
                workflow_payload = dict(workflow_payload)
                workflow_payload["decision"] = "BLOCK"
                workflow_payload["final_state"] = "BLOCKED"
                workflow_payload["block_stage"] = "shared_memory_fabric"
                workflow_payload["block_reason"] = sm_err or "shared_memory_fabric_failed"
                workflow_payload["shared_memory_payload"] = sm_payload
                atomic_write_json(workflow_path, workflow_payload)
            else:
                sm_object_path = str(sm_payload.get("object_path") or "").strip()
                if sm_object_path:
                    sm_object = Path(sm_object_path).expanduser().resolve()
                    if is_within(repo_root, sm_object):
                        shared_memory_rel = sm_object.relative_to(repo_root).as_posix()

        manifest_ok = False
        manifest_rel: Optional[str] = None
        manifest_err = ""
        if decision == "PASS":
            manifest_ok, manifest_rel, manifest_err = _emit_promotion_trace_manifest(
                repo_root=repo_root,
                entry=entry,
                candidate_path=candidate_path,
                candidate=candidate,
                workflow_payload=workflow_payload,
                workflow_path=workflow_path,
                shared_memory_object_rel=shared_memory_rel,
                trace_schema_path=trace_schema_path,
            )
            if not manifest_ok:
                decision = "BLOCK"
                workflow_payload = dict(workflow_payload)
                workflow_payload["decision"] = "BLOCK"
                workflow_payload["final_state"] = "BLOCKED"
                workflow_payload["block_stage"] = "promotion_trace_manifest"
                workflow_payload["block_reason"] = manifest_err or "promotion_trace_manifest_failed"
                atomic_write_json(workflow_path, workflow_payload)

        if decision == "PASS":
            entry["status"] = "PROMOTED"
            entry["last_block_reason"] = None
            rc = 0
        else:
            entry["status"] = "BLOCKED"
            entry["last_block_reason"] = str(workflow_payload.get("block_reason") or "workflow_blocked")
            rc = 2

        entry["updated_at"] = now_iso()
        entry["workflow_decision_path"] = workflow_path.relative_to(repo_root).as_posix()
        entry["promotion_manifest_path"] = manifest_rel
        entry["shared_memory_object_path"] = shared_memory_rel

        state["updated_at"] = now_iso()
        atomic_write_json(state_path, state)

        event = {
            "ts": now_iso(),
            "schema": "clawd.knowledge_promotion_queue.event.v1",
            "event": "promote",
            "entry_id": entry["queue_entry_id"],
            "promotion_id": entry.get("promotion_id"),
            "decision": decision,
            "status": entry["status"],
            "workflow_decision_path": entry["workflow_decision_path"],
            "promotion_manifest_path": entry.get("promotion_manifest_path"),
            "shared_memory_object_path": entry.get("shared_memory_object_path"),
        }
        event_record = _append_event(repo_root, events_log_path, event)

        return rc, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "promote",
            "ok": decision == "PASS",
            "entry": entry,
            "workflow_decision": workflow_payload,
            "state_path": str(state_path),
            "event_record": event_record,
        }

    except Exception as exc:
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "promote",
            "ok": False,
            "error": str(exc),
            "state_path": str(state_path),
        }


def cmd_list(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    state_path = Path(args.state_path).expanduser()
    if not state_path.is_absolute():
        state_path = (repo_root / state_path).resolve()

    if not is_within(repo_root, state_path):
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "list",
            "ok": False,
            "error": "unsafe_state_path",
            "state_path": str(state_path),
        }

    try:
        state = _load_or_init_state(state_path)
        entries = state.get("entries") if isinstance(state.get("entries"), list) else []
        if args.status:
            entries = [row for row in entries if str(row.get("status") or "") == args.status]

        return 0, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "list",
            "ok": True,
            "state_path": str(state_path),
            "entries": entries,
            "count": len(entries),
        }
    except Exception as exc:
        return 2, {
            "schema": "clawd.knowledge_promotion_queue.result.v1",
            "action": "list",
            "ok": False,
            "error": str(exc),
            "state_path": str(state_path),
        }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Knowledge review/approval/promotion queue runtime (v1)")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--state-path", default=str(DEFAULT_STATE_PATH), help="Queue state JSON path")
    ap.add_argument("--events-log", default=str(DEFAULT_EVENTS_LOG), help="Queue events JSONL path")
    ap.add_argument(
        "--promotion-schema-path",
        default=str(DEFAULT_PROMOTION_SCHEMA_PATH),
        help="Promotion candidate schema path",
    )
    ap.add_argument(
        "--promotion-trace-schema-path",
        default=str(DEFAULT_PROMOTION_TRACE_SCHEMA_PATH),
        help="Promotion trace manifest schema path",
    )
    ap.add_argument(
        "--shared-memory-runner",
        default=str(DEFAULT_SHARED_MEMORY_RUNNER),
        help="Shared memory fabric runtime path",
    )
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON")

    sub = ap.add_subparsers(dest="command", required=True)

    p_enqueue = sub.add_parser("enqueue", help="enqueue promotion candidate")
    p_enqueue.add_argument("--candidate", required=True, help="Promotion candidate JSON path")
    p_enqueue.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_review = sub.add_parser("review", help="record review decision")
    p_review.add_argument("--entry-id", required=True, help="Queue entry id")
    p_review.add_argument("--decision", required=True, choices=sorted(REVIEW_TO_STATUS.keys()))
    p_review.add_argument("--reviewer-role", required=True)
    p_review.add_argument("--reviewer-id", required=True)
    p_review.add_argument("--reviewed-at", default="")
    p_review.add_argument("--rationale", default="")
    p_review.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_promote = sub.add_parser("promote", help="run promotion workflow for approved entry")
    p_promote.add_argument("--entry-id", required=True, help="Queue entry id")
    p_promote.add_argument("--doctrine-object", default="", help="Optional doctrine object path override")
    p_promote.add_argument("--publish-note-path", default="", help="Optional publish note path override")
    p_promote.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_list = sub.add_parser("list", help="list queue state")
    p_list.add_argument("--status", default="", choices=[""] + sorted(QUEUE_STATUSES), help="Optional status filter")
    p_list.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    wrapper_guard_error = _enforce_wrapper_only_contract(str(args.command or ""))
    if wrapper_guard_error is not None:
        print(json.dumps(wrapper_guard_error, ensure_ascii=False, indent=2 if args.json else None))
        return 2

    if args.command == "enqueue":
        rc, payload = cmd_enqueue(args)
    elif args.command == "review":
        rc, payload = cmd_review(args)
    elif args.command == "promote":
        # normalize optional empty strings -> None
        args.doctrine_object = args.doctrine_object or None
        args.publish_note_path = args.publish_note_path or None
        rc, payload = cmd_promote(args)
    elif args.command == "list":
        rc, payload = cmd_list(args)
    else:
        rc, payload = 2, {"ok": False, "error": f"unknown_command:{args.command}"}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(payload if isinstance(payload, dict) else {"payload": payload}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
