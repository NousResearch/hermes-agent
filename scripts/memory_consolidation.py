#!/usr/bin/env python3
"""Governed background memory consolidation runtime (MEM-02 bounded slice).

Scope (bounded):
- consolidate historical daily memory files (memory/YYYY-MM-DD.md)
- produce auditable consolidated artifact with provenance
- archive originals (never delete)
- append consolidation ledger rows
- support deterministic rollback using ledger provenance

Fail-closed governance gate:
- consolidation only allowed when continuity readiness=READY
- mutation_gate.status must be allowed
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_MEMORY_ROOT = Path("memory")
DEFAULT_LEDGER_PATH = Path("memory/consolidation_ledger.jsonl")
DEFAULT_ARCHIVE_ROOT = Path("memory/archive")
DEFAULT_CONSOLIDATED_ROOT = Path("memory/consolidated")
DEFAULT_RUNTIME_LATEST = Path("state/continuity/latest/memory_consolidation_latest.json")
DEFAULT_CONTINUITY_CURRENT = Path("state/continuity/current.json")
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_SOURCE_BYTES = 200_000
DEFAULT_OLDER_THAN_DAYS = 1

RESULT_SCHEMA = "clawd.memory_consolidation.result.v1"
ARTIFACT_SCHEMA = "clawd.memory_consolidation.artifact.v1"
LEDGER_SCHEMA = "clawd.memory_consolidation.ledger_entry.v1"
RUNTIME_SCHEMA = "clawd.memory_consolidation.runtime_latest.v1"
SCRIPT_VERSION = "mem_consolidation_v2_2026_04_04"

GATE_GOVERNANCE = "governance_gate"
GATE_DISCOVERY_ELIGIBILITY = "discovery_eligibility_gate"
GATE_SOURCE_VALIDATION = "source_validation_gate"
GATE_POST_ARTIFACT_VALIDATION = "post_artifact_validation_gate"
GATE_ROLLBACK_VALIDATION = "rollback_validation_gate"

FAILURE_CODE_GOVERNANCE = "FAILED_GOVERNANCE_GATE"
FAILURE_CODE_SOURCE_VALIDATION = "FAILED_SOURCE_VALIDATION"
FAILURE_CODE_ARCHIVE_MOVE = "FAILED_ARCHIVE_MOVE"
FAILURE_CODE_POST_ARTIFACT_VALIDATION = "FAILED_POST_ARTIFACT_VALIDATION"
FAILURE_CODE_ROLLBACK_VALIDATION = "FAILED_ROLLBACK_VALIDATION"

DATE_FILE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.md$")


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def now_iso() -> str:
    return now_utc().isoformat().replace("+00:00", "Z")


def stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(stable_json(dict(row)) + "\n")


def parse_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def is_within(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def resolve_repo_path(repo_root: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def safe_repo_path(repo_root: Path, raw_path: str | Path) -> Tuple[bool, Path, Optional[str]]:
    try:
        resolved = resolve_repo_path(repo_root, raw_path)
    except Exception as exc:
        return False, repo_root, f"path_resolve_failed:{exc}"
    if not is_within(repo_root, resolved):
        return False, resolved, "path_outside_repo"
    return True, resolved, None


def safe_rel(repo_root: Path, target: Path) -> str:
    return target.resolve().relative_to(repo_root.resolve()).as_posix()


def daily_file_date(path: Path) -> Optional[dt.date]:
    m = DATE_FILE_RE.fullmatch(path.name)
    if not m:
        return None
    try:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


def summarize_text(text: str, max_chars: int = 280) -> Tuple[str, str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "(empty)", ""

    title = ""
    for ln in lines:
        if ln.startswith("#"):
            title = ln.lstrip("#").strip()
            break
    if not title:
        title = lines[0]

    body: List[str] = []
    for ln in lines:
        if ln.startswith("#"):
            continue
        body.append(ln)
        if len(body) >= 3:
            break
    summary = " ".join(body).strip() if body else title
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return title[:160], summary


def governance_gate(continuity_payload: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    readiness = str(continuity_payload.get("readiness") or "")
    mutation_gate = continuity_payload.get("mutation_gate") if isinstance(continuity_payload.get("mutation_gate"), dict) else {}
    mutation_status = str(mutation_gate.get("status") or "")

    ok = readiness == "READY" and mutation_status == "allowed"
    details = {
        "readiness": readiness,
        "mutation_gate_status": mutation_status,
        "ok": ok,
    }
    if not ok:
        reasons = []
        if readiness != "READY":
            reasons.append("readiness_not_ready")
        if mutation_status != "allowed":
            reasons.append("mutation_gate_not_allowed")
        details["blocking_reasons"] = reasons
    return ok, details


def discover_candidates(memory_root: Path, *, older_than_days: int) -> List[Tuple[dt.date, Path]]:
    today = now_utc().date()
    cutoff = today - dt.timedelta(days=max(0, int(older_than_days)))

    found: List[Tuple[dt.date, Path]] = []
    if not memory_root.exists() or not memory_root.is_dir():
        return found

    for path in memory_root.iterdir():
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        d = daily_file_date(path)
        if d is None:
            continue
        if d <= cutoff:
            found.append((d, path))

    found.sort(key=lambda row: (row[0], row[1].name))
    return found


def _required_artifact_fields_ok(payload: Mapping[str, Any]) -> bool:
    required = [
        "schema",
        "batch_id",
        "generated_at",
        "script_version",
        "strategy",
        "governance",
        "memory_md_provenance",
        "source_files",
        "consolidated_entries",
    ]
    for key in required:
        if key not in payload:
            return False
    if payload.get("schema") != ARTIFACT_SCHEMA:
        return False
    if not isinstance(payload.get("source_files"), list) or not payload.get("source_files"):
        return False
    if not isinstance(payload.get("consolidated_entries"), list) or not payload.get("consolidated_entries"):
        return False
    return True


def _restore_moves(moves: Iterable[Tuple[Path, Path]]) -> None:
    for src, archived in reversed(list(moves)):
        if archived.exists() and not src.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(archived), str(src))


def _event_id(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _previous_source_sha_by_path(rows: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    by_path: Dict[str, str] = {}
    for row in rows:
        if str(row.get("event_type") or "") != "CONSOLIDATION_APPLIED":
            continue
        inputs = row.get("inputs") if isinstance(row.get("inputs"), dict) else {}
        files = inputs.get("files") if isinstance(inputs.get("files"), list) else []
        for file_row in files:
            if not isinstance(file_row, dict):
                continue
            source_path = str(file_row.get("source_path") or "").strip()
            source_sha256 = str(file_row.get("source_sha256") or "").strip()
            if source_path and source_sha256:
                by_path[source_path] = source_sha256
    return by_path


def _validate_artifact_and_archive(
    *,
    repo_root: Path,
    artifact_path: Path,
    source_files: Iterable[Mapping[str, Any]],
) -> Tuple[bool, str]:
    if not artifact_path.exists() or not artifact_path.is_file():
        return False, "artifact_missing"
    if max(0, int(artifact_path.stat().st_size)) <= 0:
        return False, "artifact_empty"

    try:
        parsed_artifact = parse_json(artifact_path)
    except Exception as exc:
        return False, f"artifact_parse_failed:{exc}"

    if not isinstance(parsed_artifact, dict) or not _required_artifact_fields_ok(parsed_artifact):
        return False, "artifact_schema_invalid"

    for row in source_files:
        source_path = str(row.get("source_path") or "").strip()
        archived_rel = str(row.get("archived_path") or "").strip()
        expected_sha = str(row.get("source_sha256") or "").strip()
        if not source_path or not archived_rel or not expected_sha:
            return False, "archive_provenance_row_invalid"

        archived_abs = resolve_repo_path(repo_root, archived_rel)
        if not archived_abs.exists() or not archived_abs.is_file():
            return False, f"archived_file_missing:{archived_rel}"

        archived_sha = f"sha256:{file_sha256(archived_abs)}"
        if archived_sha != expected_sha:
            return False, f"archived_hash_mismatch:{source_path}"

    return True, "ok"


def cmd_run(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()

    for raw in [
        args.memory_root,
        args.ledger_path,
        args.archive_root,
        args.consolidated_root,
        args.runtime_latest_path,
        args.continuity_current_path,
    ]:
        ok, _, reason = safe_repo_path(repo_root, raw)
        if not ok:
            return 2, {"schema": RESULT_SCHEMA, "action": "run", "ok": False, "error": reason}

    memory_root = resolve_repo_path(repo_root, args.memory_root)
    ledger_path = resolve_repo_path(repo_root, args.ledger_path)
    archive_root = resolve_repo_path(repo_root, args.archive_root)
    consolidated_root = resolve_repo_path(repo_root, args.consolidated_root)
    runtime_latest_path = resolve_repo_path(repo_root, args.runtime_latest_path)
    continuity_current_path = resolve_repo_path(repo_root, args.continuity_current_path)

    validation_gates_passed: List[str] = []

    def fail_run(
        *,
        error: str,
        failure_code: str,
        gate_name: str,
        detail: Optional[str] = None,
        governance: Optional[Mapping[str, Any]] = None,
        status: str = "failed",
        batch_id: Optional[str] = None,
        candidate_count: Optional[int] = None,
        file_count: Optional[int] = None,
        source_bytes: Optional[int] = None,
        write_runtime_latest: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        failure = {
            "gate": gate_name,
            "error": error,
            "failure_code": failure_code,
        }
        if detail:
            failure["detail"] = detail

        if write_runtime_latest:
            runtime_payload: Dict[str, Any] = {
                "schema": RUNTIME_SCHEMA,
                "updated_at": now_iso(),
                "status": status,
                "reason": error,
                "failure_code": failure_code,
                "validation_gates_passed": list(validation_gates_passed),
                "last_validation_gate_failure": failure,
                "ledger_path": safe_rel(repo_root, ledger_path),
            }
            if governance is not None:
                runtime_payload["governance"] = dict(governance)
            if batch_id:
                runtime_payload["batch_id"] = batch_id
            if candidate_count is not None:
                runtime_payload["candidate_count"] = max(0, int(candidate_count))
            if file_count is not None:
                runtime_payload["file_count"] = max(0, int(file_count))
            if source_bytes is not None:
                runtime_payload["source_bytes"] = max(0, int(source_bytes))
            atomic_write_json(runtime_latest_path, runtime_payload)

        payload: Dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "action": "run",
            "ok": False,
            "error": error,
            "failure_code": failure_code,
            "validation_gates_passed": list(validation_gates_passed),
            "last_validation_gate_failure": failure,
        }
        if governance is not None:
            payload["governance"] = dict(governance)
        if batch_id:
            payload["batch_id"] = batch_id
        if candidate_count is not None:
            payload["candidate_count"] = max(0, int(candidate_count))
        if file_count is not None:
            payload["file_count"] = max(0, int(file_count))
        if source_bytes is not None:
            payload["source_bytes"] = max(0, int(source_bytes))
        return 2, payload

    if not continuity_current_path.exists() or not continuity_current_path.is_file():
        rc, payload = fail_run(
            error="continuity_current_missing",
            failure_code=FAILURE_CODE_GOVERNANCE,
            gate_name=GATE_GOVERNANCE,
            detail=str(continuity_current_path),
        )
        payload["continuity_current_path"] = str(continuity_current_path)
        return rc, payload

    continuity = parse_json(continuity_current_path)
    if not isinstance(continuity, dict):
        return fail_run(
            error="continuity_current_not_object",
            failure_code=FAILURE_CODE_GOVERNANCE,
            gate_name=GATE_GOVERNANCE,
        )

    gate_ok, gate = governance_gate(continuity)
    if not gate_ok:
        return fail_run(
            error="governance_gate_blocked",
            failure_code=FAILURE_CODE_GOVERNANCE,
            gate_name=GATE_GOVERNANCE,
            governance=gate,
            status="blocked_governance_gate",
        )
    validation_gates_passed.append(GATE_GOVERNANCE)

    discovered = discover_candidates(memory_root, older_than_days=args.older_than_days)
    validation_gates_passed.append(GATE_DISCOVERY_ELIGIBILITY)
    if not discovered:
        payload = {
            "schema": RESULT_SCHEMA,
            "action": "run",
            "ok": True,
            "status": "no_op",
            "reason": "no_eligible_daily_memory_files",
            "governance": gate,
            "candidate_count": 0,
            "validation_gates_passed": list(validation_gates_passed),
        }
        atomic_write_json(
            runtime_latest_path,
            {
                "schema": RUNTIME_SCHEMA,
                "updated_at": now_iso(),
                "status": "no_op",
                "reason": payload["reason"],
                "candidate_count": 0,
                "validation_gates_passed": list(validation_gates_passed),
                "last_validation_gate_failure": None,
                "ledger_path": safe_rel(repo_root, ledger_path),
            },
        )
        return 0, payload

    selected: List[Tuple[dt.date, Path]] = []
    total_bytes = 0
    max_bytes = max(1, int(args.max_source_bytes))
    for d, path in discovered:
        if len(selected) >= max(1, int(args.batch_size)):
            break
        file_size = max(0, int(path.stat().st_size))
        if selected and (total_bytes + file_size > max_bytes):
            break
        if not selected and file_size > max_bytes:
            rc, payload = fail_run(
                error="batch_byte_budget_exceeded",
                failure_code=FAILURE_CODE_SOURCE_VALIDATION,
                gate_name=GATE_SOURCE_VALIDATION,
                governance=gate,
                candidate_count=len(discovered),
            )
            payload["max_source_bytes"] = max_bytes
            payload["first_candidate"] = safe_rel(repo_root, path)
            payload["first_candidate_bytes"] = file_size
            return rc, payload
        selected.append((d, path))
        total_bytes += file_size

    if not selected:
        return fail_run(
            error="no_files_selected_after_budgeting",
            failure_code=FAILURE_CODE_SOURCE_VALIDATION,
            gate_name=GATE_SOURCE_VALIDATION,
            governance=gate,
            candidate_count=len(discovered),
        )

    ts = now_utc().strftime("%Y%m%dt%H%M%SZ").lower()
    batch_seed = "|".join(safe_rel(repo_root, p) for _, p in selected)
    batch_digest = hashlib.sha256((batch_seed + "|" + now_iso()).encode("utf-8")).hexdigest()[:8]
    batch_id = f"mcb_{ts}_{batch_digest}"

    archive_batch_root = archive_root / batch_id
    artifact_path = consolidated_root / f"{batch_id}.json"

    source_files: List[Dict[str, Any]] = []
    consolidated_entries: List[Dict[str, Any]] = []

    for _, src in selected:
        if not src.exists() or not src.is_file():
            return fail_run(
                error="source_missing_before_read",
                failure_code=FAILURE_CODE_SOURCE_VALIDATION,
                gate_name=GATE_SOURCE_VALIDATION,
                governance=gate,
                detail=safe_rel(repo_root, src),
                batch_id=batch_id,
                candidate_count=len(discovered),
            )
        text = src.read_text(encoding="utf-8", errors="replace")
        title, summary = summarize_text(text)
        lines = text.splitlines()
        src_rel = safe_rel(repo_root, src)
        archived_rel = safe_rel(repo_root, archive_batch_root / src.name)

        source_files.append(
            {
                "source_path": src_rel,
                "archived_path": archived_rel,
                "source_sha256": f"sha256:{file_sha256(src)}",
                "source_bytes": max(0, int(src.stat().st_size)),
                "source_line_count": len(lines),
            }
        )
        consolidated_entries.append(
            {
                "source_path": src_rel,
                "title": title,
                "summary": summary,
                "line_count": len(lines),
            }
        )

    previous_source_sha = _previous_source_sha_by_path(parse_jsonl(ledger_path))
    unchanged_files: List[Dict[str, Any]] = []
    for row in source_files:
        src_rel = str(row.get("source_path") or "")
        src_sha = str(row.get("source_sha256") or "")
        if previous_source_sha.get(src_rel) == src_sha:
            unchanged_files.append({"source_path": src_rel, "source_sha256": src_sha})

    if args.fault_inject_source_hash_mismatch and source_files:
        source_files[0]["source_sha256"] = "sha256:FAULT_INJECTED_MISMATCH"

    memory_md_path = repo_root / "MEMORY.md"
    memory_md_prov: Dict[str, Any]
    if memory_md_path.exists() and memory_md_path.is_file():
        memory_md_prov = {
            "path": safe_rel(repo_root, memory_md_path),
            "sha256": f"sha256:{file_sha256(memory_md_path)}",
            "bytes": max(0, int(memory_md_path.stat().st_size)),
        }
    else:
        memory_md_prov = {"path": "MEMORY.md", "present": False}

    continuity_sha = f"sha256:{file_sha256(continuity_current_path)}"

    artifact_payload: Dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "batch_id": batch_id,
        "generated_at": now_iso(),
        "script_version": SCRIPT_VERSION,
        "strategy": {
            "batch_size": max(1, int(args.batch_size)),
            "max_source_bytes": max_bytes,
            "older_than_days": max(0, int(args.older_than_days)),
            "summarization": "heading_plus_first_lines_v1",
        },
        "governance": {
            **gate,
            "continuity_current_path": safe_rel(repo_root, continuity_current_path),
            "continuity_current_sha256": continuity_sha,
        },
        "memory_md_provenance": memory_md_prov,
        "source_files": source_files,
        "consolidated_entries": consolidated_entries,
        "unchanged_files": unchanged_files,
    }

    if args.fault_inject_invalid_artifact:
        artifact_payload.pop("source_files", None)

    for row in source_files:
        src_rel = str(row.get("source_path") or "")
        expected_sha = str(row.get("source_sha256") or "")
        src_abs = resolve_repo_path(repo_root, src_rel)
        if not src_abs.exists() or not src_abs.is_file():
            return fail_run(
                error="source_missing_before_archive",
                failure_code=FAILURE_CODE_SOURCE_VALIDATION,
                gate_name=GATE_SOURCE_VALIDATION,
                governance=gate,
                detail=src_rel,
                batch_id=batch_id,
                candidate_count=len(discovered),
                file_count=len(source_files),
                source_bytes=total_bytes,
            )
        actual_sha = f"sha256:{file_sha256(src_abs)}"
        if expected_sha != actual_sha:
            return fail_run(
                error="source_hash_mismatch_before_archive",
                failure_code=FAILURE_CODE_SOURCE_VALIDATION,
                gate_name=GATE_SOURCE_VALIDATION,
                governance=gate,
                detail=src_rel,
                batch_id=batch_id,
                candidate_count=len(discovered),
                file_count=len(source_files),
                source_bytes=total_bytes,
            )
    validation_gates_passed.append(GATE_SOURCE_VALIDATION)

    if args.dry_run:
        payload = {
            "schema": RESULT_SCHEMA,
            "action": "run",
            "ok": True,
            "status": "dry_run",
            "batch_id": batch_id,
            "governance": gate,
            "candidate_count": len(discovered),
            "file_count": len(source_files),
            "source_bytes": total_bytes,
            "selected_source_paths": [str(row.get("source_path")) for row in source_files],
            "unchanged_file_count": len(unchanged_files),
            "validation_gates_passed": list(validation_gates_passed),
        }
        atomic_write_json(
            runtime_latest_path,
            {
                "schema": RUNTIME_SCHEMA,
                "updated_at": now_iso(),
                "status": "dry_run",
                "batch_id": batch_id,
                "candidate_count": len(discovered),
                "file_count": len(source_files),
                "source_bytes": total_bytes,
                "selected_source_paths": payload["selected_source_paths"],
                "unchanged_file_count": len(unchanged_files),
                "validation_gates_passed": list(validation_gates_passed),
                "last_validation_gate_failure": None,
                "ledger_path": safe_rel(repo_root, ledger_path),
            },
        )
        return 0, payload

    atomic_write_json(artifact_path, artifact_payload)

    moved: List[Tuple[Path, Path]] = []
    try:
        for row in source_files:
            src_abs = resolve_repo_path(repo_root, str(row["source_path"]))
            archived_abs = resolve_repo_path(repo_root, str(row["archived_path"]))
            archived_abs.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_abs), str(archived_abs))
            moved.append((src_abs, archived_abs))
    except Exception as exc:
        _restore_moves(moved)
        if artifact_path.exists():
            artifact_path.unlink()
        return fail_run(
            error="archive_move_failed",
            failure_code=FAILURE_CODE_ARCHIVE_MOVE,
            gate_name=GATE_SOURCE_VALIDATION,
            governance=gate,
            detail=str(exc),
            status="failed_rolled_back",
            batch_id=batch_id,
            candidate_count=len(discovered),
            file_count=len(source_files),
            source_bytes=total_bytes,
        )

    artifact_ok, artifact_reason = _validate_artifact_and_archive(
        repo_root=repo_root,
        artifact_path=artifact_path,
        source_files=source_files,
    )
    if not artifact_ok:
        _restore_moves(moved)
        if artifact_path.exists():
            artifact_path.unlink()

        failure_event = {
            "schema": LEDGER_SCHEMA,
            "event_type": "CONSOLIDATION_FAILED_ROLLED_BACK",
            "recorded_at": now_iso(),
            "batch_id": batch_id,
            "script_version": SCRIPT_VERSION,
            "status": "failed_rolled_back",
            "reason": "post_artifact_validation_failed",
            "failure_code": FAILURE_CODE_POST_ARTIFACT_VALIDATION,
            "validation_gates_passed": list(validation_gates_passed),
            "last_validation_gate_failure": {
                "gate": GATE_POST_ARTIFACT_VALIDATION,
                "error": "post_artifact_validation_failed",
                "failure_code": FAILURE_CODE_POST_ARTIFACT_VALIDATION,
                "detail": artifact_reason,
            },
            "inputs": {
                "files": source_files,
                "file_count": len(source_files),
                "source_bytes": total_bytes,
            },
            "outputs": {
                "artifact_path": safe_rel(repo_root, artifact_path),
                "artifact_present_after_rollback": artifact_path.exists(),
            },
        }
        failure_event["event_id"] = _event_id(failure_event)
        append_jsonl(ledger_path, failure_event)

        atomic_write_json(
            runtime_latest_path,
            {
                "schema": RUNTIME_SCHEMA,
                "updated_at": now_iso(),
                "status": "failed_rolled_back",
                "batch_id": batch_id,
                "reason": "post_artifact_validation_failed",
                "failure_code": FAILURE_CODE_POST_ARTIFACT_VALIDATION,
                "validation_gates_passed": list(validation_gates_passed),
                "last_validation_gate_failure": {
                    "gate": GATE_POST_ARTIFACT_VALIDATION,
                    "error": "post_artifact_validation_failed",
                    "failure_code": FAILURE_CODE_POST_ARTIFACT_VALIDATION,
                    "detail": artifact_reason,
                },
                "ledger_path": safe_rel(repo_root, ledger_path),
            },
        )

        return 2, {
            "schema": RESULT_SCHEMA,
            "action": "run",
            "ok": False,
            "error": "post_artifact_validation_failed",
            "failure_code": FAILURE_CODE_POST_ARTIFACT_VALIDATION,
            "batch_id": batch_id,
            "validation_gates_passed": list(validation_gates_passed),
            "last_validation_gate_failure": {
                "gate": GATE_POST_ARTIFACT_VALIDATION,
                "error": "post_artifact_validation_failed",
                "failure_code": FAILURE_CODE_POST_ARTIFACT_VALIDATION,
                "detail": artifact_reason,
            },
        }

    validation_gates_passed.append(GATE_POST_ARTIFACT_VALIDATION)

    artifact_sha = f"sha256:{file_sha256(artifact_path)}"

    ledger_entry: Dict[str, Any] = {
        "schema": LEDGER_SCHEMA,
        "event_type": "CONSOLIDATION_APPLIED",
        "recorded_at": now_iso(),
        "batch_id": batch_id,
        "script_version": SCRIPT_VERSION,
        "status": "applied",
        "validation_gates_passed": list(validation_gates_passed),
        "governance": {
            **gate,
            "continuity_current_path": safe_rel(repo_root, continuity_current_path),
            "continuity_current_sha256": continuity_sha,
        },
        "inputs": {
            "files": source_files,
            "file_count": len(source_files),
            "source_bytes": total_bytes,
        },
        "outputs": {
            "artifact_path": safe_rel(repo_root, artifact_path),
            "artifact_sha256": artifact_sha,
            "archive_root": safe_rel(repo_root, archive_batch_root),
            "unchanged_file_count": len(unchanged_files),
        },
        "rollback": {
            "command": f"python3 scripts/memory_consolidation.py rollback --batch-id {batch_id} --json",
            "policy": "deterministic_from_ledger",
        },
    }
    ledger_entry["event_id"] = _event_id(ledger_entry)
    append_jsonl(ledger_path, ledger_entry)

    runtime_payload = {
        "schema": RUNTIME_SCHEMA,
        "updated_at": now_iso(),
        "status": "applied",
        "batch_id": batch_id,
        "artifact_path": safe_rel(repo_root, artifact_path),
        "artifact_sha256": artifact_sha,
        "file_count": len(source_files),
        "source_bytes": total_bytes,
        "candidate_count": len(discovered),
        "unchanged_file_count": len(unchanged_files),
        "validation_gates_passed": list(validation_gates_passed),
        "last_validation_gate_failure": None,
        "ledger_path": safe_rel(repo_root, ledger_path),
    }
    atomic_write_json(runtime_latest_path, runtime_payload)

    return 0, {
        "schema": RESULT_SCHEMA,
        "action": "run",
        "ok": True,
        "status": "applied",
        "batch_id": batch_id,
        "governance": gate,
        "file_count": len(source_files),
        "source_bytes": total_bytes,
        "candidate_count": len(discovered),
        "unchanged_file_count": len(unchanged_files),
        "validation_gates_passed": list(validation_gates_passed),
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha,
        "ledger_event_id": ledger_entry["event_id"],
        "ledger_path": str(ledger_path),
    }


def _latest_applied_entry_for_batch(rows: List[Dict[str, Any]], batch_id: str) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    for idx in range(len(rows) - 1, -1, -1):
        row = rows[idx]
        if str(row.get("batch_id") or "") != batch_id:
            continue
        if str(row.get("event_type") or "") == "CONSOLIDATION_APPLIED":
            return idx, row
    return None, None


def cmd_rollback(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()

    for raw in [args.ledger_path, args.runtime_latest_path]:
        ok, _, reason = safe_repo_path(repo_root, raw)
        if not ok:
            return 2, {"schema": RESULT_SCHEMA, "action": "rollback", "ok": False, "error": reason}

    ledger_path = resolve_repo_path(repo_root, args.ledger_path)
    runtime_latest_path = resolve_repo_path(repo_root, args.runtime_latest_path)

    batch_id = str(args.batch_id or "").strip()
    if not batch_id:
        return 2, {"schema": RESULT_SCHEMA, "action": "rollback", "ok": False, "error": "batch_id_required"}

    rows = parse_jsonl(ledger_path)
    apply_idx, apply_row = _latest_applied_entry_for_batch(rows, batch_id)
    if apply_row is None or apply_idx is None:
        return 2, {
            "schema": RESULT_SCHEMA,
            "action": "rollback",
            "ok": False,
            "error": "batch_not_found_or_not_applied",
            "batch_id": batch_id,
        }

    for row in rows[apply_idx + 1 :]:
        if str(row.get("batch_id") or "") == batch_id and str(row.get("event_type") or "") == "CONSOLIDATION_ROLLBACK_APPLIED":
            return 0, {
                "schema": RESULT_SCHEMA,
                "action": "rollback",
                "ok": True,
                "status": "already_rolled_back",
                "batch_id": batch_id,
                "rollback_event_id": row.get("event_id"),
            }

    inputs = apply_row.get("inputs") if isinstance(apply_row.get("inputs"), dict) else {}
    files = inputs.get("files") if isinstance(inputs.get("files"), list) else []
    outputs = apply_row.get("outputs") if isinstance(apply_row.get("outputs"), dict) else {}

    artifact_rel = str(outputs.get("artifact_path") or "").strip()
    expected_artifact_sha = str(outputs.get("artifact_sha256") or "").strip()
    artifact_abs: Optional[Path] = resolve_repo_path(repo_root, artifact_rel) if artifact_rel else None

    restored: List[Dict[str, Any]] = []
    moved: List[Tuple[Path, Path]] = []

    try:
        if artifact_abs is not None:
            if not artifact_abs.exists() or not artifact_abs.is_file():
                raise RuntimeError(f"artifact_missing_before_rollback:{artifact_rel}")
            if expected_artifact_sha:
                artifact_sha = f"sha256:{file_sha256(artifact_abs)}"
                if artifact_sha != expected_artifact_sha:
                    raise RuntimeError(f"artifact_hash_mismatch:{artifact_rel}")

        for row in files:
            if not isinstance(row, dict):
                continue
            src_rel = str(row.get("source_path") or "").strip()
            archived_rel = str(row.get("archived_path") or "").strip()
            expected_sha = str(row.get("source_sha256") or "").strip()
            if not src_rel or not archived_rel:
                raise RuntimeError("ledger_file_row_missing_path")
            if not expected_sha:
                raise RuntimeError(f"ledger_file_row_missing_source_sha:{src_rel or archived_rel}")

            src_abs = resolve_repo_path(repo_root, src_rel)
            archived_abs = resolve_repo_path(repo_root, archived_rel)
            if src_abs.exists():
                raise RuntimeError(f"restore_target_exists:{src_rel}")
            if not archived_abs.exists():
                raise RuntimeError(f"archived_source_missing:{archived_rel}")
            archived_sha = f"sha256:{file_sha256(archived_abs)}"
            if archived_sha != expected_sha:
                raise RuntimeError(f"archived_hash_mismatch:{src_rel}")

            src_abs.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(archived_abs), str(src_abs))
            moved.append((src_abs, archived_abs))
            restored.append({"source_path": src_rel, "restored_from": archived_rel})
    except Exception as exc:
        # Best-effort forward restore to previous state.
        for src_abs, archived_abs in reversed(moved):
            if src_abs.exists() and not archived_abs.exists():
                archived_abs.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_abs), str(archived_abs))

        failure_detail = str(exc)
        rollback_failure_event: Dict[str, Any] = {
            "schema": LEDGER_SCHEMA,
            "event_type": "CONSOLIDATION_ROLLBACK_FAILED",
            "recorded_at": now_iso(),
            "batch_id": batch_id,
            "script_version": SCRIPT_VERSION,
            "status": "rollback_failed",
            "failure_code": FAILURE_CODE_ROLLBACK_VALIDATION,
            "reason": "rollback_restore_failed",
            "rollback_of_event_id": apply_row.get("event_id"),
            "last_validation_gate_failure": {
                "gate": GATE_ROLLBACK_VALIDATION,
                "error": "rollback_restore_failed",
                "failure_code": FAILURE_CODE_ROLLBACK_VALIDATION,
                "detail": failure_detail,
            },
            "inputs": {
                "files": files,
                "file_count": len(files),
            },
            "outputs": {
                "restored_count_before_failure": len(restored),
            },
        }
        rollback_failure_event["event_id"] = _event_id(rollback_failure_event)
        append_jsonl(ledger_path, rollback_failure_event)

        atomic_write_json(
            runtime_latest_path,
            {
                "schema": RUNTIME_SCHEMA,
                "updated_at": now_iso(),
                "status": "rollback_failed",
                "batch_id": batch_id,
                "reason": "rollback_restore_failed",
                "failure_code": FAILURE_CODE_ROLLBACK_VALIDATION,
                "last_validation_gate_failure": {
                    "gate": GATE_ROLLBACK_VALIDATION,
                    "error": "rollback_restore_failed",
                    "failure_code": FAILURE_CODE_ROLLBACK_VALIDATION,
                    "detail": failure_detail,
                },
                "ledger_path": safe_rel(repo_root, ledger_path),
            },
        )

        return 2, {
            "schema": RESULT_SCHEMA,
            "action": "rollback",
            "ok": False,
            "error": "rollback_restore_failed",
            "failure_code": FAILURE_CODE_ROLLBACK_VALIDATION,
            "detail": failure_detail,
            "batch_id": batch_id,
            "rollback_failure_event_id": rollback_failure_event["event_id"],
        }

    if artifact_abs is not None and artifact_abs.exists() and artifact_abs.is_file():
        artifact_abs.unlink()

    rollback_event: Dict[str, Any] = {
        "schema": LEDGER_SCHEMA,
        "event_type": "CONSOLIDATION_ROLLBACK_APPLIED",
        "recorded_at": now_iso(),
        "batch_id": batch_id,
        "script_version": SCRIPT_VERSION,
        "status": "rolled_back",
        "rollback_of_event_id": apply_row.get("event_id"),
        "restored_files": restored,
        "artifact_removed": bool(artifact_abs and not artifact_abs.exists()),
    }
    rollback_event["event_id"] = _event_id(rollback_event)
    append_jsonl(ledger_path, rollback_event)

    atomic_write_json(
        runtime_latest_path,
        {
            "schema": RUNTIME_SCHEMA,
            "updated_at": now_iso(),
            "status": "rolled_back",
            "batch_id": batch_id,
            "rollback_event_id": rollback_event["event_id"],
            "restored_count": len(restored),
        },
    )

    return 0, {
        "schema": RESULT_SCHEMA,
        "action": "rollback",
        "ok": True,
        "status": "rolled_back",
        "batch_id": batch_id,
        "restored_count": len(restored),
        "rollback_event_id": rollback_event["event_id"],
    }


def cmd_status(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    ok, _, reason = safe_repo_path(repo_root, args.ledger_path)
    if not ok:
        return 2, {"schema": RESULT_SCHEMA, "action": "status", "ok": False, "error": reason}

    ledger_path = resolve_repo_path(repo_root, args.ledger_path)
    rows = parse_jsonl(ledger_path)

    applied = [r for r in rows if str(r.get("event_type") or "") == "CONSOLIDATION_APPLIED"]
    rolled_back = [r for r in rows if str(r.get("event_type") or "") == "CONSOLIDATION_ROLLBACK_APPLIED"]
    failed_rb = [r for r in rows if str(r.get("event_type") or "") == "CONSOLIDATION_FAILED_ROLLED_BACK"]

    latest = rows[-1] if rows else None
    return 0, {
        "schema": RESULT_SCHEMA,
        "action": "status",
        "ok": True,
        "ledger_path": str(ledger_path),
        "ledger_row_count": len(rows),
        "applied_count": len(applied),
        "rolled_back_count": len(rolled_back),
        "failed_rolled_back_count": len(failed_rb),
        "latest_event": latest,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Governed memory consolidation runtime (MEM-02 bounded slice)")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--memory-root", default=str(DEFAULT_MEMORY_ROOT), help="Memory root directory")
    ap.add_argument("--ledger-path", default=str(DEFAULT_LEDGER_PATH), help="Consolidation ledger (jsonl)")
    ap.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT), help="Archive root for source files")
    ap.add_argument("--consolidated-root", default=str(DEFAULT_CONSOLIDATED_ROOT), help="Consolidated artifact root")
    ap.add_argument(
        "--runtime-latest-path",
        default=str(DEFAULT_RUNTIME_LATEST),
        help="Runtime latest status artifact path",
    )
    ap.add_argument(
        "--continuity-current-path",
        default=str(DEFAULT_CONTINUITY_CURRENT),
        help="Continuity current path used for governance gate",
    )
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON")

    sub = ap.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run one bounded consolidation batch")
    p_run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_run.add_argument("--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES)
    p_run.add_argument("--older-than-days", type=int, default=DEFAULT_OLDER_THAN_DAYS)
    p_run.add_argument("--dry-run", action="store_true", help="Evaluate one batch without mutating files")
    p_run.add_argument("--fault-inject-invalid-artifact", action="store_true", help=argparse.SUPPRESS)
    p_run.add_argument("--fault-inject-source-hash-mismatch", action="store_true", help=argparse.SUPPRESS)
    p_run.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_rollback = sub.add_parser("rollback", help="Rollback a previously applied consolidation batch")
    p_rollback.add_argument("--batch-id", required=True)
    p_rollback.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_status = sub.add_parser("status", help="Inspect consolidation ledger status")
    p_status.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.command == "run":
        rc, payload = cmd_run(args)
    elif args.command == "rollback":
        rc, payload = cmd_rollback(args)
    elif args.command == "status":
        rc, payload = cmd_status(args)
    else:
        rc, payload = 2, {"schema": RESULT_SCHEMA, "ok": False, "error": f"unknown_command:{args.command}"}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(stable_json(payload if isinstance(payload, dict) else {"payload": payload}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
