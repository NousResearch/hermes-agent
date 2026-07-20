#!/usr/bin/env python3
"""Private, atomic state transitions for model-fusion; no child execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

ARTIFACTS = frozenset({
    "run.json", "sources.json", "judge.json", "judgment.json", "summary.json", "diagnostics.json"
})
SOURCE_ROLES = ("architect", "builder")
SOURCE_FIELDS = frozenset({
    "role", "status", "lane", "requested_provider", "requested_model",
    "actual_provider", "actual_model", "fallback_used",
})
MAX_SOURCE_CHARS = 8_000
MAX_DIAGNOSTIC_CHARS = 512
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("artifact must contain an object")
    return value


def _bounded(value: Any) -> Any:
    if isinstance(value, str):
        return value[:MAX_DIAGNOSTIC_CHARS]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return [_bounded(item) for item in value[:16]]
    if isinstance(value, dict):
        return {str(key)[:64]: _bounded(item) for key, item in list(value.items())[:16]}
    return type(value).__name__


def write_artifact(run_dir: Path | str, name: str, value: dict[str, Any]) -> Path:
    """Atomically replace one allow-listed JSON artifact with mode 0600."""
    run = Path(run_dir)
    if name not in ARTIFACTS:
        raise ValueError("artifact name is not allowed")
    if not isinstance(value, dict):
        raise ValueError("artifact value must be an object")
    if name == "diagnostics.json":
        value = _bounded(value)
    data = (json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=run)
    temporary_path = Path(temporary)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, run / name)
        os.chmod(run / name, 0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return run / name


def _guard_path(root: Path, session_id: str) -> Path:
    token = hashlib.sha256(session_id.encode()).hexdigest()
    return root / ".guards" / token


def create_run(root: Path | str, run_id: str, session_id: str, request: str) -> Path:
    """Create a private run and acquire the originating-session guard."""
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError("run_id is invalid")
    if not isinstance(session_id, str) or not session_id or not isinstance(request, str) or not request:
        raise ValueError("session_id and request are required")
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(base, 0o700)
    guards = base / ".guards"
    guards.mkdir(mode=0o700, exist_ok=True)
    os.chmod(guards, 0o700)
    guard = _guard_path(base, session_id)
    try:
        fd = os.open(guard, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("an active model-fusion run already exists for this session") from exc
    run = base / run_id
    try:
        run.mkdir(mode=0o700)
        os.chmod(run, 0o700)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(run_id)
        write_artifact(run, "run.json", {
            "run_id": run_id, "session_id": session_id, "request": request,
            "phase": "sources_pending", "judge_staged": False, "guard": guard.name,
        })
        return run
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        guard.unlink(missing_ok=True)
        raise


def _normalize_source(expected_role: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"role": expected_role, "status": "unknown"}
    role = raw.get("role", expected_role)
    status = raw.get("status", "unknown")
    if role != expected_role or status not in {"completed", "failed", "cancelled", "unknown"}:
        status = "unknown"
    if status == "completed":
        provenance_valid = (
            raw.get("lane") == expected_role
            and raw.get("fallback_used") is False
            and isinstance(raw.get("requested_provider"), str)
            and raw.get("requested_provider") == raw.get("actual_provider")
            and isinstance(raw.get("requested_model"), str)
            and raw.get("requested_model") == raw.get("actual_model")
        )
        if not provenance_valid:
            status = "unknown"
    result = {key: raw[key] for key in SOURCE_FIELDS if key in raw}
    result["role"] = expected_role
    result["status"] = status
    summary = raw.get("summary")
    if status == "completed":
        if not isinstance(summary, str):
            result["status"] = "unknown"
        else:
            result["content"] = summary[:MAX_SOURCE_CHARS]
    return result


def apply_source_batch(run_dir: Path | str, completion: dict[str, Any]) -> dict[str, Any]:
    """Consume the single ordered consolidated completion idempotently."""
    run = Path(run_dir)
    existing = run / "sources.json"
    if existing.exists():
        return _read(existing)
    results = completion.get("results") if isinstance(completion, dict) else None
    if not isinstance(results, list) or len(results) != len(SOURCE_ROLES):
        results = [None, None]
    batch_status = completion.get("status", "unknown") if isinstance(completion, dict) else "unknown"
    if batch_status != "completed":
        results = [None, None]
    sources = [_normalize_source(role, results[index]) for index, role in enumerate(SOURCE_ROLES)]
    success_count = sum(item["status"] == "completed" for item in sources)
    payload = {
        "batch_status": batch_status,
        "source_success_count": success_count,
        "sources": sources,
        "phase": "sources_complete" if success_count else "failed",
    }
    write_artifact(run, "sources.json", payload)
    state = _read(run / "run.json")
    state["phase"] = payload["phase"]
    write_artifact(run, "run.json", state)
    if not success_count:
        release_guard(run)
    return payload


def stage_judge(run_dir: Path | str) -> bool:
    """Atomically record judge dispatch eligibility exactly once."""
    run = Path(run_dir)
    sources = _read(run / "sources.json")
    if sources.get("source_success_count", 0) < 1:
        return False
    judge = run / "judge.json"
    try:
        fd = os.open(judge, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    payload = b'{"status":"staged"}\n'
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    state = _read(run / "run.json")
    state.update({"phase": "judge_pending", "judge_staged": True})
    write_artifact(run, "run.json", state)
    return True


def finalize_judgment(
    run_dir: Path | str, judgment: dict[str, Any], *, summary: dict[str, Any]
) -> dict[str, Any]:
    """Persist a validated judgment and compact summary, then release the guard."""
    run = Path(run_dir)
    judge = _read(run / "judge.json")
    if judge.get("status") != "staged":
        raise RuntimeError("judge was not staged")
    write_artifact(run, "judgment.json", judgment)
    write_artifact(run, "summary.json", summary)
    write_artifact(run, "judge.json", {"status": "completed"})
    state = _read(run / "run.json")
    state["phase"] = "completed"
    write_artifact(run, "run.json", state)
    release_guard(run)
    return state


def finalize_invalid_judgment(run_dir: Path | str, code: str, detail: str) -> dict[str, Any]:
    """Record bounded diagnostics for an invalid judge result and finish partial."""
    run = Path(run_dir)
    judge = _read(run / "judge.json")
    if judge.get("status") != "staged":
        raise RuntimeError("judge was not staged")
    write_artifact(run, "diagnostics.json", {"code": code, "detail": detail})
    write_artifact(run, "judge.json", {"status": "invalid"})
    state = _read(run / "run.json")
    state["phase"] = "partial"
    write_artifact(run, "run.json", state)
    release_guard(run)
    return state


def release_guard(run_dir: Path | str) -> bool:
    run = Path(run_dir)
    state = _read(run / "run.json")
    guard_name = state.get("guard")
    if not isinstance(guard_name, str):
        return False
    guard = run.parent / ".guards" / guard_name
    if not guard.exists():
        return False
    if guard.read_text(encoding="utf-8") != state.get("run_id"):
        return False
    guard.unlink()
    return True
