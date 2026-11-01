"""run-receipts core: append-only, hash-chained per-run evidence records.

A "run" is one agent turn: it opens on the first hook event for a session
(tool call, API call, or session start) and is finalized by ``on_session_end``
which carries the outcome (completed / failed / interrupted). The finalized
receipt is appended as one JSON line to ``<HERMES_HOME>/receipts/runs.ndjson``.

Two integrity properties:

- Every record carries ``sha256`` over its canonical JSON (all fields except
  ``sha256`` itself) — tampering with any field breaks the record hash.
- Every record carries ``prev_sha256`` = the ``sha256`` of the previous line —
  deleting or reordering lines breaks the chain.

Privacy contract: args/results/error messages are NEVER stored raw — only
their sha256 and sizes. Model, provider, platform, timings, counts and
outcomes are the only plaintext fields.

Concurrency: gateway multiplex, cron children and CLI runs are separate
processes appending to the same per-profile file. A small lockfile
(POSIX ``fcntl.flock``, Windows ``msvcrt.locking``, best-effort fallback)
serializes the read-tail + append so chains do not fork.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
RECEIPTS_DIRNAME = "receipts"
RECEIPTS_FILENAME = "runs.ndjson"
_LOCK_FILENAME = "runs.ndjson.lock"
_TAIL_BYTES = 64 * 1024
_MAX_PENDING = 256

GENESIS_PREV = "0" * 64


def _canonical_sha256(record: Dict[str, Any]) -> str:
    """sha256 over the record's canonical JSON, excluding the sha256 field."""
    payload = {k: v for k, v in record.items() if k != "sha256"}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def content_sha256(value: Any) -> str:
    """sha256 of a value's canonical form — used for args/results/messages, never the content."""
    try:
        blob = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
        )
    except Exception:
        blob = repr(value)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def receipts_path(hermes_home: Path) -> Path:
    return Path(hermes_home) / RECEIPTS_DIRNAME / RECEIPTS_FILENAME


def _read_last_sha256(path: Path) -> str:
    """sha256 field of the last well-formed line, or GENESIS_PREV for a new/corrupt tail."""
    try:
        size = path.stat().st_size
        if size == 0:
            return GENESIS_PREV
        with open(path, "rb") as fh:
            fh.seek(max(0, size - _TAIL_BYTES))
            tail = fh.read()
        lines = [ln for ln in tail.split(b"\n") if ln.strip()]
        if not lines:
            return GENESIS_PREV
        last = json.loads(lines[-1].decode("utf-8"))
        sha = last.get("sha256")
        return sha if isinstance(sha, str) and len(sha) == 64 else GENESIS_PREV
    except Exception:
        return GENESIS_PREV


def write_receipt(hermes_home: Path, record: Dict[str, Any]) -> Path:
    """Chain-link and append one receipt; returns the file written.

    The tail read and the append must happen under the same lock — two
    processes that both read the same tail would produce a forked chain.
    """
    path = receipts_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / _LOCK_FILENAME
    try:
        with open(lock_path, "a+b") as lock_fh:
            if sys.platform == "win32":
                import msvcrt

                lock_fh.seek(0)
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                record["prev_sha256"] = _read_last_sha256(path)
                record["sha256"] = _canonical_sha256(record)
                with open(path, "a", encoding="utf-8") as out:
                    out.write(
                        json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n"
                    )
                    out.flush()
                    os.fsync(out.fileno())
            finally:
                if sys.platform == "win32":
                    import msvcrt

                    lock_fh.seek(0)
                    msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        # Lockfile unusable (read-only fs, exotic platform) — append unlocked rather than lose the run.
        record.setdefault("prev_sha256", _read_last_sha256(path))
        record.setdefault("sha256", _canonical_sha256(record))
        with open(path, "a", encoding="utf-8") as out:
            out.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
    return path


def iter_receipts(path: Path):
    """Yield (line_no, record_or_None) for each non-empty line."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield line_no, json.loads(raw)
                except json.JSONDecodeError:
                    yield line_no, None
    except FileNotFoundError:
        return


def verify_chain(path: Path) -> Dict[str, Any]:
    """Walk the file; return a report. ``ok`` is True only if every line is well-formed,
    self-hashes correctly, and links to its predecessor."""
    report: Dict[str, Any] = {
        "ok": True,
        "path": str(path),
        "records": 0,
        "errors": [],
        "first_error_line": None,
    }
    prev_sha = GENESIS_PREV
    for line_no, rec in iter_receipts(path):
        if rec is None:
            report["ok"] = False
            report["errors"].append(f"line {line_no}: not valid JSON")
            if report["first_error_line"] is None:
                report["first_error_line"] = line_no
            continue
        if not isinstance(rec, dict):
            report["ok"] = False
            report["errors"].append(f"line {line_no}: not an object")
            if report["first_error_line"] is None:
                report["first_error_line"] = line_no
            continue
        declared = rec.get("prev_sha256")
        if declared != prev_sha:
            report["ok"] = False
            report["errors"].append(
                f"line {line_no}: prev_sha256={declared!r} does not match previous sha256 {prev_sha!r}"
            )
            if report["first_error_line"] is None:
                report["first_error_line"] = line_no
        actual = _canonical_sha256(rec)
        if rec.get("sha256") != actual:
            report["ok"] = False
            report["errors"].append(
                f"line {line_no}: sha256 mismatch (record claims {rec.get('sha256')!r}, computed {actual!r})"
            )
            if report["first_error_line"] is None:
                report["first_error_line"] = line_no
        prev_sha = rec.get("sha256") or prev_sha
        report["records"] += 1
    if report["records"] == 0 and not path.exists():
        report["errors"].append("no receipts file found")
    return report


# --------------------------------------------------------------------------
# In-flight run accumulation (turn loop callbacks are synchronous — keep cheap)


class _PendingRun:
    __slots__ = (
        "session_id",
        "task_id",
        "turn_id",
        "model",
        "provider",
        "platform",
        "started_at",
        "tool_calls",
        "api_calls",
        "errors",
    )

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.task_id = ""
        self.turn_id = ""
        self.model = ""
        self.provider = ""
        self.platform = ""
        self.started_at = time.time()
        self.tool_calls: List[Dict[str, Any]] = []
        self.api_calls: List[Dict[str, Any]] = []
        self.errors = 0


_pending: Dict[str, _PendingRun] = {}
_pending_lock = threading.Lock()


def _pending_for(session_id: str) -> _PendingRun:
    run = _pending.get(session_id)
    if run is None:
        if len(_pending) >= _MAX_PENDING:
            oldest = min(_pending.values(), key=lambda r: r.started_at)
            _pending.pop(oldest.session_id, None)
        run = _pending[session_id] = _PendingRun(session_id)
    return run


def note_session_start(session_id: str, model: str = "", platform: str = "") -> None:
    if not session_id:
        return
    with _pending_lock:
        run = _pending_for(session_id)
        if model:
            run.model = model
        if platform:
            run.platform = platform


def note_tool_call(
    session_id: str,
    *,
    task_id: str = "",
    tool_call_id: str = "",
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    duration_ms: Any = None,
    turn_id: str = "",
) -> None:
    if not session_id:
        return
    entry: Dict[str, Any] = {
        "id": tool_call_id or "",
        "name": tool_name or "",
        "args_sha256": content_sha256(args),
    }
    if result is not None:
        entry["result_sha256"] = content_sha256(result)
        try:
            entry["result_chars"] = (
                len(result) if isinstance(result, str) else len(str(result))
            )
        except Exception:
            pass
    if isinstance(duration_ms, (int, float)):
        entry["duration_ms"] = duration_ms
    with _pending_lock:
        run = _pending_for(session_id)
        if task_id:
            run.task_id = task_id
        if turn_id:
            run.turn_id = turn_id
        run.tool_calls.append(entry)


def note_api_call(
    session_id: str,
    *,
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    model: str = "",
    provider: str = "",
    api_duration: Any = None,
    started_at: Any = None,
    ended_at: Any = None,
    api_call_count: Any = None,
) -> None:
    if not session_id:
        return
    entry: Dict[str, Any] = {
        "id": api_request_id or "",
        "ok": True,
        "model": model or "",
        "provider": provider or "",
    }
    if isinstance(api_duration, (int, float)):
        entry["duration_s"] = round(api_duration, 3)
    if isinstance(api_call_count, int):
        entry["n"] = api_call_count
    with _pending_lock:
        run = _pending_for(session_id)
        if task_id:
            run.task_id = task_id
        if turn_id:
            run.turn_id = turn_id
        if model:
            run.model = model
        if provider:
            run.provider = provider
        if isinstance(started_at, (int, float)) and started_at < run.started_at:
            run.started_at = started_at
        run.api_calls.append(entry)


def note_api_error(
    session_id: str,
    *,
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    model: str = "",
    provider: str = "",
    api_duration: Any = None,
    api_call_count: Any = None,
    status_code: Any = None,
    retryable: Any = None,
    reason: str = "",
    error: Any = None,
) -> None:
    if not session_id:
        return
    error = error if isinstance(error, dict) else {}
    entry: Dict[str, Any] = {
        "id": api_request_id or "",
        "ok": False,
        "model": model or "",
        "provider": provider or "",
        "error_type": error.get("type") or "",
        # error["message"] may carry unredacted request content — hash only.
        "error_sha256": content_sha256(error.get("message") or ""),
    }
    if isinstance(status_code, int):
        entry["status_code"] = status_code
    if isinstance(retryable, bool):
        entry["retryable"] = retryable
    if reason:
        entry["reason"] = reason
    if isinstance(api_duration, (int, float)):
        entry["duration_s"] = round(api_duration, 3)
    if isinstance(api_call_count, int):
        entry["n"] = api_call_count
    with _pending_lock:
        run = _pending_for(session_id)
        if task_id:
            run.task_id = task_id
        if turn_id:
            run.turn_id = turn_id
        if model:
            run.model = model
        if provider:
            run.provider = provider
        run.api_calls.append(entry)
        run.errors += 1


def finalize_run(
    session_id: str,
    *,
    task_id: str = "",
    turn_id: str = "",
    completed: bool = False,
    failed: bool = False,
    interrupted: bool = False,
    turn_exit_reason: str = "",
    model: str = "",
    platform: str = "",
) -> Optional[Dict[str, Any]]:
    """Seal the pending run into a receipt record. Returns the record (caller writes it)."""
    if not session_id:
        return None
    with _pending_lock:
        run = _pending.pop(session_id, None)
    if run is None:
        run = _PendingRun(session_id)
    ended_at = time.time()
    outcome = "interrupted" if interrupted else ("failed" if failed else "completed")
    record: Dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "session_id": session_id,
        "task_id": task_id or run.task_id,
        "turn_id": turn_id or run.turn_id,
        "platform": platform or run.platform,
        "model": model or run.model,
        "provider": run.provider,
        "started_at": round(run.started_at, 3),
        "ended_at": round(ended_at, 3),
        "duration_s": round(ended_at - run.started_at, 3),
        "outcome": outcome,
        "exit_reason": turn_exit_reason or "",
        "tool_calls": run.tool_calls,
        "api_calls": run.api_calls,
        "counts": {
            "tool_calls": len(run.tool_calls),
            "api_calls": len(run.api_calls),
            "errors": run.errors,
        },
    }
    return record


def drop_pending(session_id: str) -> None:
    """Session closed without a turn boundary — discard any residue."""
    with _pending_lock:
        _pending.pop(session_id, None)


# --------------------------------------------------------------------------
# CLI formatting helpers


def format_latest(path: Path) -> str:
    last: Optional[Dict[str, Any]] = None
    for _ln, rec in iter_receipts(path):
        if rec is not None:
            last = rec
    if last is None:
        return "No receipts recorded yet."
    return json.dumps(last, indent=2, sort_keys=True)


def format_stats(path: Path) -> str:
    total = 0
    outcomes: Dict[str, int] = {}
    tool_calls = 0
    api_calls = 0
    for _ln, rec in iter_receipts(path):
        if rec is None:
            continue
        total += 1
        outcomes[rec.get("outcome") or "unknown"] = (
            outcomes.get(rec.get("outcome") or "unknown", 0) + 1
        )
        counts = rec.get("counts") or {}
        tool_calls += int(counts.get("tool_calls") or 0)
        api_calls += int(counts.get("api_calls") or 0)
    if total == 0:
        return "No receipts recorded yet."
    lines = [
        f"Runs: {total}",
        f"Tool calls: {tool_calls}",
        f"API calls: {api_calls}",
        "Outcomes:",
    ]
    for outcome, n in sorted(outcomes.items()):
        lines.append(f"  {outcome}: {n}")
    return "\n".join(lines)


def format_verify(report: Dict[str, Any]) -> str:
    if report["ok"]:
        return f"OK — {report['records']} receipt(s), chain intact ({report['path']})"
    lines = [f"CHAIN BROKEN at line {report['first_error_line']} ({report['path']})"]
    lines.extend(f"  {e}" for e in report["errors"][:10])
    return "\n".join(lines)
