"""Best-effort, profile-local provenance records for one-shot CLI runs."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from hermes_constants import get_hermes_home

_AUDIT_FILENAME = "oneshot-audit.jsonl"
_INPUT_MODES = frozenset({"prompt", "query", "query-file", "programmatic"})


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _executable_basename(path: object) -> str | None:
    value = str(path or "").strip()
    return Path(value).name or None


def _parent_executable() -> str | None:
    """Return only the parent executable's basename; command arguments are private."""
    try:
        import psutil

        parent = psutil.Process(os.getppid())
        return _executable_basename(parent.exe() or parent.name())
    except Exception:
        return None


def _safe_process_value(fn) -> object | None:
    try:
        return fn()
    except Exception:
        return None


def _tty_name() -> str | None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            if stream is not None and stream.isatty():
                return os.ttyname(stream.fileno())
        except Exception:
            continue
    return None


def _append_record(path: Path | None, record: dict[str, object]) -> None:
    """Append one complete JSONL record with one write, without affecting the CLI run."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except Exception:
        pass


class OneshotAudit:
    """Correlates the started and finished records for one one-shot invocation."""

    __slots__ = ("_common", "_finished", "_lock", "_path")

    def __init__(self, path: Path | None, common: dict[str, object]) -> None:
        self._path = path
        self._common = common
        self._finished = False
        self._lock = threading.Lock()

    def finish(self, outcome: str, exit_code: int, session_id: object = None) -> None:
        """Write the terminal outcome at most once; audit failures never escape."""
        try:
            with self._lock:
                if self._finished:
                    return
                self._finished = True
            record = {
                **self._common,
                "event": "finished",
                "timestamp": _utc_timestamp(),
                "outcome": outcome,
                "exit_code": int(exit_code),
                "session_id": str(session_id) if session_id is not None else None,
            }
            _append_record(self._path, record)
        except Exception:
            pass


class _NoopAudit(OneshotAudit):
    def finish(self, outcome: str, exit_code: int, session_id: object = None) -> None:
        return None


def start_oneshot_audit(prompt: str, input_mode: str = "programmatic") -> OneshotAudit:
    """Start a one-shot audit pair without retaining prompt text or process arguments."""
    try:
        prompt_bytes = (prompt or "").encode("utf-8", errors="surrogatepass")
        normalized_mode = input_mode if input_mode in _INPUT_MODES else "programmatic"
        common: dict[str, object] = {
            "audit_id": uuid.uuid4().hex,
            "input_mode": normalized_mode,
            "prompt_chars": len(prompt or ""),
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "uid": _safe_process_value(os.getuid) if hasattr(os, "getuid") else None,
            "cwd": _safe_process_value(os.getcwd),
            "tty": _tty_name(),
            "executable": _executable_basename(sys.executable),
            "parent_executable": _parent_executable(),
        }
        path = get_hermes_home() / "logs" / _AUDIT_FILENAME
    except Exception:
        return _NoopAudit()

    try:
        audit = OneshotAudit(path, common)
        _append_record(path, {**common, "event": "started", "timestamp": _utc_timestamp()})
        return audit
    except Exception:
        return _NoopAudit()
