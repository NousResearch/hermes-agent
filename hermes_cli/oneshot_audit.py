"""Profile-local provenance ledger for CLI one-shot invocations."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from hermes_constants import get_hermes_home


_APPEND_LOCK = threading.Lock()
_AUDIT_FILENAME = "oneshot-audit.jsonl"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _uid() -> int | None:
    with contextlib.suppress(AttributeError, OSError):
        return os.getuid()
    return None


def _tty() -> str | None:
    try:
        if not sys.stdin.isatty():
            return None
        with contextlib.suppress(AttributeError, OSError, ValueError):
            return os.ttyname(sys.stdin.fileno())
        name = getattr(sys.stdin, "name", None)
        return str(name) if name and not str(name).startswith("<") else None
    except Exception:
        return None


def _parent_executable() -> str | None:
    try:
        import psutil

        return psutil.Process(os.getppid()).name() or None
    except Exception:
        return None


def _lock_file(handle) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append_record(record: dict[str, Any]) -> None:
    """Append one complete JSONL record under a cross-process lock."""
    logs_dir = get_hermes_home() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    lock_path = logs_dir / f".{_AUDIT_FILENAME}.lock"
    with _APPEND_LOCK, lock_path.open("a+b") as lock_handle:
        _lock_file(lock_handle)
        try:
            fd = os.open(logs_dir / _AUDIT_FILENAME, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(fd, remaining)
                    if written <= 0:
                        raise OSError("short write to one-shot audit ledger")
                    remaining = remaining[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            _unlock_file(lock_handle)


def _normalized_exit_code(code: object) -> int:
    return code if isinstance(code, int) else (0 if code is None else 1)


class OneShotAudit:
    """Idempotent start/finish writer for one CLI invocation."""

    def __init__(self, prompt: str | None, input_mode: str):
        prompt_text = prompt if isinstance(prompt, str) else None
        self.audit_id = uuid.uuid4().hex
        self._session_id: str | None = None
        self._outcome_hint: str | None = None
        self._finished = False
        self._base = {
            "audit_id": self.audit_id,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "uid": _uid(),
            "cwd": os.getcwd(),
            "tty": _tty(),
            "parent_executable": _parent_executable(),
            "input_mode": input_mode,
            "prompt_sha256": (
                hashlib.sha256(prompt_text.encode("utf-8", "surrogatepass")).hexdigest()
                if prompt_text is not None
                else None
            ),
            "prompt_chars": len(prompt_text) if prompt_text is not None else None,
        }
        self._write("started", outcome=None, exit_code=None)

    @classmethod
    def start(cls, prompt: str | None, input_mode: str) -> OneShotAudit | None:
        """Start an audit without making ledger I/O fatal to the invocation."""
        try:
            return cls(prompt, input_mode)
        except Exception:
            return None

    def bind_session(self, session_id: object) -> None:
        if session_id:
            self._session_id = str(session_id)

    def set_outcome_hint(self, outcome: str) -> None:
        self._outcome_hint = outcome

    def finish(self, exit_code: object = 0, outcome: str | None = None) -> None:
        if self._finished:
            return
        code = _normalized_exit_code(exit_code)
        resolved_outcome = outcome or self._outcome_hint
        if resolved_outcome is None:
            if code == 0:
                resolved_outcome = "normal"
            elif code == 130:
                resolved_outcome = "interrupted"
            elif code == 2:
                resolved_outcome = "validation_error"
            else:
                resolved_outcome = "agent_error"
        try:
            self._write("finished", outcome=resolved_outcome, exit_code=code)
        except Exception:
            return
        self._finished = True

    def _write(self, event: str, *, outcome: str | None, exit_code: int | None) -> None:
        _append_record(
            {
                **self._base,
                "event": event,
                "timestamp": _timestamp(),
                "outcome": outcome,
                "exit_code": exit_code,
                "session_id": self._session_id,
            }
        )


def finish_from_exception(audit: OneShotAudit | None, exc: BaseException) -> None:
    if audit is None:
        return
    if isinstance(exc, KeyboardInterrupt):
        audit.finish(130, "interrupted")
    elif isinstance(exc, SystemExit):
        audit.finish(exc.code)
    else:
        audit.finish(1, "agent_error")