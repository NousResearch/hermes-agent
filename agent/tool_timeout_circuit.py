"""Cross-restart circuit breaker for exact terminal calls that timed out.

Only HMAC fingerprints and expiry times are persisted; tool arguments and
session identifiers never touch disk in plaintext. The profile-local key,
lock, and ledger are owner-only. Storage remains best-effort so an unavailable
cache does not make every tool call fail.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_TIMEOUT_RETRY_TTL_SECONDS = 15 * 60
_MAX_ENTRIES = 256
_lock = threading.Lock()
_ephemeral_key = secrets.token_bytes(32)


def _ledger_path() -> Path:
    return get_hermes_home() / "cache" / "tool-timeout-circuit.json"


def _key_path() -> Path:
    return _ledger_path().with_name("tool-timeout-circuit.key")


def _lock_path() -> Path:
    return _ledger_path().with_name("tool-timeout-circuit.lock")


def _secure_cache_dir() -> Path:
    directory = _ledger_path().parent
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass
    return directory


def _fingerprint_key() -> bytes:
    """Load or atomically create the profile-local HMAC key.

    If storage is unavailable, use a process-local random key. That degrades
    restart persistence but never degrades the privacy property to an unkeyed
    digest or plaintext.
    """
    path = _key_path()
    try:
        _secure_cache_dir()
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            key = path.read_bytes()
        else:
            try:
                key = secrets.token_bytes(32)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(key)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
        os.chmod(path, 0o600)
        if len(key) != 32:
            raise ValueError("invalid timeout circuit key length")
        return key
    except (OSError, ValueError):
        return _ephemeral_key


def tool_call_fingerprint(tool_name: str, args: dict[str, Any], session_id: str) -> str:
    fingerprint_args = dict(args)
    if tool_name == "terminal":
        # Timeout length and the internal approval replay bit do not change the
        # shell operation or its possible side effects. Changing only either
        # must not evade the retry circuit.
        fingerprint_args.pop("timeout", None)
        fingerprint_args.pop("force", None)
    canonical = json.dumps(
        {
            "session_id": session_id or "",
            "tool_name": tool_name,
            "args": fingerprint_args,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=repr,
    ).encode("utf-8")
    return hmac.new(_fingerprint_key(), canonical, hashlib.sha256).hexdigest()


@contextmanager
def _process_ledger_lock() -> Iterator[None]:
    """Serialize ledger read-modify-write across gateway/CLI processes."""
    fd: int | None = None
    locked = False
    try:
        _secure_cache_dir()
        fd = os.open(_lock_path(), os.O_RDWR | os.O_CREAT, 0o600)
        os.chmod(_lock_path(), 0o600)
        if os.name == "nt":  # pragma: no cover - Windows CI only
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
                os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
    except OSError as exc:
        logger.debug("could not lock tool timeout circuit: %s", exc)

    try:
        # If cross-process locking was unavailable, the caller still has the
        # process-local lock and atomic replacement prevents partial JSON.
        yield
    finally:
        if fd is not None:
            if locked:
                try:
                    if os.name == "nt":  # pragma: no cover - Windows CI only
                        import msvcrt

                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            try:
                os.close(fd)
            except OSError:
                pass


def _load_live_entries(now: float) -> dict[str, float]:
    try:
        path = _ledger_path()
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return {}
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return {}
        live: dict[str, float] = {}
        for key, expiry in entries.items():
            if (
                isinstance(key, str)
                and len(key) == 64
                and isinstance(expiry, (int, float))
                and not isinstance(expiry, bool)
                and float(expiry) > now
            ):
                live[key] = float(expiry)
        return live
    except (OSError, ValueError, TypeError):
        return {}


def _save_entries(entries: dict[str, float]) -> None:
    path = _ledger_path()
    tmp: Path | None = None
    try:
        _secure_cache_dir()
        # Keep latest-expiring entries only when malformed clocks or many
        # sessions create pressure; every retained entry is still bounded.
        bounded = dict(sorted(entries.items(), key=lambda item: item[1])[-_MAX_ENTRIES:])
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "entries": bounded}, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    except OSError as exc:
        logger.debug("could not persist tool timeout circuit: %s", exc)
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def is_tool_timeout_blocked(
    tool_name: str, args: dict[str, Any], session_id: str
) -> bool:
    now = time.time()
    with _lock:
        with _process_ledger_lock():
            fingerprint = tool_call_fingerprint(tool_name, args, session_id)
            return fingerprint in _load_live_entries(now)


def record_tool_timeout(
    tool_name: str,
    args: dict[str, Any],
    session_id: str,
) -> None:
    now = time.time()
    with _lock:
        with _process_ledger_lock():
            fingerprint = tool_call_fingerprint(tool_name, args, session_id)
            entries = _load_live_entries(now)
            entries[fingerprint] = now + _TIMEOUT_RETRY_TTL_SECONDS
            _save_entries(entries)
