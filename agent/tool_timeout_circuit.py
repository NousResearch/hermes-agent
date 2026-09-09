"""Privacy-preserving circuit for exact retries after terminal timeouts.

Every process keeps a bounded in-memory circuit. POSIX profiles additionally
persist HMAC fingerprints and expiry times behind a cross-process file lock;
tool arguments and session identifiers never touch disk in plaintext. When
secure persistence or locking is unavailable, the circuit degrades to the
process-local map instead of performing an unlocked shared read/modify/write.
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
from typing import Any, Callable, Iterator

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_TIMEOUT_RETRY_TTL_SECONDS = 15 * 60
_MAX_ENTRIES = 256
_lock = threading.Lock()
_ephemeral_key = secrets.token_bytes(32)
_memory_entries: dict[str, float] = {}


def _ledger_path() -> Path:
    return get_hermes_home() / "cache" / "tool-timeout-circuit.json"


def _key_path() -> Path:
    return _ledger_path().with_name("tool-timeout-circuit.key")


def _lock_path() -> Path:
    return _ledger_path().with_name("tool-timeout-circuit.lock")


def _persistent_storage_supported() -> bool:
    # chmod/open modes do not establish an owner-only DACL on Windows. Keep the
    # circuit process-local there rather than writing sensitive correlation data
    # with permissions we cannot guarantee.
    return os.name == "posix"


def _secure_cache_dir() -> Path:
    directory = _ledger_path().parent
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    return directory


def _fingerprint_payload(tool_name: str, args: dict[str, Any], session_id: str) -> bytes:
    fingerprint_args = dict(args)
    if tool_name == "terminal":
        # Timeout length and the internal approval replay bit do not change the
        # shell operation or its possible side effects.
        fingerprint_args.pop("timeout", None)
        fingerprint_args.pop("force", None)
    return json.dumps(
        {"session_id": session_id or "", "tool_name": tool_name, "args": fingerprint_args},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=repr,
    ).encode("utf-8")


def _fingerprint(payload: bytes, key: bytes) -> str:
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _persistent_key() -> bytes:
    """Load or atomically create the profile-local key while the ledger lock is held."""
    path = _key_path()
    _secure_cache_dir()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        key = path.read_bytes()
    else:
        key = secrets.token_bytes(32)
        with os.fdopen(fd, "wb") as handle:
            handle.write(key)
            handle.flush()
            os.fsync(handle.fileno())
    os.chmod(path, 0o600)
    if len(key) != 32:
        raise ValueError("invalid timeout circuit key length")
    return key


def tool_call_fingerprint(tool_name: str, args: dict[str, Any], session_id: str) -> str:
    """Return the stable process-local fingerprint used by the fallback circuit."""
    return _fingerprint(_fingerprint_payload(tool_name, args, session_id), _ephemeral_key)


@contextmanager
def _process_ledger_lock() -> Iterator[bool]:
    """Yield whether the POSIX cross-process lock was acquired."""
    if not _persistent_storage_supported():
        yield False
        return
    fd: int | None = None
    locked = False
    try:
        _secure_cache_dir()
        fd = os.open(_lock_path(), os.O_RDWR | os.O_CREAT, 0o600)
        os.chmod(_lock_path(), 0o600)
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
    except OSError as exc:
        logger.debug("could not lock tool timeout circuit: %s", exc)
    try:
        yield locked
    finally:
        if fd is not None:
            if locked:
                try:
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
        os.chmod(path, 0o600)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return {}
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return {}
        return {
            key: float(expiry)
            for key, expiry in entries.items()
            if isinstance(key, str)
            and len(key) == 64
            and isinstance(expiry, (int, float))
            and not isinstance(expiry, bool)
            and float(expiry) > now
        }
    except (OSError, ValueError, TypeError):
        return {}


def _save_entries(entries: dict[str, float]) -> None:
    path = _ledger_path()
    tmp: Path | None = None
    try:
        _secure_cache_dir()
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


def _prune_memory(now: float) -> None:
    live = {key: expiry for key, expiry in _memory_entries.items() if expiry > now}
    _memory_entries.clear()
    _memory_entries.update(dict(sorted(live.items(), key=lambda item: item[1])[-_MAX_ENTRIES:]))


def _persistent_fingerprint(payload: bytes) -> str | None:
    try:
        return _fingerprint(payload, _persistent_key())
    except (OSError, ValueError):
        return None


def _blocked_locked(payload: bytes, now: float, process_locked: bool) -> bool:
    process_fp = _fingerprint(payload, _ephemeral_key)
    _prune_memory(now)
    if process_fp in _memory_entries:
        return True
    if not process_locked:
        return False
    persistent_fp = _persistent_fingerprint(payload)
    return persistent_fp is not None and persistent_fp in _load_live_entries(now)


def is_tool_timeout_blocked(tool_name: str, args: dict[str, Any], session_id: str) -> bool:
    payload = _fingerprint_payload(tool_name, args, session_id)
    now = time.time()
    with _lock:
        with _process_ledger_lock() as process_locked:
            return _blocked_locked(payload, now, process_locked)


def try_admit_tool_call(
    tool_name: str,
    args: dict[str, Any],
    session_id: str,
    commit: Callable[[], bool],
) -> bool:
    """Atomically order circuit admission against timeout records, then commit dispatch.

    ``commit`` must be quick and side-effect free apart from marking the per-call
    lifecycle. It is invoked while the circuit locks are held, making either the
    timeout record or the dispatch commitment win deterministically.
    """
    payload = _fingerprint_payload(tool_name, args, session_id)
    now = time.time()
    with _lock:
        with _process_ledger_lock() as process_locked:
            if _blocked_locked(payload, now, process_locked):
                return False
            return bool(commit())


def record_tool_timeout(tool_name: str, args: dict[str, Any], session_id: str) -> None:
    payload = _fingerprint_payload(tool_name, args, session_id)
    now = time.time()
    expiry = now + _TIMEOUT_RETRY_TTL_SECONDS
    with _lock:
        process_fp = _fingerprint(payload, _ephemeral_key)
        _prune_memory(now)
        _memory_entries[process_fp] = expiry
        _prune_memory(now)
        with _process_ledger_lock() as process_locked:
            if not process_locked:
                return
            persistent_fp = _persistent_fingerprint(payload)
            if persistent_fp is None:
                return
            entries = _load_live_entries(now)
            entries[persistent_fp] = expiry
            _save_entries(entries)
