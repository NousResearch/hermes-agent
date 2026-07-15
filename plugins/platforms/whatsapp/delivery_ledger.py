"""Sanitized dead-letter ledger for WhatsApp bridge deliveries.

Disabled by default — upstream Hermes never writes this file unless a
profile opts in via ``HERMES_WA_DEAD_LETTER_LEDGER_ENABLED``. Each record
is a single JSONL line containing only sanitized, non-identifying fields:
timestamp, platform, route, a hash of the idempotency key, attempt count,
error category/status and resolution state. Message content, chat ids,
phone numbers and tokens are never accepted by this module at all, so a
caller cannot accidentally leak them into the ledger.
"""

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils import env_bool

_IS_WINDOWS = os.name == "nt"
if _IS_WINDOWS:
    import msvcrt
else:
    import fcntl

_LEDGER_ENABLED_ENV = "HERMES_WA_DEAD_LETTER_LEDGER_ENABLED"
_LEDGER_PATH_ENV = "HERMES_WA_DEAD_LETTER_LEDGER_PATH"

# Guards concurrent appends from threads within this process; the file lock
# below (flock/msvcrt) guards concurrent appends across processes.
_write_lock = threading.Lock()


def is_ledger_enabled() -> bool:
    """True only when a profile has explicitly opted into the ledger."""
    return env_bool(_LEDGER_ENABLED_ENV, False)


def default_ledger_path() -> Path:
    """Default ledger location under the Hermes state directory."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state" / "whatsapp_delivery_ledger.jsonl"


def ledger_path() -> Path:
    """Resolve the active ledger path, honoring the env override."""
    override = os.getenv(_LEDGER_PATH_ENV, "").strip()
    return Path(override) if override else default_ledger_path()


def _hash_idempotency_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _append_line(path: Path, line: str) -> None:
    """Append one line to *path*, holding a cross-process advisory lock.

    The lock is taken on a sibling ``.lock`` file rather than *path* itself —
    locking the data file directly would require a placeholder byte on
    Windows (``msvcrt.locking`` needs at least one byte to lock), which would
    permanently corrupt the first line of the ledger.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _write_lock:
        with open(lock_path, "a+", encoding="utf-8") as lock_handle:
            lock_handle.seek(0, os.SEEK_END)
            if lock_handle.tell() == 0:
                lock_handle.write("\0")
                lock_handle.flush()
            lock_handle.seek(0)
            try:
                if _IS_WINDOWS:
                    msvcrt.locking(lock_handle.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                try:
                    if _IS_WINDOWS:
                        lock_handle.seek(0)
                        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


def record_dead_letter(
    *,
    platform: str,
    route: str,
    idempotency_key: str,
    attempts: int,
    category: str,
    status: Optional[int] = None,
    resolution: str = "open",
) -> Optional[str]:
    """Append a sanitized dead-letter record; no-op unless the ledger is enabled.

    Returns a dead-letter reference string (usable as ``DeliveryOutcome.dead_letter_ref``)
    or ``None`` when the ledger is disabled. Only sanitized identifiers are
    accepted — there is no parameter for message content, chat id or token,
    so a caller cannot pass them even by mistake.
    """
    if not is_ledger_enabled():
        return None

    key_hash = _hash_idempotency_key(idempotency_key)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform,
        "route": route,
        "idempotency_key_hash": key_hash,
        "attempts": attempts,
        "category": category,
        "status": status,
        "resolution": resolution,
    }
    path = ledger_path()
    _append_line(path, json.dumps(entry, separators=(",", ":")))
    return f"{path}#{key_hash}"
