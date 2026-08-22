"""Local index of text we've sent via ``sendRichMessage`` (Bot API 10.1).

Telegram does NOT echo a rich message's content back in ``reply_to_message``
when a user replies to it (verified: ``.text``/``.caption`` empty,
``.api_kwargs`` None). So replies to the launchd briefings / any rich send
arrive with no quotable text and the agent is blind to what was referenced.

Fix: remember ``message_id -> text`` at send time, look it up by
``reply_to_id`` on inbound. This module is the single source of truth for that
index.

Best-effort and dependency-free: every operation swallows errors and degrades
to a no-op / ``None`` so it can never break a send or an inbound message.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Optional

_MAX_ENTRIES = 1000
_MAX_TEXT_CHARS = 2000
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _reset_thread_locks_after_fork() -> None:
    global _THREAD_LOCKS, _THREAD_LOCKS_GUARD

    _THREAD_LOCKS = {}
    _THREAD_LOCKS_GUARD = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_thread_locks_after_fork)


def _store_path() -> str:
    # Resolve via get_hermes_home() so the active profile override is honored.
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    return os.path.join(str(home), "state", "rich_sent_index.json")


def _key(chat_id, message_id) -> str:
    return f"{chat_id}:{message_id}"


def _retention_order(item) -> tuple[int, int | float]:
    entry = item[1]
    timestamp = entry.get("ts") if isinstance(entry, dict) else None
    if isinstance(timestamp, int) and not isinstance(timestamp, bool):
        return (1, timestamp)
    if isinstance(timestamp, float) and math.isfinite(timestamp):
        return (1, timestamp)
    return (0, 0)


def _thread_lock(path: str) -> threading.Lock:
    normalized = os.path.normcase(os.path.abspath(path))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(normalized, threading.Lock())


@contextmanager
def _record_lock(path: str):
    """Serialize one index's read-modify-write transaction."""
    with _thread_lock(path):
        # Keep a stable sidecar inode: deleting a lock file can split waiters
        # across old/new inodes. OS locks are released automatically on crash.
        with open(f"{path}.lock", "a+b") as lock_file:
            if os.name == "nt":
                import msvcrt

                # Windows byte-range locks may extend beyond EOF, so the lock
                # file can stay empty and needs no racy initialization write.
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def record(chat_id, message_id, text: Optional[str]) -> None:
    """Persist ``text`` for ``(chat_id, message_id)``. No-op on any failure."""
    if not text or message_id is None or chat_id is None:
        return
    try:
        path = _store_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _record_lock(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    data = {}
            except (FileNotFoundError, ValueError):
                data = {}
            data[_key(chat_id, message_id)] = {
                "t": text[:_MAX_TEXT_CHARS],
                "ts": int(time.time()),
            }
            # Trim oldest by timestamp when over cap.
            if len(data) > _MAX_ENTRIES:
                for k, _ in sorted(data.items(), key=_retention_order)[
                    : len(data) - _MAX_ENTRIES
                ]:
                    data.pop(k, None)

            tmp = None
            try:
                # Never reuse a temp name: a hard-crashed writer may leave its
                # inert file behind, while ordinary failures clean up below.
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=os.path.dirname(path),
                    prefix=f"{os.path.basename(path)}.tmp.",
                    delete=False,
                ) as fh:
                    tmp = fh.name
                    json.dump(data, fh, ensure_ascii=False)
                # The temp is closed first for Windows; same-directory replace
                # keeps publication atomic on every supported local filesystem.
                os.replace(tmp, path)
            finally:
                if tmp is not None:
                    try:
                        os.unlink(tmp)
                    except FileNotFoundError:
                        pass
    except Exception:
        return


def lookup(chat_id, message_id) -> Optional[str]:
    """Return stored text for ``(chat_id, message_id)`` or ``None``."""
    if message_id is None or chat_id is None:
        return None
    try:
        with open(_store_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        entry = data.get(_key(chat_id, message_id))
        if isinstance(entry, dict):
            return entry.get("t") or None
    except (FileNotFoundError, ValueError, AttributeError):
        return None
    return None
