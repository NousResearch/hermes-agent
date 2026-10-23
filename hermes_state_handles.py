"""Session handles: a short typeable name per session (``tidy-amber-otter``) beside its machine id.

The handle is derived from the session id (seeded draw), so a retried create mints the same one, and a
UNIQUE collision moves to the next salt. Words come from coolname's curated adjective/animal lists.
"""

from __future__ import annotations

import copy
import hashlib
import random
import sqlite3
import threading
from typing import Optional

HANDLE_INDEX_SQL = "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_handle ON sessions(handle) WHERE handle IS NOT NULL"
HANDLE_MISSING_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_sessions_handle_missing ON sessions(id) WHERE handle IS NULL"
_MAX_WORD_CHARS = 7
_MAX_SALTS = 64
_BACKFILL_LIMIT = 20000
# Words that read badly as a label someone else sees in a screenshot or a shared command line.
_BLOCKED_WORDS = frozenset((
    "berserk", "cocky", "crazy", "demonic", "devious", "fat", "gay", "girlish", "greedy", "hairy", "heretic",
    "illegal", "kickass", "liberal", "loutish", "macho", "married", "naughty", "pompous", "sexy", "skinny",
    "uppish", "uptight", "vegan", "woodoo",
))

_lock = threading.Lock()
_generator = None
_rng = random.Random()


def _build_generator():
    from coolname import RandomGenerator
    from coolname.data import config

    config = copy.deepcopy(config)
    for key in ("2", "3", "4"):
        config.pop(key, None)
    config["all"] = {"type": "nested", "lists": ["aan"], "ensure_unique": True}
    for entry in config.values():
        if entry.get("type") == "words":
            kept = [w for w in entry["words"] if w not in _BLOCKED_WORDS and len(w) <= _MAX_WORD_CHARS]
            entry["words"] = kept or entry["words"]
    return RandomGenerator(config, rand=_rng)


def handle_for(session_id: str, salt: int = 0) -> str:
    """The handle *session_id* draws at *salt* (same inputs, same handle)."""
    global _generator
    seed = hashlib.blake2b(f"{session_id}\0{salt}".encode(), digest_size=16).digest()
    with _lock:
        if _generator is None:
            try:
                _generator = _build_generator()
            except ImportError as exc:
                _generator = exc
        if isinstance(_generator, ImportError):
            raise _generator
        _rng.seed(seed)
        return _generator.generate_slug()


def mint_handle(conn: sqlite3.Connection, session_id: str) -> Optional[str]:
    """Give *session_id*'s row a handle if it has none, inside the caller's write transaction."""
    row = conn.execute("SELECT handle FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None or row[0]:
        return row[0] if row else None
    for salt in range(_MAX_SALTS):
        try:
            candidate = handle_for(session_id, salt)
        except ImportError:  # env not yet synced after an update: never fail a create; the open backfill fills it
            return None
        if conn.execute("SELECT 1 FROM sessions WHERE handle = ?", (candidate,)).fetchone() is None:
            conn.execute("UPDATE sessions SET handle = ? WHERE id = ? AND handle IS NULL", (candidate, session_id))
            return candidate
    return None


def backfill_handles(cursor: sqlite3.Cursor, limit: int = _BACKFILL_LIMIT) -> int:
    """Mint handles for up to *limit* rows that have none (legacy rows, raw-INSERT paths) in one savepoint;
    returns rows filled."""
    ids = [r[0] for r in cursor.execute(
        "SELECT id FROM sessions INDEXED BY idx_sessions_handle_missing WHERE handle IS NULL LIMIT ?", (limit,))]
    if not ids:
        return 0
    taken = {r[0] for r in cursor.execute("SELECT handle FROM sessions WHERE handle IS NOT NULL")}
    updates = []
    try:
        for session_id in ids:
            handle = next((h for h in (handle_for(session_id, salt) for salt in range(_MAX_SALTS)) if h not in taken), None)
            if handle:
                taken.add(handle)
                updates.append((handle, session_id))
    except ImportError:
        pass
    cursor.execute("SAVEPOINT session_handles")
    try:
        # OR IGNORE: another process may claim a handle after `taken` was read; that row fills next open.
        cursor.executemany("UPDATE OR IGNORE sessions SET handle = ? WHERE id = ? AND handle IS NULL", updates)
    except BaseException:
        cursor.execute("ROLLBACK TO session_handles")
        raise
    finally:
        cursor.execute("RELEASE session_handles")
    return len(updates)
