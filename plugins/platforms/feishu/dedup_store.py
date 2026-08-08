"""Cross-profile inbound-message dedup for the Feishu adapter (issue #78514).

Feishu delivers events at least once: after a websocket blip or a server-side
retry the same ``message_id`` arrives again, sometimes hours later.  The
adapter already guards against that — but only within the instance that saw
the message first.

Under ``gateway.multiplex_profiles: true`` every profile runs its own Feishu
adapter against the SAME bot inside ONE gateway process (the app-id lock in
``gateway/status.py`` re-acquires when the holder is the current PID, so
nothing stops a sibling profile from connecting), and each instance keeps a
private cache persisted to its own PROFILE home —
``<root>/profiles/<name>/feishu_seen_message_ids.json``.  Feishu spreads a
bot's events across the open connections, so a redelivered event can land on a
sibling adapter, where it looks brand new: the agent re-answers a question
from hours ago and every message-triggered side effect (memory writes, file
writes, outbound calls) runs a second time.

This module keeps the record ONE level up — in the shared Hermes root, keyed
by ``platform + app_id`` — so every profile's adapter consults the same state.

SQLite rather than the previous JSON file because dedup is fundamentally a
test-and-set: two adapters racing the same redelivered id must not both read
"absent" and both process it.  ``INSERT OR IGNORE`` inside an IMMEDIATE
transaction is atomic across threads AND across processes; a read-modify-write
of a JSON file is neither, however atomically the file itself is replaced.

Connections are opened per operation rather than held for the adapter's
lifetime.  An open handle would pin the database file — which on Windows
blocks deleting the directory containing it — and would have to be threaded
through adapter disconnect/reconnect to be released.  The cost is nothing
next to what it replaces: the previous path rewrote an entire JSON file on
every inbound message.

Failures degrade rather than propagate.  A store that cannot be opened or
written leaves the adapter on its previous per-instance behaviour: callers see
"not a duplicate" and the message is delivered.  Dedup failing open risks a
repeated message; failing closed would silently drop real ones, which is the
worse trade on a message-delivery path.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_messages (
    namespace  TEXT NOT NULL,
    message_id TEXT NOT NULL,
    seen_at    REAL NOT NULL,
    PRIMARY KEY (namespace, message_id)
)
"""

# Prune scans the namespace; amortize it instead of paying on every message.
_PRUNE_EVERY = 200

# `seen()` runs synchronously on the adapter's asyncio loop, so a blocked
# write blocks inbound handling for the whole gateway. Transactions here are
# sub-millisecond and contention is two adapters landing the same redelivered
# burst, so a low ceiling is enough — and a busy-timeout expiry degrades to
# fail-open (deliver the message), the same trade the rest of this module
# makes. Keep this small enough that the worst case is a hiccup, not a stall.
_BUSY_TIMEOUT_SECONDS = 2.0


class SharedMessageDedupStore:
    """Atomic, cross-process seen-message store scoped to one ``namespace``.

    ``namespace`` isolates unrelated bots that share the file: two Feishu apps
    running under different profiles must not shadow each other's ids, and a
    future caller on another platform must not collide with Feishu at all.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        namespace: str,
        ttl_seconds: float,
        max_entries: int,
    ) -> None:
        self._db_path = Path(db_path)
        self._namespace = namespace
        self._ttl_seconds = float(ttl_seconds)
        self._max_entries = max(int(max_entries), 1)
        # Not required for correctness — BEGIN IMMEDIATE already serializes
        # writers across connections and processes — but it keeps this
        # process's own adapter threads off the busy-retry path.
        self._lock = threading.Lock()
        self._writes_since_prune = 0
        # Set after an unrecoverable sqlite error so a broken or read-only
        # store is not retried once per inbound message.
        self._disabled = False
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            # WAL is recorded in the file header, so setting it once here
            # applies to every later connection, including sibling processes.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_SCHEMA)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # isolation_level=None: transactions are issued explicitly below, so
        # BEGIN IMMEDIATE actually takes the write lock up front instead of
        # sqlite3's implicit deferred transaction, which can only discover a
        # competing writer at COMMIT (SQLITE_BUSY on upgrade).
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=_BUSY_TIMEOUT_SECONDS,
            isolation_level=None,
        )
        try:
            # Absorbs the brief write-lock overlap when two profiles receive
            # the same redelivered burst.
            conn.execute(f"PRAGMA busy_timeout={int(_BUSY_TIMEOUT_SECONDS * 1000)}")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self, conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def _fail(self, action: str) -> None:
        self._disabled = True
        logger.warning(
            "[Feishu] Shared dedup store unusable (%s) at %s; falling back to "
            "per-adapter deduplication. Redelivered messages may be processed "
            "once per profile.",
            action,
            self._db_path,
            exc_info=True,
        )

    def _prune(self, conn: sqlite3.Connection, now: float) -> None:
        """Drop TTL-expired rows, then trim the namespace to ``max_entries``."""
        try:
            with self._transaction(conn):
                if self._ttl_seconds > 0:
                    conn.execute(
                        "DELETE FROM seen_messages WHERE namespace = ? AND seen_at < ?",
                        (self._namespace, now - self._ttl_seconds),
                    )
                # Keep the most recent ids: those are the ones a redelivery is
                # actually likely to repeat.
                conn.execute(
                    "DELETE FROM seen_messages "
                    "WHERE namespace = ? AND message_id NOT IN ("
                    "    SELECT message_id FROM seen_messages "
                    "    WHERE namespace = ? ORDER BY seen_at DESC LIMIT ?"
                    ")",
                    (self._namespace, self._namespace, self._max_entries),
                )
        except sqlite3.Error:
            # A failed prune only costs disk; keep deduplicating.
            logger.debug("[Feishu] Shared dedup prune failed", exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seen(self, message_id: str) -> bool:
        """Record ``message_id`` and report whether it was already present.

        Atomic test-and-set: exactly one caller across all threads and
        processes observes ``False`` for a given id within the TTL window.
        Returns ``False`` (deliver the message) if the store is unusable.
        """
        if not message_id or self._disabled:
            return False

        now = time.time()
        with self._lock:
            try:
                with self._connect() as conn:
                    with self._transaction(conn):
                        # An entry past its TTL is not a duplicate — drop it
                        # first so the INSERT below re-records the id with a
                        # fresh timestamp, matching the previous cache's TTL
                        # semantics.
                        if self._ttl_seconds > 0:
                            conn.execute(
                                "DELETE FROM seen_messages "
                                "WHERE namespace = ? AND message_id = ? AND seen_at < ?",
                                (self._namespace, message_id, now - self._ttl_seconds),
                            )
                        cursor = conn.execute(
                            "INSERT OR IGNORE INTO seen_messages "
                            "(namespace, message_id, seen_at) VALUES (?, ?, ?)",
                            (self._namespace, message_id, now),
                        )
                        duplicate = cursor.rowcount == 0

                    if not duplicate:
                        self._writes_since_prune += 1
                        if self._writes_since_prune >= _PRUNE_EVERY:
                            self._writes_since_prune = 0
                            self._prune(conn, now)
            except sqlite3.Error:
                self._fail("record message id")
                return False

            return duplicate

    def import_legacy(self, entries: Mapping[str, float]) -> None:
        """Seed the shared store from a per-profile cache (one-time upgrade).

        Without this, upgrading mid-window would reopen the replay hole the
        old cache was still covering: its ids would be absent here and the
        next redelivery inside the 24h TTL would be treated as new.
        Idempotent — ``INSERT OR IGNORE`` keeps whichever row was written
        first, not the oldest ``seen_at``. Retaining a later timestamp only
        lengthens the TTL window for that id, which errs toward deduplicating,
        so it is not worth an upsert to correct.
        """
        if self._disabled or not entries:
            return

        ttl = self._ttl_seconds
        now = time.time()
        rows = []
        for message_id, seen_at in entries.items():
            if not message_id:
                continue
            try:
                recorded_at = float(seen_at)
            except (TypeError, ValueError):
                continue
            # recorded_at == 0.0 is the pre-TTL on-disk format, treated as
            # immortal for one migration cycle by the loader that produced it.
            if ttl > 0 and recorded_at != 0.0 and now - recorded_at >= ttl:
                continue
            rows.append((self._namespace, str(message_id), recorded_at))

        if not rows:
            return

        with self._lock:
            try:
                with self._connect() as conn:
                    with self._transaction(conn):
                        conn.executemany(
                            "INSERT OR IGNORE INTO seen_messages "
                            "(namespace, message_id, seen_at) VALUES (?, ?, ?)",
                            rows,
                        )
            except sqlite3.Error:
                self._fail("import legacy dedup state")

    def release(self, message_id: str) -> None:
        """Drop a previously claimed id so another caller may take it.

        ``seen()`` claims an id before the caller knows whether it will act on
        it. When the caller then declines for a reason of its own — a Feishu
        profile whose admission policy rejects a message a sibling profile
        would accept — the claim has to come back, or one profile's policy
        silently suppresses another's delivery.

        Claiming first and releasing on decline keeps the test-and-set atomic;
        checking admission before claiming would let two adapters both pass
        the check before either recorded.
        """
        if not message_id or self._disabled:
            return

        with self._lock:
            try:
                with self._connect() as conn:
                    with self._transaction(conn):
                        conn.execute(
                            "DELETE FROM seen_messages "
                            "WHERE namespace = ? AND message_id = ?",
                            (self._namespace, message_id),
                        )
            except sqlite3.Error:
                # Worst case the claim stands and a sibling skips one
                # message — not worth disabling the store over.
                logger.debug(
                    "[Feishu] Shared dedup release failed for %s", message_id, exc_info=True
                )

    def count(self) -> int:
        """Rows currently recorded for this namespace (diagnostics/tests)."""
        try:
            with self._connect() as conn:
                return int(
                    conn.execute(
                        "SELECT COUNT(*) FROM seen_messages WHERE namespace = ?",
                        (self._namespace,),
                    ).fetchone()[0]
                )
        except sqlite3.Error:
            return 0
