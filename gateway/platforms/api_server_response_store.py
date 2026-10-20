"""SQLite-backed store for Responses API state (``previous_response_id`` chaining, named conversations)."""

import json
import logging
import sqlite3
import time
from contextlib import closing, suppress
from pathlib import Path
from typing import Any, Optional

# Logger parity with the origin module (moved log records keep their name).
logger = logging.getLogger("gateway.platforms.api_server")

MAX_STORED_RESPONSES = 100
# Settled Idempotency-Key identities retained per response-body slot (see ResponseStore).
IDENTITY_RETENTION_FACTOR = 10
# The stored terminal envelope of a body row; a corrupt body reads as absent, never raises.
_RECORD = "json_extract(CASE WHEN json_valid(data) THEN data END, '$.response"


class ResponseStore:
    """SQLite-backed LRU store for Responses API state (full conversation history per response
    for ``previous_response_id`` chaining). Persists across restarts; in-memory fallback.

    Idempotency-Key identity (``response_keys`` + ``response_admissions`` +
    ``response_output_order``) deliberately outlives body LRU eviction so an exact retry still
    replays, conflicts or reconstructs its terminal result. Its own lifecycle is a count bound:
    at most ``_identity_bound()`` *settled* identities (a terminal idempotency record was
    stored), ``IDENTITY_RETENTION_FACTOR * max_size`` unless ``max_identities`` is given. Past
    the bound the oldest-settled are dropped from all three tables plus their ``idem:`` replay
    record, in the same transaction as body eviction. An unsettled identity (admission pending,
    running, or outcome unknown) neither counts nor is ever dropped; it joins the bound once an
    observer stores its record or, with no observer left, once its canonical admission is terminal. Public ``delete`` of a settled response drops its identity the same way.

    An expired or deleted key is not a new request: the canonical admission still carries the
    request id, so a reuse is refused with ``admission_conflict`` and never re-executes.
    """

    def __init__(self, max_size: int = MAX_STORED_RESPONSES, db_path: Optional[str] = None,
                 max_identities: Optional[int] = None):
        self._max_size = max_size
        self._max_identities = max_identities
        if db_path is None:
            db_path = ":memory:"
            with suppress(Exception):
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "response_store.db")
        self._db_path: Optional[str] = db_path if db_path != ":memory:" else None
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except Exception:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._db_path = None
        # Shared WAL-fallback so response_store.db degrades gracefully on NFS/SMB/FUSE homes.
        from hermes_state_wal import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="response_store.db")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "response_id TEXT PRIMARY KEY, data TEXT NOT NULL, accessed_at REAL NOT NULL)")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations (name TEXT PRIMARY KEY, response_id TEXT NOT NULL)")
        # Compact immutable request identity outlives the LRU response bodies.
        # Distinct canonical session targets must not turn a global retry key
        # into a last-writer-wins response cache entry.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS response_keys (request_key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, created_at INTEGER NOT NULL)")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS response_admissions (request_key TEXT PRIMARY KEY, admission_id TEXT NOT NULL)")
        self._conn.execute('CREATE TABLE IF NOT EXISTS response_output_order '
            '(request_key TEXT NOT NULL, item_id TEXT NOT NULL, output_index INTEGER NOT NULL, '
            'PRIMARY KEY(request_key,item_id), UNIQUE(request_key,output_index))')
        from hermes_cli.sqlite_util import add_column_if_missing
        add_column_if_missing(self._conn, 'response_keys', 'model_json', 'model_json TEXT')
        add_column_if_missing(self._conn, 'response_keys', 'settled_at', 'settled_at REAL')
        add_column_if_missing(self._conn, 'response_keys', 'response_id', 'response_id TEXT')
        self._conn.execute('CREATE INDEX IF NOT EXISTS response_keys_settled ON response_keys(settled_at)')
        self._conn.execute('CREATE INDEX IF NOT EXISTS response_keys_response ON response_keys(response_id)')
        # Rows written before settlement was tracked: settled iff their replay record survives.
        self._mark_settled('1', ())
        self._settle_terminal_admissions()
        # The identity bound holds from open, not only from the next write (a lowered bound or a
        # store that crashed past it would otherwise keep every settled identity until then).
        self._evict_and_commit()
        # Conversation history lives here: owner-only perms, once at init (not per commit).
        self._tighten_file_permissions()

    def _tighten_file_permissions(self) -> None:
        """Force owner-only permissions on the DB and SQLite sidecars."""
        if not self._db_path:
            return
        for candidate in (Path(self._db_path), Path(f"{self._db_path}-wal"), Path(f"{self._db_path}-shm")):
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug("Failed to restrict response store permissions for %s", candidate, exc_info=True)

    def get(self, response_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a stored response by ID (updates access time for LRU)."""
        row = self._conn.execute(
            "SELECT data FROM responses WHERE response_id = ?", (response_id,)).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE responses SET accessed_at = ? WHERE response_id = ?",
            (time.time(), response_id))
        self._conn.commit()
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupted JSON in response store for id=%s, evicting entry", response_id)
            self._conn.execute("DELETE FROM responses WHERE response_id = ?", (response_id,))
            self._conn.commit()
            return None

    def put(self, response_id: str, data: dict[str, Any]) -> None:
        """Store a response, evicting the oldest if at capacity."""
        insert = 'INSERT OR IGNORE' if response_id.startswith('idem:') else 'INSERT OR REPLACE'
        self._conn.execute(
            insert + " INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            (response_id, json.dumps(data, default=str), time.time()))
        if response_id.startswith('idem:'):
            self._mark_settled('request_key = ?', (response_id,))
        self._evict_and_commit()

    def _mark_settled(self, selector: str, params: tuple) -> None:
        """A stored terminal replay record settles its identity; only then may it age out."""
        self._conn.execute(
            f"UPDATE response_keys SET settled_at=?, response_id=(SELECT {_RECORD}.id') FROM responses "
            "WHERE responses.response_id=request_key) WHERE settled_at IS NULL AND "
            f"{selector} AND EXISTS (SELECT 1 FROM responses WHERE responses.response_id=request_key "
            f"AND {_RECORD}') IS NOT NULL)", (time.time(), *params))

    def _settle_terminal_admissions(self, at: Optional[float] = None) -> None:
        """Settle an identity whose canonical admission already finished but left no replay
        record: rows migrated from a release that did not track settlement (their record aged
        out of the old body LRU), or a request whose client left before the record was written
        (a disconnected stream's observer is gone, so nothing else would ever settle it).

        Runs at open and before every bound or delete decision, so it does not depend on any
        HTTP observer. The authority is this home's ``state.db`` (live row or retired
        tombstone), read-only. Only a terminal admission settles: at ``at``, or on open at its
        acceptance time so a migrated row ages out first. Its deterministic response id is
        recorded so ``DELETE /v1/responses/<id>`` reaches it. A queued/started/unknown one keeps
        its identity, so its exact retry is unchanged."""
        if not self._db_path:
            return
        pending = dict(self._conn.execute(
            'SELECT a.admission_id, k.request_key FROM response_keys k JOIN response_admissions a '
            'ON a.request_key=k.request_key WHERE k.settled_at IS NULL').fetchall())
        state_db = Path(self._db_path).parent / 'state.db'
        if not pending or not state_db.is_file():
            return
        from hermes_state_holders import read_only_db_uri
        from hermes_state_terminal import ADMISSION_PREFIX
        terminal = []
        ids = list(pending)
        try:
            with closing(sqlite3.connect(read_only_db_uri(state_db), uri=True)) as canon:
                for start in range(0, len(ids), 500):
                    chunk = ids[start:start + 500]
                    marks = ','.join('?' * len(chunk))
                    terminal += [row[0] for row in canon.execute(
                        f"SELECT admission_id FROM session_admissions WHERE status='terminal' "
                        f"AND admission_id IN ({marks})", chunk)]
                    terminal += [row[0][len(ADMISSION_PREFIX):] for row in canon.execute(
                        f'SELECT key FROM state_meta WHERE key IN ({marks})',
                        [ADMISSION_PREFIX + admission_id for admission_id in chunk])]
        except sqlite3.Error:
            # Unreadable canonical store: nothing is proven terminal, so nothing is settled.
            logger.debug('Responses identity settlement skipped: %s unreadable', state_db, exc_info=True)
            return
        from gateway.platforms.api_server_response_identity import durable_response_id
        self._conn.executemany(
            'UPDATE response_keys SET settled_at=COALESCE(?, created_at), response_id=COALESCE(response_id, ?) '
            'WHERE request_key=? AND settled_at IS NULL',
            [(at, durable_response_id(pending[a]), pending[a]) for a in set(terminal)])

    def _identity_bound(self) -> int:
        return self._max_identities or IDENTITY_RETENTION_FACTOR * self._max_size

    def _forget_identities(self, selector: str, params: tuple) -> int:
        """Drop the selected settled identities from every compact table and their replay record.

        Runs inside the caller's open transaction; ``response_keys`` goes last so the selector
        sees the same rows for each table."""
        keys = f'SELECT request_key FROM response_keys WHERE settled_at IS NOT NULL AND {selector}'
        for table, column in (('response_output_order', 'request_key'),
                              ('response_admissions', 'request_key'), ('responses', 'response_id')):
            self._conn.execute(f'DELETE FROM {table} WHERE {column} IN ({keys})', params)
        return self._conn.execute(f'DELETE FROM response_keys WHERE request_key IN ({keys})', params).rowcount

    def _evict_and_commit(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        if count > self._max_size:
            evict_ids = [row[0] for row in self._conn.execute(
                "SELECT response_id FROM responses ORDER BY accessed_at ASC LIMIT ?",
                (count - self._max_size,)).fetchall()]
            if evict_ids:
                placeholders = ",".join("?" for _ in evict_ids)
                # Conversation mappings pointing at evicted responses go too.
                self._conn.execute(f"DELETE FROM conversations WHERE response_id IN ({placeholders})", evict_ids)
                self._conn.execute(f"DELETE FROM responses WHERE response_id IN ({placeholders})", evict_ids)
        self._settle_terminal_admissions(time.time())
        excess = self._conn.execute(
            'SELECT COUNT(*) FROM response_keys WHERE settled_at IS NOT NULL').fetchone()[0] - self._identity_bound()
        if excess > 0:
            self._forget_identities('1 ORDER BY settled_at, rowid LIMIT ?', (excess,))
            # A late observer of an already-retired key may have re-allocated derivative rows.
            for table in ('response_output_order', 'response_admissions'):
                self._conn.execute(f'DELETE FROM {table} WHERE request_key NOT IN (SELECT request_key FROM response_keys)')
        self._conn.commit()

    def bind_request_key(self, key: str, fingerprint: str, *, model=None) -> bool:
        """Reserve one immutable digest before any canonical admission can run."""
        with self._conn:
            self._conn.execute('INSERT OR IGNORE INTO response_keys '
                '(request_key,fingerprint,created_at,model_json) VALUES (?,?,?,?)',
                (key, fingerprint, int(time.time()), json.dumps(model)))
            row = self._conn.execute('SELECT fingerprint FROM response_keys WHERE request_key=?', (key,)).fetchone()
        return row[0] == fingerprint

    def request_model(self, key: str, fallback):
        """The accepted wire label survives route changes and response-body LRU eviction."""
        with self._conn:
            # Upgrade older compact bindings only on a cache miss; their existing cached
            # envelopes keep replaying byte-for-byte through the ordinary fast path.
            self._conn.execute('UPDATE response_keys SET model_json=? WHERE request_key=? AND model_json IS NULL',
                               (json.dumps(fallback), key))
            row = self._conn.execute('SELECT model_json FROM response_keys WHERE request_key=?', (key,)).fetchone()
        return json.loads(row[0]) if row is not None else fallback

    def output_index(self, key: str, item_id: str) -> int:
        """Allocate before live publication; retries and parallel observers share this order.

        One compact id/integer pair per output item survives body LRU eviction. No tool output,
        reasoning text or cumulative conversation is copied into this identity table.
        """
        with self._conn:
            self._conn.execute('INSERT OR IGNORE INTO response_output_order '
                '(request_key,item_id,output_index) SELECT ?,?,COALESCE(MAX(output_index)+1,0) '
                'FROM response_output_order WHERE request_key=?', (key, item_id, key))
            row = self._conn.execute('SELECT output_index FROM response_output_order WHERE request_key=? AND item_id=?',
                                     (key, item_id)).fetchone()
        return row[0]

    def order_output(self, key: str, items):
        return sorted(items, key=lambda item: self.output_index(key, item['id']))

    def request_created_at(self, key: str) -> Optional[int]:
        row = self._conn.execute('SELECT created_at FROM response_keys WHERE request_key=?', (key,)).fetchone()
        return row[0] if row else None

    def request_key_matches(self, key: str, fingerprint: str) -> bool:
        row = self._conn.execute('SELECT fingerprint FROM response_keys WHERE request_key=?', (key,)).fetchone()
        return row is None or row[0] == fingerprint

    def request_admission(self, key: str) -> Optional[str]:
        row = self._conn.execute('SELECT admission_id FROM response_admissions WHERE request_key=?', (key,)).fetchone()
        return row[0] if row else None

    def bind_request_admission(self, key: str, admission_id: str) -> None:
        from hermes_state_runtime import RuntimeStoreError
        with self._conn:
            self._conn.execute('INSERT OR IGNORE INTO response_admissions VALUES (?,?)', (key, admission_id))
            if self.request_admission(key) != admission_id:
                raise RuntimeStoreError('admission_conflict')

    def forget_unadmitted_request(self, key: str) -> None:
        with self._conn:
            self._conn.execute('DELETE FROM response_keys WHERE request_key=? AND NOT EXISTS '
                '(SELECT 1 FROM response_admissions WHERE request_key=?)', (key, key))

    def delete(self, response_id: str) -> bool:
        """Remove a response, conversation mappings to it and its settled Idempotency-Key
        identity (replay record included). True if anything was found and deleted."""
        self._conn.execute("DELETE FROM conversations WHERE response_id = ?", (response_id,))
        cursor = self._conn.execute("DELETE FROM responses WHERE response_id = ?", (response_id,))
        self._settle_terminal_admissions(time.time())
        forgotten = self._forget_identities('response_id = ?', (response_id,))
        self._conn.commit()
        return cursor.rowcount > 0 or forgotten > 0

    def get_conversation(self, name: str) -> Optional[str]:
        """Get the latest response_id for a conversation name."""
        row = self._conn.execute("SELECT response_id FROM conversations WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def set_conversation(self, name: str, response_id: str) -> None:
        """Map a conversation name to its latest response_id."""
        self._conn.execute("INSERT OR REPLACE INTO conversations (name, response_id) VALUES (?, ?)", (name, response_id))
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        with suppress(Exception):
            self._conn.close()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return row[0] if row else 0
