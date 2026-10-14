"""Append-only SQLite journal with hash chain, transactions, snapshots.

Durability model:
  * SQLite in WAL mode with synchronous=FULL — every commit is fsynced.
    (On macOS additionally ``checkpoint_fullfsync=1``: Apple's plain
    ``fsync`` does not guarantee data-on-platter, mirroring the guard in
    ``hermes_state.py``.)
  * Records produced inside a runtime transaction are buffered in memory and
    written together with the TransactionCommit record in ONE SQLite
    transaction, so a crash at any point either persists the whole
    transaction or none of it (no partial-transaction replay needed).
  * Each record carries a sha256 checksum of its payload and a prev_hash
    linking it to the previous record (hash chain).  `verify_chain` detects
    torn tails / corruption and truncates back to the last valid prefix.
  * Every multi-statement write goes through the ``_txn`` context manager:
    a failed statement ROLLs BACK instead of wedging the shared connection
    inside an open transaction (isolation_level=None autocommit mode means
    an unhandled error would otherwise leave the transaction open and every
    later BEGIN would fail).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

GENESIS_HASH = b"\x00" * 32

# Record types
SESSION_START = "SessionStart"
USER_MESSAGE = "UserMessage"
ASSISTANT_MESSAGE = "AssistantMessage"
TOOL_CALL_INVOKED = "ToolCallInvoked"
TOOL_CALL_RESULT = "ToolCallResult"
OUTBOX_ENQUEUED = "OutboxEnqueued"
OUTBOX_DELIVERED = "OutboxDelivered"
TXN_COMMIT = "TransactionCommit"
COMPACTION_SNAPSHOT = "CompactionSnapshot"
COMPACTION_COMPLETE = "CompactionComplete"
GUARDRAIL_AUDIT = "GuardrailAudit"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    record_type    TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    transaction_id TEXT,
    payload        BLOB NOT NULL,
    checksum       BLOB NOT NULL,
    prev_hash      BLOB NOT NULL,
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_journal_session ON journal(session_id, seq);

CREATE TABLE IF NOT EXISTS outbox (
    outbox_id     TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    seq_hint      INTEGER NOT NULL,
    channel       TEXT NOT NULL,
    payload       BLOB NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    attempts      INTEGER NOT NULL DEFAULT 0,
    next_retry_at REAL NOT NULL DEFAULT 0,
    last_error    TEXT,
    created_at    REAL NOT NULL,
    delivered_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_outbox_pending ON outbox(session_id, status, seq_hint);

CREATE TABLE IF NOT EXISTS dlq (
    outbox_id  TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    channel    TEXT NOT NULL,
    payload    BLOB NOT NULL,
    error      TEXT,
    attempts   INTEGER NOT NULL,
    moved_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshot (
    snapshot_id TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    base_seq    INTEGER NOT NULL,
    state       BLOB NOT NULL,
    checksum    BLOB NOT NULL,
    complete    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    payload_id    TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    action        TEXT NOT NULL,
    policy_id     TEXT NOT NULL,
    envelope_hash BLOB NOT NULL,
    created_at    REAL NOT NULL
);
"""


@contextmanager
def _txn(conn: sqlite3.Connection):
    """BEGIN IMMEDIATE ... COMMIT with ROLLBACK on any error.

    Never let an exception escape between BEGIN and COMMIT without rolling
    back — with isolation_level=None the connection would stay inside the
    open transaction and every subsequent BEGIN would raise "cannot start a
    transaction within a transaction" until process restart.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


@dataclass
class Record:
    seq: int
    record_type: str
    session_id: str
    transaction_id: Optional[str]
    payload: dict
    checksum: bytes
    prev_hash: bytes
    created_at: float


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _checksum(record_type: str, session_id: str, transaction_id: Optional[str],
              payload_bytes: bytes, prev_hash: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(prev_hash)
    h.update(record_type.encode())
    h.update(session_id.encode())
    h.update((transaction_id or "").encode())
    h.update(payload_bytes)
    return h.digest()


class Journal:
    """Owns the SQLite database. Thread-safe; one writer at a time."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False,
                                     isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        if sys.platform == "darwin":
            try:
                self._conn.execute("PRAGMA checkpoint_fullfsync=1")
            except sqlite3.OperationalError:
                pass
        with self._lock:
            self._conn.executescript(_SCHEMA)
        self._tip_hash = self._load_tip_hash()

    # -- chain helpers -----------------------------------------------------
    def _load_tip_hash(self) -> bytes:
        row = self._conn.execute(
            "SELECT checksum FROM journal ORDER BY seq DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS_HASH

    def verify_chain(self, repair: bool = False) -> tuple[bool, int]:
        """Walk the chain; return (ok, first_bad_seq or -1).

        With repair=True, truncate the journal at the first corrupt record
        (torn-tail recovery) and roll back any dependent state.
        """
        with self._lock:
            prev = GENESIS_HASH
            cur = self._conn.execute(
                "SELECT seq, record_type, session_id, transaction_id, payload,"
                " checksum, prev_hash FROM journal ORDER BY seq")
            for seq, rtype, sid, txid, payload, cksum, prev_hash in cur:
                expected = _checksum(rtype, sid, txid, payload, prev)
                if prev_hash != prev or cksum != expected:
                    if repair:
                        with _txn(self._conn):
                            self._conn.execute(
                                "DELETE FROM journal WHERE seq >= ?", (seq,))
                        self._tip_hash = self._load_tip_hash()
                    return False, seq
                prev = cksum
            return True, -1

    # -- transactions ------------------------------------------------------
    def begin(self, session_id: str) -> "JournalTransaction":
        return JournalTransaction(self, session_id)

    def _commit_records(self, session_id: str, transaction_id: str,
                        records: list[tuple[str, dict]],
                        outbox_rows: list[tuple[str, str, dict]],
                        extra_sql: Optional[list[tuple[str, tuple]]] = None,
                        ) -> list[int]:
        """Atomically append `records` + TransactionCommit, insert outbox
        rows, and run `extra_sql` statements in one fsynced SQLite
        transaction. Returns assigned seqs.

        `extra_sql` lets callers (e.g. the outbox worker's delivered-mark)
        make a journal append and a table update atomic — a crash can never
        observe one without the other.
        """
        now = time.time()
        with self._lock:
            prev = self._tip_hash
            with _txn(self._conn):
                seqs = []
                all_records = records + [(TXN_COMMIT, {"txn": transaction_id})]
                for rtype, payload in all_records:
                    pb = _canonical(payload)
                    cksum = _checksum(rtype, session_id, transaction_id, pb, prev)
                    cur = self._conn.execute(
                        "INSERT INTO journal (record_type, session_id,"
                        " transaction_id, payload, checksum, prev_hash,"
                        " created_at) VALUES (?,?,?,?,?,?,?)",
                        (rtype, session_id, transaction_id, pb, cksum, prev, now))
                    seqs.append(cur.lastrowid)
                    prev = cksum
                for outbox_id, channel, payload in outbox_rows:
                    self._conn.execute(
                        "INSERT INTO outbox (outbox_id, session_id, seq_hint,"
                        " channel, payload, status, created_at)"
                        " VALUES (?,?,?,?,?,'pending',?)",
                        (outbox_id, session_id, seqs[-1], channel,
                         _canonical(payload), now))
                for sql, params in (extra_sql or []):
                    self._conn.execute(sql, params)
            self._tip_hash = prev
            return seqs

    def append(self, session_id: str, record_type: str, payload: dict,
               transaction_id: Optional[str] = None,
               extra_sql: Optional[list[tuple[str, tuple]]] = None) -> int:
        """Append a single standalone (auto-committed) record, optionally
        with extra SQL statements in the same atomic transaction."""
        return self._commit_records(
            session_id, transaction_id or str(uuid.uuid4()),
            [(record_type, payload)], [], extra_sql=extra_sql)[0]

    # -- reads -------------------------------------------------------------
    def records(self, session_id: Optional[str] = None,
                after_seq: int = 0) -> Iterator[Record]:
        q = ("SELECT seq, record_type, session_id, transaction_id, payload,"
             " checksum, prev_hash, created_at FROM journal WHERE seq > ?")
        args: list[Any] = [after_seq]
        if session_id:
            q += " AND session_id = ?"
            args.append(session_id)
        q += " ORDER BY seq"
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        for r in rows:
            yield Record(r[0], r[1], r[2], r[3], json.loads(r[4]), r[5], r[6], r[7])

    def committed_transactions(self, session_id: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT transaction_id FROM journal"
                " WHERE session_id = ? AND record_type = ?",
                (session_id, TXN_COMMIT)).fetchall()
        return {r[0] for r in rows}

    def enqueued_outbox_ids(self) -> set[str]:
        """outbox_ids with a surviving OutboxEnqueued journal record.

        Exact-match by parsing the canonical JSON payloads — no SQL LIKE
        (an id containing a wildcard would false-positive, and substring
        matches are not identity checks).
        """
        ids: set[str] = set()
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM journal WHERE record_type = ?",
                (OUTBOX_ENQUEUED,)).fetchall()
        for (payload,) in rows:
            try:
                oid = json.loads(payload).get("outbox_id")
            except (ValueError, TypeError):
                continue
            if isinstance(oid, str):
                ids.add(oid)
        return ids

    # -- snapshots / compaction -------------------------------------------
    def write_snapshot(self, session_id: str, base_seq: int, state: dict) -> str:
        """Two-phase snapshot: write row (complete=0), fsync, mark complete.
        Recovery ignores snapshots with complete=0."""
        snapshot_id = str(uuid.uuid4())
        sb = _canonical(state)
        cksum = hashlib.sha256(sb).digest()
        with self._lock:
            with _txn(self._conn):
                self._conn.execute(
                    "INSERT INTO snapshot (snapshot_id, session_id, base_seq,"
                    " state, checksum, complete, created_at)"
                    " VALUES (?,?,?,?,?,0,?)",
                    (snapshot_id, session_id, base_seq, sb, cksum, time.time()))
            self.append(session_id, COMPACTION_SNAPSHOT,
                        {"snapshot_id": snapshot_id, "base_seq": base_seq})
            self.append(session_id, COMPACTION_COMPLETE,
                        {"snapshot_id": snapshot_id, "base_seq": base_seq},
                        extra_sql=[(
                            "UPDATE snapshot SET complete = 1"
                            " WHERE snapshot_id = ?", (snapshot_id,))])
        return snapshot_id

    def latest_snapshot(self, session_id: str) -> Optional[tuple[int, dict]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT base_seq, state, checksum FROM snapshot"
                " WHERE session_id = ? AND complete = 1"
                " ORDER BY base_seq DESC LIMIT 1", (session_id,)).fetchone()
        if not row:
            return None
        base_seq, state, cksum = row
        if hashlib.sha256(state).digest() != cksum:
            return None
        return base_seq, json.loads(state)

    def compact(self, session_id: str, state: dict,
                keep_after_seq: Optional[int] = None) -> str:
        """Snapshot current state, then delete journal records the snapshot
        covers (except outbox-relevant records still pending).

        The DELETE and the hash-chain re-root happen in ONE SQLite
        transaction: a crash between them would otherwise leave surviving
        records whose prev_hash points at deleted rows, and the next
        verify_chain(repair=True) would truncate the entire surviving
        journal — silently dropping fsync-durable state.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(seq) FROM journal WHERE session_id = ?",
                (session_id,)).fetchone()
            base_seq = row[0] or 0
            sid = self.write_snapshot(session_id, base_seq, state)
            cutoff = keep_after_seq if keep_after_seq is not None else base_seq
            with _txn(self._conn):
                self._conn.execute(
                    "DELETE FROM journal WHERE session_id = ? AND seq <= ?"
                    " AND record_type NOT IN (?,?,?,?)",
                    (session_id, cutoff, COMPACTION_SNAPSHOT,
                     COMPACTION_COMPLETE, OUTBOX_ENQUEUED, OUTBOX_DELIVERED))
                # Chain is intentionally re-rooted after compaction (same
                # transaction as the DELETE): the snapshot checksum vouches
                # for the deleted prefix.
                prev = GENESIS_HASH
                rows = self._conn.execute(
                    "SELECT seq, record_type, session_id, transaction_id,"
                    " payload FROM journal ORDER BY seq").fetchall()
                for seq, rtype, rsid, txid, payload in rows:
                    cksum = _checksum(rtype, rsid, txid, payload, prev)
                    self._conn.execute(
                        "UPDATE journal SET checksum = ?, prev_hash = ?"
                        " WHERE seq = ?", (cksum, prev, seq))
                    prev = cksum
            self._tip_hash = prev
        return sid

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class JournalTransaction:
    """Buffers records in memory; nothing is durable until commit()."""

    def __init__(self, journal: Journal, session_id: str):
        self.journal = journal
        self.session_id = session_id
        self.transaction_id = str(uuid.uuid4())
        self._records: list[tuple[str, dict]] = []
        self._outbox: list[tuple[str, str, dict]] = []
        self._done = False

    def record(self, record_type: str, payload: dict) -> None:
        assert not self._done, "transaction already finished"
        self._records.append((record_type, payload))

    def enqueue_outbound(self, channel: str, payload: dict,
                         outbox_id: Optional[str] = None) -> str:
        assert not self._done, "transaction already finished"
        outbox_id = outbox_id or str(uuid.uuid4())
        self._records.append((OUTBOX_ENQUEUED, {
            "outbox_id": outbox_id, "channel": channel, "payload": payload}))
        self._outbox.append((outbox_id, channel, payload))
        return outbox_id

    def commit(self) -> list[int]:
        assert not self._done, "transaction already finished"
        self._done = True
        return self.journal._commit_records(
            self.session_id, self.transaction_id, self._records, self._outbox)

    def rollback(self) -> None:
        self._done = True
        self._records.clear()
        self._outbox.clear()

    def __enter__(self) -> "JournalTransaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._done:
            return
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
