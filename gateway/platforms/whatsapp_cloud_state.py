"""Durable replay claims and one-shot callback capabilities for WhatsApp Cloud."""

from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.sqlite_util import open_db, write_txn


logger = logging.getLogger("gateway.platforms.whatsapp_cloud")

# Meta may retry webhooks for seven days. Keep the full documented retry horizon,
# rather than an in-memory FIFO whose protection disappears on process restart.
REPLAY_RETENTION_SECONDS = 8 * 24 * 60 * 60

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS whatsapp_cloud_wamids (
           wamid TEXT PRIMARY KEY,
           seen_at REAL NOT NULL
       )""",
    """CREATE TABLE IF NOT EXISTS whatsapp_cloud_callbacks (
           capability_digest BLOB PRIMARY KEY,
           generation TEXT NOT NULL,
           kind TEXT NOT NULL,
           target_id TEXT NOT NULL,
           session_key TEXT NOT NULL,
           chat_id TEXT NOT NULL,
           user_id TEXT NOT NULL,
           message_id TEXT,
           created_at REAL NOT NULL,
           consumed_at REAL
       )""",
    "CREATE INDEX IF NOT EXISTS whatsapp_cloud_wamids_seen ON whatsapp_cloud_wamids(seen_at)",
    "CREATE INDEX IF NOT EXISTS whatsapp_cloud_callbacks_created ON whatsapp_cloud_callbacks(created_at)",
)


@dataclass(frozen=True)
class CallbackClaim:
    """Result of atomically checking and optionally consuming one callback capability."""

    status: str
    target_id: str | None = None
    session_key: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def _initialize(conn: sqlite3.Connection) -> None:
    for statement in _SCHEMA:
        conn.execute(statement)
    conn.commit()


def _capability_digest(capability: str) -> bytes:
    return hashlib.sha256(capability.encode("utf-8", "surrogatepass")).digest()


class WhatsAppCloudReplayStore:
    """SQLite-backed callback generation fence and inbound WAMID claim ledger."""

    def __init__(
        self, db_path: str | Path | None = None, *, generation: str | None = None
    ):
        if db_path is None:
            from hermes_constants import get_hermes_home

            db_path = get_hermes_home() / "state" / "whatsapp_cloud_replay.db"
        self._db_path = None if str(db_path) == ":memory:" else Path(db_path)
        self.generation = generation or secrets.token_hex(16)
        self._lock = threading.RLock()
        try:
            if self._db_path is None:
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                _initialize(self._conn)
            else:
                self._conn = open_db(
                    self._db_path,
                    db_label="whatsapp_cloud_replay.db",
                    busy_timeout_ms=10_000,
                    check_same_thread=False,
                    initialize=_initialize,
                )
                self._tighten_permissions()
        except Exception as exc:
            logger.error(
                "[whatsapp_cloud] durable callback replay storage unavailable; "
                "interactive callbacks will be generation-local: %s",
                exc,
            )
            self._db_path = None
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            _initialize(self._conn)

    @property
    def durable(self) -> bool:
        return self._db_path is not None

    def _tighten_permissions(self) -> None:
        for suffix in ("", "-wal", "-shm") if self._db_path is not None else ():
            candidate = Path(f"{self._db_path}{suffix}")
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug(
                    "[whatsapp_cloud] failed to restrict replay-store permissions",
                    exc_info=True,
                )

    def _prune_locked(self, now: float) -> None:
        cutoff = now - REPLAY_RETENTION_SECONDS
        self._conn.execute(
            "DELETE FROM whatsapp_cloud_wamids WHERE seen_at < ?", (cutoff,)
        )
        self._conn.execute(
            "DELETE FROM whatsapp_cloud_callbacks WHERE created_at < ?", (cutoff,)
        )

    def claim_wamid(self, wamid: str) -> bool:
        """Atomically claim an inbound message id; False means it was already admitted."""
        if not wamid:
            return True
        now = time.time()
        with self._lock, write_txn(self._conn):
            self._prune_locked(now)
            inserted = self._conn.execute(
                "INSERT OR IGNORE INTO whatsapp_cloud_wamids(wamid, seen_at) VALUES(?, ?)",
                (wamid, now),
            ).rowcount
        return inserted == 1

    def reserve_callback(
        self,
        *,
        kind: str,
        target_id: str,
        session_key: str,
        chat_id: str,
        user_id: str,
    ) -> str:
        """Mint and persist an unguessable callback capability before its message is sent."""
        now = time.time()
        with self._lock, write_txn(self._conn):
            self._prune_locked(now)
            for _ in range(8):
                capability = secrets.token_urlsafe(18)
                try:
                    self._conn.execute(
                        """INSERT INTO whatsapp_cloud_callbacks(
                               capability_digest, generation, kind, target_id, session_key,
                               chat_id, user_id, created_at
                           ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            _capability_digest(capability),
                            self.generation,
                            kind,
                            str(target_id),
                            str(session_key),
                            str(chat_id),
                            str(user_id),
                            now,
                        ),
                    )
                    return capability
                except sqlite3.IntegrityError:
                    continue
        raise RuntimeError("could not mint a unique WhatsApp callback capability")

    def activate_callback(self, capability: str, message_id: str | None) -> bool:
        """Bind a reserved capability to the exact outbound prompt message."""
        if not message_id:
            self.discard_callback(capability)
            return False
        with self._lock, write_txn(self._conn):
            changed = self._conn.execute(
                """UPDATE whatsapp_cloud_callbacks
                      SET message_id=?
                    WHERE capability_digest=? AND generation=?
                      AND message_id IS NULL AND consumed_at IS NULL""",
                (str(message_id), _capability_digest(capability), self.generation),
            ).rowcount
        return changed == 1

    def discard_callback(self, capability: str) -> None:
        with self._lock, write_txn(self._conn):
            self._conn.execute(
                "DELETE FROM whatsapp_cloud_callbacks WHERE capability_digest=? AND generation=?",
                (_capability_digest(capability), self.generation),
            )

    def claim_callback(
        self,
        capability: str,
        *,
        kind: str,
        user_id: str,
        chat_id: str,
        message_id: str,
    ) -> CallbackClaim:
        """Consume only a live capability bound to this generation, kind and origin tuple."""
        now = time.time()
        digest = _capability_digest(capability)
        with self._lock, write_txn(self._conn):
            self._prune_locked(now)
            row = self._conn.execute(
                """SELECT generation, kind, target_id, session_key, chat_id, user_id,
                          message_id, consumed_at
                     FROM whatsapp_cloud_callbacks
                    WHERE capability_digest=?""",
                (digest,),
            ).fetchone()
            if row is None:
                return CallbackClaim("missing")
            if row["generation"] != self.generation:
                return CallbackClaim("stale")
            if row["consumed_at"] is not None:
                return CallbackClaim("consumed")
            expected = (row["kind"], row["user_id"], row["chat_id"], row["message_id"])
            presented = (str(kind), str(user_id), str(chat_id), str(message_id))
            if not row["message_id"] or expected != presented:
                return CallbackClaim("foreign")
            changed = self._conn.execute(
                """UPDATE whatsapp_cloud_callbacks SET consumed_at=?
                    WHERE capability_digest=? AND generation=? AND consumed_at IS NULL""",
                (now, digest, self.generation),
            ).rowcount
            if changed != 1:
                return CallbackClaim("consumed")
            return CallbackClaim(
                "accepted",
                target_id=str(row["target_id"]),
                session_key=str(row["session_key"]),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
