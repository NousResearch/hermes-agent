"""Durable idempotency, nonce, dispatch, and audit journal."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from uuid import uuid4

from .errors import AdapterError


class DispatchStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nonces (
                    key_id TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    PRIMARY KEY (key_id, nonce)
                );
                CREATE TABLE IF NOT EXISTS dispatches (
                    dispatch_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_sha256 TEXT NOT NULL,
                    cycle_id TEXT NOT NULL,
                    principal TEXT NOT NULL,
                    request_json TEXT,
                    packet_json TEXT,
                    packet_sha256 TEXT,
                    reservation_event_id TEXT,
                    phase TEXT NOT NULL,
                    task_id TEXT,
                    result_json TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    dispatch_id TEXT,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS dispatch_reservations (
                    dispatch_id TEXT PRIMARY KEY,
                    packet_json TEXT NOT NULL,
                    packet_sha256 TEXT NOT NULL,
                    reservation_event_id TEXT NOT NULL UNIQUE,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(dispatch_id) REFERENCES dispatches(dispatch_id),
                    CHECK(length(packet_sha256) = 64)
                );
                CREATE TRIGGER IF NOT EXISTS dispatch_reservations_immutable_update
                BEFORE UPDATE ON dispatch_reservations
                BEGIN
                    SELECT RAISE(ABORT, 'dispatch reservation is immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS dispatch_reservations_immutable_delete
                BEFORE DELETE ON dispatch_reservations
                BEGIN
                    SELECT RAISE(ABORT, 'dispatch reservation is immutable');
                END;
                """
            )
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(dispatches)")
            }
            if "packet_json" not in columns:
                conn.execute("ALTER TABLE dispatches ADD COLUMN packet_json TEXT")
            if "packet_sha256" not in columns:
                conn.execute("ALTER TABLE dispatches ADD COLUMN packet_sha256 TEXT")
            if "reservation_event_id" not in columns:
                conn.execute("ALTER TABLE dispatches ADD COLUMN reservation_event_id TEXT")
            conn.execute(
                """
                INSERT OR IGNORE INTO dispatch_reservations(
                    dispatch_id, packet_json, packet_sha256,
                    reservation_event_id, created_at
                )
                SELECT dispatch_id, packet_json, packet_sha256,
                       reservation_event_id, created_at
                FROM dispatches
                WHERE packet_json IS NOT NULL
                  AND packet_sha256 IS NOT NULL
                  AND reservation_event_id IS NOT NULL
                """
            )
        self.path.chmod(0o600)

    def consume_nonce(
        self, key_id: str, nonce: str, expires_at: int, *, now: int | None = None
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM nonces WHERE expires_at < ?",
                (int(time.time()) if now is None else now,),
            )
            try:
                conn.execute(
                    "INSERT INTO nonces(key_id, nonce, expires_at) VALUES (?, ?, ?)",
                    (key_id, nonce, expires_at),
                )
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise AdapterError(
                    "REPLAY_REJECTED", "request nonce has already been used"
                ) from exc
            conn.commit()

    def reserve(
        self,
        dispatch_id: str,
        idempotency_key: str,
        request_sha256: str,
        cycle_id: str,
        principal: str,
        request: dict | None = None,
        packet: dict | None = None,
    ) -> tuple[dict, bool]:
        now = int(time.time())
        packet_json = (
            json.dumps(packet, sort_keys=True, separators=(",", ":"))
            if packet is not None
            else None
        )
        packet_sha256 = (
            hashlib.sha256(packet_json.encode()).hexdigest() if packet_json else None
        )
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM dispatches WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                if existing["request_sha256"] != request_sha256:
                    conn.rollback()
                    raise AdapterError(
                        "IDEMPOTENCY_CONFLICT",
                        "idempotency key is bound to a different canonical request",
                    )
                if existing["principal"] != principal:
                    conn.rollback()
                    raise AdapterError(
                        "IDEMPOTENCY_CONFLICT",
                        "idempotency key belongs to another principal",
                    )
                conn.commit()
                return dict(existing), False
            collision = conn.execute(
                "SELECT principal, idempotency_key, request_sha256 "
                "FROM dispatches WHERE dispatch_id = ?",
                (dispatch_id,),
            ).fetchone()
            if collision:
                conn.rollback()
                raise AdapterError(
                    "IDEMPOTENCY_CONFLICT",
                    "dispatch identity is already bound to another request",
                )
            try:
                conn.execute(
                    """
                    INSERT INTO dispatches(
                        dispatch_id,idempotency_key,request_sha256,cycle_id,
                        principal,request_json,packet_json,packet_sha256,
                        reservation_event_id,phase,created_at,updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'RESERVED', ?, ?)
                    """,
                    (
                        dispatch_id,
                        idempotency_key,
                        request_sha256,
                        cycle_id,
                        principal,
                        json.dumps(request, sort_keys=True) if request else None,
                        packet_json,
                        packet_sha256,
                        (event_id := self.new_event_id()),
                        now,
                        now,
                    ),
                )
                if packet_json is not None and packet_sha256 is not None:
                    conn.execute(
                        """
                        INSERT INTO dispatch_reservations(
                            dispatch_id,packet_json,packet_sha256,
                            reservation_event_id,created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            dispatch_id,
                            packet_json,
                            packet_sha256,
                            event_id,
                            now,
                        ),
                    )
                self._insert_audit(
                    conn,
                    event_id=event_id,
                    dispatch_id=dispatch_id,
                    kind="DISPATCH_RESERVED",
                    payload={
                        "principal": principal,
                        "request_sha256": request_sha256,
                        "packet_sha256": packet_sha256,
                        "packet_bytes": packet_json,
                    },
                    created_at=now,
                )
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise AdapterError(
                    "IDEMPOTENCY_CONFLICT",
                    "dispatch or idempotency identity already exists",
                ) from exc
            row = conn.execute(
                "SELECT * FROM dispatches WHERE dispatch_id = ?", (dispatch_id,)
            ).fetchone()
            conn.commit()
            return dict(row), True

    def assert_packet_identity(
        self, dispatch_id: str, packet_json: str | None = None
    ) -> dict:
        """Compare every packet projection with the immutable reservation."""
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM dispatches WHERE dispatch_id=?", (dispatch_id,)
            ).fetchone()
            reservation = conn.execute(
                "SELECT * FROM dispatch_reservations WHERE dispatch_id=?",
                (dispatch_id,),
            ).fetchone()
            if not row or not reservation:
                conn.rollback()
                raise AdapterError("DISPATCH_STATE_UNKNOWN", "dispatch is missing")
            authoritative_json = reservation["packet_json"]
            authoritative_sha = reservation["packet_sha256"]
            audit = conn.execute(
                "SELECT kind,payload_json FROM audit_events WHERE event_id=?",
                (reservation["reservation_event_id"],),
            ).fetchone()
            audit_payload = json.loads(audit["payload_json"]) if audit else {}
            observed_json = row["packet_json"] if packet_json is None else packet_json
            observed_sha = (
                hashlib.sha256(observed_json.encode()).hexdigest()
                if isinstance(observed_json, str)
                else ""
            )
            mismatch = bool(
                hashlib.sha256(authoritative_json.encode()).hexdigest()
                != authoritative_sha
                or row["packet_json"] != authoritative_json
                or row["packet_sha256"] != authoritative_sha
                or observed_json != authoritative_json
                or observed_sha != authoritative_sha
                or row["reservation_event_id"] != reservation["reservation_event_id"]
                or not audit
                or audit["kind"] != "DISPATCH_RESERVED"
                or audit_payload.get("packet_bytes") != authoritative_json
                or audit_payload.get("packet_sha256") != authoritative_sha
            )
            if mismatch:
                self._insert_audit(
                    conn,
                    event_id=self.new_event_id(),
                    dispatch_id=dispatch_id,
                    kind="EXECUTION_PACKET_CONFLICT",
                    payload={
                        "expected_sha256": authoritative_sha,
                        "observed_sha256": observed_sha,
                    },
                    created_at=int(time.time()),
                )
                conn.execute(
                    "UPDATE dispatches SET phase='PACKET_CONFLICT', updated_at=? WHERE dispatch_id=?",
                    (int(time.time()), dispatch_id),
                )
                conn.commit()
                raise AdapterError("CONTRACT_MISMATCH", "execution packet identity mismatch")
            conn.commit()
            value = dict(row)
            value["packet_json"] = authoritative_json
            value["packet_sha256"] = authoritative_sha
            value["reservation_event_id"] = reservation["reservation_event_id"]
            return value

    def _insert_audit(
        self,
        conn: sqlite3.Connection,
        *,
        event_id: str,
        dispatch_id: str | None,
        kind: str,
        payload: dict,
        created_at: int,
    ) -> None:
        redacted = self._redact(payload)
        previous = conn.execute(
            "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_hash = previous["event_hash"] if previous else "0" * 64
        material = json.dumps(
            {
                "event_id": event_id,
                "dispatch_id": dispatch_id,
                "kind": kind,
                "payload": redacted,
                "created_at": created_at,
                "previous_hash": previous_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        event_hash = hashlib.sha256(material).hexdigest()
        conn.execute(
            """
            INSERT INTO audit_events(
                event_id,dispatch_id,kind,payload_json,created_at,
                previous_hash,event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                dispatch_id,
                kind,
                json.dumps(redacted, sort_keys=True),
                created_at,
                previous_hash,
                event_hash,
            ),
        )

    def update(
        self,
        dispatch_id: str,
        *,
        phase: str,
        task_id: str | None = None,
        result: dict | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE dispatches
                SET phase=?, task_id=COALESCE(?, task_id),
                    result_json=COALESCE(?, result_json), updated_at=?
                WHERE dispatch_id=?
                """,
                (
                    phase,
                    task_id,
                    json.dumps(result, sort_keys=True) if result else None,
                    int(time.time()),
                    dispatch_id,
                ),
            )

    def get(self, dispatch_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dispatches WHERE dispatch_id=?", (dispatch_id,)
            ).fetchone()
        if not row:
            return None
        value = dict(row)
        if value.get("result_json"):
            value["result"] = json.loads(value["result_json"])
        return value

    def get_by_task(self, task_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dispatches WHERE task_id=?", (task_id,)
            ).fetchone()
        if not row:
            return None
        value = dict(row)
        if value.get("result_json"):
            value["result"] = json.loads(value["result_json"])
        return value

    def audit(self, kind: str, dispatch_id: str | None, payload: dict) -> str:
        event_id = self.new_event_id()
        redacted = self._redact(payload)
        created_at = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            previous = conn.execute(
                "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = previous["event_hash"] if previous else "0" * 64
            material = json.dumps(
                {
                    "event_id": event_id,
                    "dispatch_id": dispatch_id,
                    "kind": kind,
                    "payload": redacted,
                    "created_at": created_at,
                    "previous_hash": previous_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            event_hash = hashlib.sha256(material).hexdigest()
            conn.execute(
                """
                INSERT INTO audit_events(
                    event_id,dispatch_id,kind,payload_json,created_at,
                    previous_hash,event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    dispatch_id,
                    kind,
                    json.dumps(redacted, sort_keys=True),
                    created_at,
                    previous_hash,
                    event_hash,
                ),
            )
            conn.commit()
        return event_id

    @staticmethod
    def new_event_id() -> str:
        return f"builder_{uuid4().hex}"

    def transition_with_audit(
        self,
        dispatch_id: str,
        *,
        phase: str,
        event_id: str,
        kind: str,
        payload: dict,
        task_id: str | None = None,
        result: dict | None = None,
        expected_principal: str | None = None,
        expected_idempotency_key: str | None = None,
        expected_request_sha256: str | None = None,
        expected_phase: str | None = None,
    ) -> None:
        """Commit a state transition and its chained audit event atomically."""
        now = int(time.time())
        redacted = self._redact(payload)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            guards = ""
            values: list[object] = [
                phase,
                task_id,
                json.dumps(result, sort_keys=True) if result else None,
                now,
                dispatch_id,
            ]
            if expected_principal is not None:
                guards += " AND principal=?"
                values.append(expected_principal)
            if expected_idempotency_key is not None:
                guards += " AND idempotency_key=?"
                values.append(expected_idempotency_key)
            if expected_request_sha256 is not None:
                guards += " AND request_sha256=?"
                values.append(expected_request_sha256)
            if expected_phase is not None:
                guards += " AND phase=?"
                values.append(expected_phase)
            changed = conn.execute(
                """
                UPDATE dispatches
                SET phase=?, task_id=COALESCE(?, task_id),
                    result_json=COALESCE(?, result_json), updated_at=?
                WHERE dispatch_id=?
                """ + guards,
                values,
            )
            if changed.rowcount != 1:
                conn.rollback()
                raise AdapterError(
                    "DISPATCH_STATE_UNKNOWN", "state transition target missing"
                )
            previous = conn.execute(
                "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = previous["event_hash"] if previous else "0" * 64
            material = json.dumps(
                {
                    "event_id": event_id,
                    "dispatch_id": dispatch_id,
                    "kind": kind,
                    "payload": redacted,
                    "created_at": now,
                    "previous_hash": previous_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            event_hash = hashlib.sha256(material).hexdigest()
            conn.execute(
                """
                INSERT INTO audit_events(
                    event_id,dispatch_id,kind,payload_json,created_at,
                    previous_hash,event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    dispatch_id,
                    kind,
                    json.dumps(redacted, sort_keys=True),
                    now,
                    previous_hash,
                    event_hash,
                ),
            )
            conn.commit()

    @classmethod
    def _redact(cls, value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                lowered = str(key).lower()
                if any(
                    marker in lowered
                    for marker in ("secret", "token", "password", "credential", "api_key")
                ):
                    result[key] = "[REDACTED]"
                else:
                    result[key] = cls._redact(item)
            return result
        if isinstance(value, list):
            return [cls._redact(item) for item in value]
        return value

    def reject_reserved(
        self,
        dispatch_id: str,
        result: dict,
        *,
        principal: str,
        idempotency_key: str,
        request_sha256: str,
        current_phase: str = "RESERVED",
    ) -> bool:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                """
                UPDATE dispatches SET phase='REJECTED', result_json=?, updated_at=?
                WHERE dispatch_id=? AND principal=? AND idempotency_key=?
                  AND request_sha256=? AND phase=? AND task_id IS NULL
                """,
                (
                    json.dumps(result, sort_keys=True),
                    int(time.time()),
                    dispatch_id,
                    principal,
                    idempotency_key,
                    request_sha256,
                    current_phase,
                ),
            )
            conn.commit()
            return changed.rowcount == 1
