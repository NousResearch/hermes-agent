# -*- coding: utf-8 -*-
"""Append-only SQLite persistence for Phase 2 execution events."""
from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from ._contract import (
    EventEnvelope,
    EventType,
    LedgerConflictError,
    LedgerContractError,
    VALID_TRANSITIONS,
)


class Ledger:
    """Persist validated events with atomic idempotency and readback."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    linear_issue_id TEXT,
                    run_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    evidence_refs_json TEXT NOT NULL,
                    blocked_reason TEXT,
                    terminal_state TEXT,
                    payload_json TEXT NOT NULL,
                    previous_event_id TEXT,
                    FOREIGN KEY (previous_event_id) REFERENCES events(event_id)
                );
                CREATE INDEX IF NOT EXISTS idx_events_run_sequence
                    ON events(run_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_events_project
                    ON events(project_id);
                """
            )

    def append(self, event: EventEnvelope) -> dict[str, Any]:
        """Append an event, or return the existing record for an exact retry."""
        event.validate()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing_row = connection.execute(
                "SELECT * FROM events WHERE idempotency_key = ?",
                (event.idempotency_key,),
            ).fetchone()
            if existing_row is not None:
                existing = self._row_to_dict(existing_row)
                if self._fingerprint(existing) != event.idempotency_fingerprint():
                    raise LedgerConflictError(
                        f"idempotency key reused with divergent payload: {event.idempotency_key}"
                    )
                connection.commit()
                return copy.deepcopy(existing)

            event_id_row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event.event_id,)
            ).fetchone()
            if event_id_row is not None:
                raise LedgerConflictError(f"event_id already belongs to another event: {event.event_id}")

            self._validate_sequence_and_transition(connection, event)
            connection.execute(
                """
                INSERT INTO events (
                    event_id, event_type, schema_version, project_id,
                    linear_issue_id, run_id, agent_id, occurred_at,
                    recorded_at, sequence, idempotency_key, evidence_refs_json,
                    blocked_reason, terminal_state, payload_json, previous_event_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.schema_version,
                    event.project_id,
                    event.linear_issue_id,
                    event.run_id,
                    event.agent_id,
                    event.occurred_at,
                    event.recorded_at,
                    event.sequence,
                    event.idempotency_key,
                    json.dumps(list(event.evidence_refs), sort_keys=True, separators=(",", ":")),
                    event.blocked_reason,
                    event.terminal_state,
                    json.dumps(event.payload, sort_keys=True, separators=(",", ":")),
                    event.previous_event_id,
                ),
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            existing = self.read_by_idempotency_key(event.idempotency_key)
            if existing is not None and self._fingerprint(existing) == event.idempotency_fingerprint():
                return existing
            raise LedgerConflictError("atomic event insert conflicted") from exc
        finally:
            connection.close()

        readback = self.read(event.event_id)
        if readback is None:
            raise LedgerContractError("append committed without exact readback")
        if self._fingerprint(readback) != event.idempotency_fingerprint():
            raise LedgerContractError("append readback differs from submitted event")
        return readback

    def read(self, event_id: str) -> Optional[dict[str, Any]]:
        if not isinstance(event_id, str) or not event_id:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return None if row is None else self._row_to_dict(row)

    def read_by_idempotency_key(self, key: str) -> Optional[dict[str, Any]]:
        if not isinstance(key, str) or not key:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM events WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return None if row is None else self._row_to_dict(row)

    def list_run(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY sequence, event_id",
                (run_id,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_events(self, project_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Return detached read-only snapshots for deterministic projections."""
        with self._connect() as connection:
            if project_id is None:
                rows = connection.execute(
                    "SELECT * FROM events ORDER BY project_id, run_id, sequence, event_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM events
                    WHERE project_id = ?
                    ORDER BY project_id, run_id, sequence, event_id
                    """,
                    (project_id,),
                ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def _validate_sequence_and_transition(
        self, connection: sqlite3.Connection, event: EventEnvelope
    ) -> None:
        previous = None
        if event.previous_event_id is not None:
            previous_row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event.previous_event_id,)
            ).fetchone()
            if previous_row is None:
                raise LedgerContractError("previous_event_id does not exist")
            previous = self._row_to_dict(previous_row)
            if previous["run_id"] != event.run_id:
                raise LedgerContractError("previous event belongs to another run")
            if previous["project_id"] != event.project_id:
                raise LedgerContractError("previous event belongs to another project")
            allowed = VALID_TRANSITIONS.get(previous["event_type"], frozenset())
            if event.event_type not in allowed:
                raise LedgerContractError(
                    f"invalid transition: {previous['event_type']} -> {event.event_type}"
                )
            if event.sequence != previous["sequence"] + 1:
                raise LedgerContractError("sequence must increment by one from previous event")
        else:
            run_exists = connection.execute(
                "SELECT 1 FROM events WHERE run_id = ? LIMIT 1", (event.run_id,)
            ).fetchone()
            if run_exists is not None:
                raise LedgerContractError("previous_event_id is required after run creation")
            if event.event_type != EventType.CREATED or event.sequence != 1:
                raise LedgerContractError("a run must start with run.created sequence 1")

        if previous is not None and previous["terminal_state"] is not None:
            raise LedgerContractError("terminal event cannot be followed by another event")

    @staticmethod
    def _fingerprint(record: dict[str, Any]) -> dict[str, Any]:
        body = copy.deepcopy(record)
        body.pop("event_id", None)
        body.pop("recorded_at", None)
        return body

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "event_type": row["event_type"],
            "schema_version": row["schema_version"],
            "project_id": row["project_id"],
            "linear_issue_id": row["linear_issue_id"],
            "run_id": row["run_id"],
            "agent_id": row["agent_id"],
            "occurred_at": row["occurred_at"],
            "recorded_at": row["recorded_at"],
            "sequence": int(row["sequence"]),
            "idempotency_key": row["idempotency_key"],
            "evidence_refs": json.loads(row["evidence_refs_json"]),
            "blocked_reason": row["blocked_reason"],
            "terminal_state": row["terminal_state"],
            "payload": json.loads(row["payload_json"]),
            "previous_event_id": row["previous_event_id"],
        }
