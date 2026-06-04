"""SQLite-backed Hindsight session summary store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

SESSION_SUMMARY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SessionSummaryRecord:
    summary_key: str
    identity_scope: str
    summary_json: Any | None
    summary_text: str
    schema_version: int
    version: int
    turn: int
    turn_hash: str
    last_input_hash: str
    parent_summary_key: str | None
    status: str
    last_error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SessionSummaryWrite:
    summary_key: str
    identity_scope: str
    last_input_hash: str
    summary_json: Any | None = None
    summary_text: str = ""
    schema_version: int = SESSION_SUMMARY_SCHEMA_VERSION
    turn: int = 0
    turn_hash: str = ""
    parent_summary_key: str | None = None
    status: str = "ready"
    last_error: str | None = None
    expected_version: int | None = None
    now: str | datetime | None = None


@dataclass(frozen=True)
class SessionSummaryWriteResult:
    record: SessionSummaryRecord | None
    inserted: bool
    updated: bool
    stale: bool
    idempotent: bool


class SessionSummaryStore:
    def __init__(self, db_path: str | Path, busy_timeout_ms: int = 5000):
        self.db_path = Path(db_path)
        self.busy_timeout_ms = busy_timeout_ms
        self._conn = self._connect_with_recovery()
        self._configure()
        self._migrate()

    def get(self, summary_key: str) -> SessionSummaryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM session_summaries WHERE summary_key = ?",
            (summary_key,),
        ).fetchone()
        return _row_to_record(row) if row else None

    def upsert(self, write: SessionSummaryWrite) -> SessionSummaryWriteResult:
        _validate_write(write)
        existing = self.get(write.summary_key)
        if existing and existing.last_input_hash == write.last_input_hash:
            return SessionSummaryWriteResult(
                record=existing,
                inserted=False,
                updated=False,
                stale=False,
                idempotent=True,
            )
        if existing and write.expected_version is not None and write.expected_version != existing.version:
            return SessionSummaryWriteResult(
                record=existing,
                inserted=False,
                updated=False,
                stale=True,
                idempotent=False,
            )
        if not existing and write.expected_version is not None and write.expected_version > 0:
            return SessionSummaryWriteResult(
                record=None,
                inserted=False,
                updated=False,
                stale=True,
                idempotent=False,
            )

        now = _normalize_timestamp(write.now)
        params = _write_params(write, now)
        if not existing:
            self._conn.execute(
                """
                INSERT INTO session_summaries (
                    summary_key, identity_scope, summary_json, summary_text,
                    schema_version, version, turn, turn_hash, last_input_hash,
                    parent_summary_key, status, last_error, created_at, updated_at
                ) VALUES (
                    :summary_key, :identity_scope, :summary_json, :summary_text,
                    :schema_version, 1, :turn, :turn_hash, :last_input_hash,
                    :parent_summary_key, :status, :last_error, :created_at, :updated_at
                )
                """,
                params,
            )
            self._conn.commit()
            return SessionSummaryWriteResult(
                record=self.get(write.summary_key),
                inserted=True,
                updated=False,
                stale=False,
                idempotent=False,
            )

        if write.expected_version is None:
            cursor = self._conn.execute(
                """
                UPDATE session_summaries
                   SET identity_scope = :identity_scope,
                       summary_json = :summary_json,
                       summary_text = :summary_text,
                       schema_version = :schema_version,
                       version = version + 1,
                       turn = :turn,
                       turn_hash = :turn_hash,
                       last_input_hash = :last_input_hash,
                       parent_summary_key = :parent_summary_key,
                       status = :status,
                       last_error = :last_error,
                       updated_at = :updated_at
                 WHERE summary_key = :summary_key
                """,
                params,
            )
        else:
            cursor = self._conn.execute(
                """
                UPDATE session_summaries
                   SET identity_scope = :identity_scope,
                       summary_json = :summary_json,
                       summary_text = :summary_text,
                       schema_version = :schema_version,
                       version = version + 1,
                       turn = :turn,
                       turn_hash = :turn_hash,
                       last_input_hash = :last_input_hash,
                       parent_summary_key = :parent_summary_key,
                       status = :status,
                       last_error = :last_error,
                       updated_at = :updated_at
                 WHERE summary_key = :summary_key
                   AND version = :expected_version
                """,
                {**params, "expected_version": write.expected_version},
            )
        self._conn.commit()
        if cursor.rowcount == 0:
            current = self.get(write.summary_key)
            return SessionSummaryWriteResult(
                record=current,
                inserted=False,
                updated=False,
                stale=True,
                idempotent=False,
            )
        return SessionSummaryWriteResult(
            record=self.get(write.summary_key),
            inserted=False,
            updated=True,
            stale=False,
            idempotent=False,
        )

    def close(self) -> None:
        self._conn.close()

    def _connect_with_recovery(self) -> sqlite3.Connection:
        try:
            conn = self._connect()
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                conn.close()
                raise sqlite3.DatabaseError(f"SQLite integrity_check failed: {integrity}")
            return conn
        except sqlite3.DatabaseError:
            if str(self.db_path) == ":memory:" or not self.db_path.exists():
                raise
            corrupt_path = self._corrupt_path()
            for suffix in ("-wal", "-shm"):
                try:
                    Path(f"{self.db_path}{suffix}").unlink()
                except FileNotFoundError:
                    pass
            self.db_path.rename(corrupt_path)
            return self._connect()

    def _connect(self) -> sqlite3.Connection:
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=self.busy_timeout_ms / 1000)
        conn.row_factory = sqlite3.Row
        return conn

    def _configure(self) -> None:
        self._conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")

    def _migrate(self) -> None:
        self._conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS session_summaries (
                summary_key TEXT PRIMARY KEY,
                identity_scope TEXT NOT NULL,
                summary_json TEXT,
                summary_text TEXT NOT NULL DEFAULT '',
                schema_version INTEGER NOT NULL,
                version INTEGER NOT NULL,
                turn INTEGER NOT NULL,
                turn_hash TEXT NOT NULL,
                last_input_hash TEXT NOT NULL,
                parent_summary_key TEXT,
                status TEXT NOT NULL,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_session_summaries_identity_scope
                ON session_summaries(identity_scope);
            CREATE INDEX IF NOT EXISTS idx_session_summaries_parent
                ON session_summaries(parent_summary_key);
            CREATE INDEX IF NOT EXISTS idx_session_summaries_status
                ON session_summaries(status);
            PRAGMA user_version = {SESSION_SUMMARY_SCHEMA_VERSION};
            """
        )
        self._conn.commit()

    def _corrupt_path(self) -> Path:
        stamp = datetime.now(timezone.utc).isoformat().replace(":", "-").replace(".", "-")
        return self.db_path.with_name(f"{self.db_path.name}.corrupt.{stamp}")


def _validate_write(write: SessionSummaryWrite) -> None:
    if not write.summary_key:
        raise ValueError("summary_key is required")
    if not write.identity_scope:
        raise ValueError("identity_scope is required")
    if not write.last_input_hash:
        raise ValueError("last_input_hash is required")


def _normalize_timestamp(value: str | datetime | None) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if value:
        return value
    return datetime.now(timezone.utc).isoformat()


def _write_params(write: SessionSummaryWrite, now: str) -> dict[str, Any]:
    return {
        "summary_key": write.summary_key,
        "identity_scope": write.identity_scope,
        "summary_json": json.dumps(write.summary_json) if write.summary_json is not None else None,
        "summary_text": write.summary_text,
        "schema_version": write.schema_version,
        "turn": write.turn,
        "turn_hash": write.turn_hash,
        "last_input_hash": write.last_input_hash,
        "parent_summary_key": write.parent_summary_key,
        "status": write.status,
        "last_error": write.last_error,
        "created_at": now,
        "updated_at": now,
    }


def _row_to_record(row: sqlite3.Row) -> SessionSummaryRecord:
    raw_json = row["summary_json"]
    return SessionSummaryRecord(
        summary_key=row["summary_key"],
        identity_scope=row["identity_scope"],
        summary_json=json.loads(raw_json) if raw_json else None,
        summary_text=row["summary_text"],
        schema_version=row["schema_version"],
        version=row["version"],
        turn=row["turn"],
        turn_hash=row["turn_hash"],
        last_input_hash=row["last_input_hash"],
        parent_summary_key=row["parent_summary_key"],
        status=row["status"],
        last_error=row["last_error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
