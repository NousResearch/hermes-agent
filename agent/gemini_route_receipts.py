"""Profile-local receipts for Gemini delegation routing and daily reviews."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import stat
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import quote
from zoneinfo import ZoneInfo

from hermes_constants import get_hermes_home
from hermes_state import apply_wal_with_fallback


UTC = timezone.utc
DEFAULT_TIMEZONE = "America/Los_Angeles"
_TERMINAL_ATTEMPT_STATUSES = frozenset(
    {"completed", "failed", "timeout", "cancelled", "malformed", "denied", "oversized"}
)
_TERMINAL_BATCH_STATUSES = frozenset({"passed", "failed", "pipeline_failed"})


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime | None) -> datetime:
    value = value or _utc_now()
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime | None = None) -> str:
    return _as_utc(value).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def routing_day_for(when: datetime, timezone_name: str = DEFAULT_TIMEZONE) -> str:
    """Return the immutable local calendar day for an aware UTC instant."""
    return _as_utc(when).astimezone(ZoneInfo(timezone_name)).date().isoformat()


class GeminiReceiptStore:
    """Small SQLite repository isolated from Hermes conversation state."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else get_hermes_home() / "routing" / "gemini-routing.sqlite3"
        self._initialize()

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._enforce_private_mode(self.path.parent, 0o700)
        with self._write_connection() as conn:
            self._ensure_schema(conn)
        self._enforce_private_mode(self.path, 0o600)

    @staticmethod
    def _enforce_private_mode(path: Path, expected_mode: int) -> None:
        try:
            path.chmod(expected_mode)
            actual_mode = stat.S_IMODE(path.stat().st_mode)
        except OSError as exc:
            raise RuntimeError(f"could not enforce private permissions for {path}") from exc
        if actual_mode != expected_mode:
            raise RuntimeError(
                f"could not enforce private permissions for {path}: "
                f"expected {oct(expected_mode)}, got {oct(actual_mode)}"
            )

    def _open_write(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            actual = apply_wal_with_fallback(conn, db_label="routing/gemini-routing.sqlite3")
            if actual not in {"wal", "delete"}:
                raise sqlite3.OperationalError(f"unsupported journal mode returned: {actual}")
            for suffix in ("", "-wal", "-shm"):
                candidate = Path(f"{self.path}{suffix}")
                if candidate.exists():
                    self._enforce_private_mode(candidate, 0o600)
        except Exception:
            conn.close()
            raise
        return conn

    def _open_readonly(self) -> sqlite3.Connection:
        uri = f"file:{quote(str(self.path.resolve()), safe='/')}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _write_connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._open_write()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._open_readonly()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS gemini_attempts (
                receipt_id TEXT PRIMARY KEY,
                parent_session_id TEXT NOT NULL,
                parent_turn_id TEXT NOT NULL DEFAULT '',
                child_session_id TEXT NOT NULL,
                task_index INTEGER NOT NULL,
                routing_day TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                process_started_at_utc TEXT,
                completed_at_utc TEXT,
                route_requested TEXT NOT NULL,
                route_decision TEXT NOT NULL,
                route_reason TEXT NOT NULL,
                data_classification TEXT NOT NULL,
                output_contract TEXT NOT NULL,
                goal_text TEXT NOT NULL,
                context_text TEXT NOT NULL DEFAULT '',
                goal_sha256 TEXT NOT NULL,
                context_sha256 TEXT NOT NULL,
                prompt_sha256 TEXT NOT NULL,
                response_text TEXT,
                response_sha256 TEXT,
                worker_status TEXT NOT NULL,
                process_exit_code INTEGER,
                duration_ms INTEGER NOT NULL,
                requested_provider TEXT NOT NULL,
                requested_model TEXT NOT NULL,
                requested_effort TEXT NOT NULL,
                conversation_id TEXT,
                usage_json TEXT,
                raw_envelope_json TEXT,
                fallback_used INTEGER NOT NULL DEFAULT 0,
                error_code TEXT,
                error_message TEXT,
                created_at_utc TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_gemini_attempts_day
            ON gemini_attempts(routing_day, route_decision, receipt_id);

            CREATE TABLE IF NOT EXISTS daily_review_batches (
                batch_id TEXT PRIMARY KEY,
                routing_day TEXT NOT NULL UNIQUE,
                timezone TEXT NOT NULL,
                sample_size_requested INTEGER NOT NULL,
                eligible_count INTEGER NOT NULL,
                sample_seed_hex TEXT NOT NULL,
                sample_receipt_ids_json TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                completed_at_utc TEXT,
                pipeline_error TEXT,
                alert_status TEXT NOT NULL DEFAULT 'not_needed',
                alert_message_sha256 TEXT,
                slack_channel_id TEXT,
                slack_message_ts TEXT,
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS daily_review_items (
                batch_id TEXT NOT NULL,
                receipt_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                reviewer_provider TEXT NOT NULL,
                reviewer_model TEXT NOT NULL,
                review_status TEXT NOT NULL,
                verdict TEXT,
                reason TEXT,
                review_json TEXT,
                review_sha256 TEXT,
                started_at_utc TEXT NOT NULL,
                completed_at_utc TEXT,
                error_code TEXT,
                error_message TEXT,
                PRIMARY KEY (batch_id, receipt_id),
                FOREIGN KEY (batch_id) REFERENCES daily_review_batches(batch_id),
                FOREIGN KEY (receipt_id) REFERENCES gemini_attempts(receipt_id)
            );
            """
        )

    def prepare_attempt(
        self,
        *,
        parent_session_id: str,
        child_session_id: str,
        task_index: int,
        route_requested: str,
        route_decision: str,
        route_reason: str,
        data_classification: str,
        output_contract: str,
        goal_text: str,
        requested_provider: str,
        requested_model: str,
        requested_effort: str,
        parent_turn_id: str = "",
        context_text: str = "",
        started_at: datetime | None = None,
        receipt_id: str | None = None,
        timezone_name: str = DEFAULT_TIMEZONE,
    ) -> str:
        started = _as_utc(started_at)
        rid = receipt_id or f"grt_{secrets.token_hex(16)}"
        prompt = _canonical_json(
            {"goal": goal_text, "context": context_text, "output_contract": output_contract}
        )
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO gemini_attempts (
                    receipt_id, parent_session_id, parent_turn_id, child_session_id,
                    task_index, routing_day, started_at_utc, route_requested,
                    route_decision, route_reason, data_classification, output_contract,
                    goal_text, context_text, goal_sha256, context_sha256, prompt_sha256,
                    worker_status, duration_ms, requested_provider, requested_model,
                    requested_effort, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          'prepared', 0, ?, ?, ?, ?)
                """,
                (
                    rid,
                    parent_session_id,
                    parent_turn_id,
                    child_session_id,
                    int(task_index),
                    routing_day_for(started, timezone_name),
                    started.isoformat(),
                    route_requested,
                    route_decision,
                    route_reason,
                    data_classification,
                    output_contract,
                    goal_text,
                    context_text,
                    _sha256(goal_text),
                    _sha256(context_text),
                    _sha256(prompt),
                    requested_provider,
                    requested_model,
                    requested_effort,
                    _iso(),
                ),
            )
        return rid

    def mark_process_started(self, receipt_id: str, *, when: datetime | None = None) -> None:
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE gemini_attempts
                   SET process_started_at_utc=?, worker_status='running'
                   WHERE receipt_id=? AND worker_status='prepared'
                """,
                (_iso(when), receipt_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("attempt is missing or process was already started")

    def complete_attempt(
        self,
        receipt_id: str,
        *,
        worker_status: str,
        duration_ms: int,
        response_text: str | None = None,
        process_exit_code: int | None = None,
        conversation_id: str | None = None,
        usage: Mapping[str, Any] | None = None,
        raw_envelope: Mapping[str, Any] | None = None,
        fallback_used: bool = False,
        error_code: str | None = None,
        error_message: str | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        if worker_status not in _TERMINAL_ATTEMPT_STATUSES:
            raise ValueError(f"worker_status is not terminal: {worker_status}")
        with self._transaction() as conn:
            current = conn.execute(
                "SELECT worker_status FROM gemini_attempts WHERE receipt_id=?", (receipt_id,)
            ).fetchone()
            if current is None:
                raise KeyError(receipt_id)
            if current["worker_status"] in _TERMINAL_ATTEMPT_STATUSES:
                raise ValueError("attempt is already terminal")
            cursor = conn.execute(
                """
                UPDATE gemini_attempts SET
                    completed_at_utc=?, response_text=?, response_sha256=?,
                    worker_status=?, process_exit_code=?, duration_ms=?, conversation_id=?,
                    usage_json=?, raw_envelope_json=?, fallback_used=?, error_code=?, error_message=?
                WHERE receipt_id=?
                """,
                (
                    _iso(completed_at),
                    response_text,
                    _sha256(response_text) if response_text is not None else None,
                    worker_status,
                    process_exit_code,
                    max(0, int(duration_ms)),
                    conversation_id,
                    _canonical_json(dict(usage)) if usage is not None else None,
                    _canonical_json(dict(raw_envelope)) if raw_envelope is not None else None,
                    int(bool(fallback_used)),
                    error_code,
                    error_message,
                    receipt_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(receipt_id)

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def get_attempt(self, receipt_id: str) -> dict[str, Any]:
        with self._read_connection() as conn:
            row = conn.execute(
                "SELECT * FROM gemini_attempts WHERE receipt_id=?", (receipt_id,)
            ).fetchone()
        if row is None:
            raise KeyError(receipt_id)
        return dict(row)

    def count_attempts(self) -> int:
        with self._read_connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM gemini_attempts").fetchone()[0])

    def list_started_attempts_for_day(self, routing_day: str | date) -> list[dict[str, Any]]:
        day = routing_day.isoformat() if isinstance(routing_day, date) else str(routing_day)
        with self._read_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM gemini_attempts
                   WHERE routing_day=? AND route_decision='gemini'
                     AND process_started_at_utc IS NOT NULL
                   ORDER BY receipt_id
                """,
                (day,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_or_get_review_batch(
        self,
        *,
        routing_day: str,
        timezone_name: str,
        sample_size_requested: int,
        eligible_count: int,
        sample_seed_hex: str,
        sample_receipt_ids: Sequence[str],
        started_at: datetime | None = None,
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        bid = batch_id or f"grb_{secrets.token_hex(16)}"
        now = _iso(started_at)
        sample_json = _canonical_json(list(sample_receipt_ids))
        with self._transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO daily_review_batches (
                       batch_id, routing_day, timezone, sample_size_requested,
                       eligible_count, sample_seed_hex, sample_receipt_ids_json,
                       status, started_at_utc, created_at_utc
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 'preparing', ?, ?)
                """,
                (
                    bid,
                    routing_day,
                    timezone_name,
                    int(sample_size_requested),
                    int(eligible_count),
                    sample_seed_hex,
                    sample_json,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM daily_review_batches WHERE routing_day=?", (routing_day,)
            ).fetchone()
        if row is None:  # pragma: no cover - transaction invariant
            raise RuntimeError("review batch insert vanished")
        return dict(row)

    def get_review_batch(self, routing_day: str) -> dict[str, Any] | None:
        with self._read_connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_review_batches WHERE routing_day=?", (routing_day,)
            ).fetchone()
        return self._row(row)

    def claim_review_batch(self, batch_id: str) -> bool:
        """Atomically grant one runner authority to execute a prepared batch."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE daily_review_batches
                   SET status='reviewing'
                   WHERE batch_id=? AND status='preparing'
                """,
                (batch_id,),
            )
        return cursor.rowcount == 1

    def fail_stale_review_batch(
        self,
        batch_id: str,
        *,
        stale_before: datetime,
        pipeline_error: str,
        alert_message: str,
        completed_at: datetime,
    ) -> bool:
        """Atomically terminalize an abandoned review lease exactly once."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE daily_review_batches
                   SET status='pipeline_failed', completed_at_utc=?, pipeline_error=?,
                       alert_status='pending', alert_message_sha256=?
                   WHERE batch_id=?
                     AND status IN ('preparing', 'reviewing')
                     AND started_at_utc <= ?
                """,
                (
                    _iso(completed_at),
                    pipeline_error,
                    _sha256(alert_message),
                    batch_id,
                    _iso(stale_before),
                ),
            )
        return cursor.rowcount == 1

    def count_review_batches(self) -> int:
        with self._read_connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM daily_review_batches").fetchone()[0])

    def update_review_batch(
        self,
        batch_id: str,
        *,
        status: str,
        pipeline_error: str | None = None,
        alert_status: str | None = None,
        alert_message: str | None = None,
        slack_channel_id: str | None = None,
        slack_message_ts: str | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        if status not in {"preparing", "reviewing", *_TERMINAL_BATCH_STATUSES}:
            raise ValueError(f"invalid batch status: {status}")
        terminal_at = _iso(completed_at) if status in _TERMINAL_BATCH_STATUSES else None
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE daily_review_batches SET
                       status=?, completed_at_utc=?, pipeline_error=?,
                       alert_status=COALESCE(?, alert_status),
                       alert_message_sha256=COALESCE(?, alert_message_sha256),
                       slack_channel_id=COALESCE(?, slack_channel_id),
                       slack_message_ts=COALESCE(?, slack_message_ts)
                   WHERE batch_id=?
                """,
                (
                    status,
                    terminal_at,
                    pipeline_error,
                    alert_status,
                    _sha256(alert_message) if alert_message is not None else None,
                    slack_channel_id,
                    slack_message_ts,
                    batch_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(batch_id)

    def add_review_item(
        self,
        *,
        batch_id: str,
        receipt_id: str,
        ordinal: int,
        reviewer_provider: str,
        reviewer_model: str,
        review_status: str,
        verdict: str | None = None,
        reason: str | None = None,
        review_json: Mapping[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        raw = _canonical_json(dict(review_json)) if review_json is not None else None
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO daily_review_items (
                       batch_id, receipt_id, ordinal, reviewer_provider, reviewer_model,
                       review_status, verdict, reason, review_json, review_sha256,
                       started_at_utc, completed_at_utc, error_code, error_message
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    receipt_id,
                    int(ordinal),
                    reviewer_provider,
                    reviewer_model,
                    review_status,
                    verdict,
                    reason,
                    raw,
                    _sha256(raw) if raw is not None else None,
                    _iso(started_at),
                    _iso(completed_at) if completed_at is not None else None,
                    error_code,
                    error_message,
                ),
            )

    def list_review_items(self, batch_id: str) -> list[dict[str, Any]]:
        with self._read_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_review_items WHERE batch_id=? ORDER BY ordinal, receipt_id",
                (batch_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def apply_retention(
        self,
        *,
        now: datetime | None = None,
        raw_days: int = 30,
        aggregate_days: int = 180,
    ) -> dict[str, int]:
        current = _as_utc(now)
        raw_cutoff = (current - timedelta(days=max(0, int(raw_days)))).isoformat()
        aggregate_cutoff = (current - timedelta(days=max(0, int(aggregate_days)))).isoformat()
        with self._transaction() as conn:
            raw_cursor = conn.execute(
                """UPDATE gemini_attempts SET
                       goal_text='', context_text='', response_text=NULL,
                       usage_json=NULL, raw_envelope_json=NULL, error_message=NULL
                   WHERE started_at_utc < ?
                     AND (goal_text != '' OR context_text != '' OR response_text IS NOT NULL
                          OR usage_json IS NOT NULL OR raw_envelope_json IS NOT NULL
                          OR error_message IS NOT NULL)
                """,
                (raw_cutoff,),
            )
            review_raw_cursor = conn.execute(
                """UPDATE daily_review_items
                   SET reason='', review_json=NULL
                   WHERE batch_id IN (
                       SELECT batch_id FROM daily_review_batches
                       WHERE created_at_utc < ?
                   )
                     AND (reason != '' OR review_json IS NOT NULL)
                """,
                (raw_cutoff,),
            )
            old_batches = [
                row[0]
                for row in conn.execute(
                    "SELECT batch_id FROM daily_review_batches WHERE created_at_utc < ?",
                    (aggregate_cutoff,),
                ).fetchall()
            ]
            deleted_items = 0
            deleted_batches = 0
            if old_batches:
                marks = ",".join("?" for _ in old_batches)
                deleted_items = conn.execute(
                    f"DELETE FROM daily_review_items WHERE batch_id IN ({marks})", old_batches
                ).rowcount
                deleted_batches = conn.execute(
                    f"DELETE FROM daily_review_batches WHERE batch_id IN ({marks})", old_batches
                ).rowcount
            deleted_attempts = conn.execute(
                """DELETE FROM gemini_attempts
                   WHERE created_at_utc < ?
                     AND receipt_id NOT IN (SELECT receipt_id FROM daily_review_items)
                """,
                (aggregate_cutoff,),
            ).rowcount
        return {
            "raw_redacted": raw_cursor.rowcount,
            "review_raw_redacted": review_raw_cursor.rowcount,
            "review_items_deleted": deleted_items,
            "review_batches_deleted": deleted_batches,
            "attempts_deleted": deleted_attempts,
        }
