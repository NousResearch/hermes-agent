"""Dedicated SQLite durability boundary for Work Router.

The store is independent from Hermes session state and Kanban.  It contains no
network client and never logs event payloads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .models import (
    CanonicalEvent,
    MeetingCandidate,
    MeetingRoundState,
    MeetingState,
    RouteDecision,
    RouterThreadState,
    WorkExecution,
)


_MEETING_TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "stopped",
        "blocked_materials",
        "blocked_preflight_error",
        "blocked_all_error",
        "blocked_report_failed",
        "blocked_publication",
        "blocked_recovery",
    }
)


class StoreError(RuntimeError):
    """Base class for durable Router store failures."""


class LeaseBusy(StoreError):
    """Another live worker owns the thread lease."""


class CASConflict(StoreError):
    """A stale state version attempted to overwrite a thread."""


class LeaseFenceConflict(StoreError):
    """A worker tried to mutate state after losing its lease generation."""


@dataclass(frozen=True)
class LeaseClaim:
    event_id: str
    channel_id: str
    thread_ts: str
    lease_owner: str
    generation: int
    attempt_count: int
    status: str = "claimed"

    def __bool__(self) -> bool:
        return self.status == "claimed"


@dataclass(frozen=True)
class MeetingStopTransition:
    meeting_id: str
    meeting_status: str
    meeting_generation: int
    operation_id: str
    response_kind: str
    content: str
    lease_owner: str
    lease_generation: int
    already_stopped: bool = False


@dataclass(frozen=True)
class Stage3aCandidateTransition:
    meeting: MeetingState
    report_operation_id: str | None = None


class RouterStore:
    """SQLite store with transaction-scoped leases, deduplication, and outbox."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5000,
        thread_ttl_seconds: float = 24 * 60 * 60,
        lease_seconds: float = 30.0,
        completion_ttl_seconds: float = 7 * 24 * 60 * 60,
        max_processing_attempts: int = 3,
    ) -> None:
        self.path = str(path)
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.thread_ttl_seconds = float(thread_ttl_seconds)
        self.lease_seconds = float(lease_seconds)
        self.completion_ttl_seconds = float(completion_ttl_seconds)
        self.max_processing_attempts = int(max_processing_attempts)
        self._lock = threading.RLock()
        self._conn = self._open()
        self.initialize()
        from .ack_journal import AckJournal
        self.ack_journal = AckJournal(self)

    def _open(self) -> sqlite3.Connection:
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            timeout=max(self.busy_timeout_ms, 1) / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={max(self.busy_timeout_ms, 1)}")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def initialize(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS router_work_executions (
                source_event_id TEXT PRIMARY KEY,
                directive_id TEXT NOT NULL,
                slack_team_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                thread_ts TEXT NOT NULL,
                directive_ts TEXT NOT NULL,
                owner_profile TEXT NOT NULL,
                owner_slack_user_id TEXT NOT NULL,
                state TEXT NOT NULL,
                owner_task_id TEXT,
                pm_task_id TEXT,
                owner_run_id INTEGER,
                pm_run_id INTEGER,
                review_generation INTEGER NOT NULL DEFAULT 0,
                approval_required INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                lifecycle_attempt_count INTEGER NOT NULL DEFAULT 0,
                next_lifecycle_attempt_at REAL
            );

            CREATE TABLE IF NOT EXISTS router_inbox (
                    event_id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    thread_ts TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','processing','completed','quarantined')),
                    ack_completed_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    claimed_at REAL,
                    completed_at REAL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS router_threads (
                    channel_id TEXT NOT NULL,
                    thread_ts TEXT NOT NULL,
                    task_id TEXT,
                    state_version INTEGER NOT NULL,
                    lock_version INTEGER NOT NULL,
                    mode TEXT NOT NULL CHECK(mode IN ('work','meeting')),
                    owner TEXT,
                    active_team TEXT,
                    advisor_stack TEXT NOT NULL,
                    last_human_target TEXT,
                    last_event_id TEXT,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY(channel_id, thread_ts)
                );
                CREATE TABLE IF NOT EXISTS router_thread_leases (
                    channel_id TEXT NOT NULL,
                    thread_ts TEXT NOT NULL,
                    lease_owner TEXT NOT NULL,
                    lease_deadline REAL NOT NULL,
                    generation INTEGER NOT NULL,
                    PRIMARY KEY(channel_id, thread_ts)
                );
                CREATE TABLE IF NOT EXISTS router_completions (
                    event_id TEXT PRIMARY KEY,
                    outcome TEXT NOT NULL,
                    target_profile TEXT,
                    confirmed_message_id TEXT,
                    completed_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS router_outbox (
                    operation_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    thread_ts TEXT NOT NULL,
                    target_profile TEXT,
                    content TEXT,
                    response_kind TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('prepared','attempting','confirmed','ambiguous','quarantined','failed')),
                    attempt_evidence TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(event_id)
                );
                CREATE TABLE IF NOT EXISTS router_handoff_counts (
                    channel_id TEXT NOT NULL,
                    thread_ts TEXT NOT NULL,
                    handoff_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(channel_id, thread_ts)
                );
                CREATE INDEX IF NOT EXISTS router_inbox_thread_status
                    ON router_inbox(channel_id, thread_ts, status, created_at);
                CREATE INDEX IF NOT EXISTS router_completions_expiry
                    ON router_completions(expires_at);
                """
            )
            # Keep the migration local and idempotent for stores created by
            # the first Router vertical slice.
            columns = {
                str(row[1])
                for row in self._conn.execute("PRAGMA table_info(router_inbox)")
            }
            if "attempt_count" not in columns:
                self._conn.execute(
                    "ALTER TABLE router_inbox ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0"
                )
            self._initialize_meeting_schema()

    def upsert_work_execution(self, execution: WorkExecution) -> None:
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO router_work_executions (
                    source_event_id,
                    directive_id,
                    slack_team_id,
                    channel_id,
                    thread_ts,
                    directive_ts,
                    owner_profile,
                    owner_slack_user_id,
                    state,
                    owner_task_id,
                    pm_task_id,
                    owner_run_id,
                    pm_run_id,
                    review_generation,
                    approval_required,
                    last_error,
                    lifecycle_attempt_count,
                    next_lifecycle_attempt_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_event_id) DO UPDATE SET
                    directive_id=excluded.directive_id,
                    slack_team_id=excluded.slack_team_id,
                    channel_id=excluded.channel_id,
                    thread_ts=excluded.thread_ts,
                    directive_ts=excluded.directive_ts,
                    owner_profile=excluded.owner_profile,
                    owner_slack_user_id=excluded.owner_slack_user_id,
                    state=excluded.state,
                    owner_task_id=excluded.owner_task_id,
                    pm_task_id=excluded.pm_task_id,
                    owner_run_id=excluded.owner_run_id,
                    pm_run_id=excluded.pm_run_id,
                    review_generation=excluded.review_generation,
                    approval_required=excluded.approval_required,
                    last_error=excluded.last_error,
                    lifecycle_attempt_count=excluded.lifecycle_attempt_count,
                    next_lifecycle_attempt_at=excluded.next_lifecycle_attempt_at
                """,
                (
                    execution.source_event_id,
                    execution.directive_id,
                    execution.slack_team_id,
                    execution.channel_id,
                    execution.thread_ts,
                    execution.directive_ts,
                    execution.owner_profile,
                    execution.owner_slack_user_id,
                    execution.state,
                    execution.owner_task_id,
                    execution.pm_task_id,
                    execution.owner_run_id,
                    execution.pm_run_id,
                    execution.review_generation,
                    int(execution.approval_required),
                    execution.last_error,
                    execution.lifecycle_attempt_count,
                    execution.next_lifecycle_attempt_at,
                ),
            )

    def get_work_execution(
        self,
        source_event_id: str,
    ) -> WorkExecution | None:
        row = self._conn.execute(
            """
            SELECT
                directive_id,
                source_event_id,
                slack_team_id,
                channel_id,
                thread_ts,
                directive_ts,
                owner_profile,
                owner_slack_user_id,
                state,
                owner_task_id,
                pm_task_id,
                owner_run_id,
                pm_run_id,
                review_generation,
                approval_required,
                last_error,
                lifecycle_attempt_count,
                next_lifecycle_attempt_at
            FROM router_work_executions
            WHERE source_event_id=?
            """,
            (source_event_id,),
        ).fetchone()

        if row is None:
            return None

        return WorkExecution(
            directive_id=str(row[0]),
            source_event_id=str(row[1]),
            slack_team_id=str(row[2]),
            channel_id=str(row[3]),
            thread_ts=str(row[4]),
            directive_ts=str(row[5]),
            owner_profile=str(row[6]),
            owner_slack_user_id=str(row[7]),
            state=str(row[8]),
            owner_task_id=row[9],
            pm_task_id=row[10],
            owner_run_id=row[11],
            pm_run_id=row[12],
            review_generation=int(row[13]),
            approval_required=bool(row[14]),
            last_error=row[15],
            lifecycle_attempt_count=int(row[16]),
            next_lifecycle_attempt_at=row[17],
        )


    def get_active_work_execution_for_thread(
        self,
        channel_id: str,
        thread_ts: str,
    ) -> WorkExecution | None:
        rows = self._conn.execute(
            """
            SELECT source_event_id
            FROM router_work_executions
            WHERE channel_id=?
              AND thread_ts=?
              AND state != ?
            """,
            (
                channel_id,
                thread_ts,
                "FINAL_RESULT",
            ),
        ).fetchall()

        if not rows:
            return None

        if len(rows) != 1:
            raise RuntimeError(
                "multiple active work executions for thread: "
                f"{channel_id}/{thread_ts}"
            )

        return self.get_work_execution(str(rows[0][0]))

    def transition_work_execution(
        self,
        source_event_id: str,
        **changes: object,
    ) -> WorkExecution:
        current = self.get_work_execution(source_event_id)
        if current is None:
            raise KeyError(
                f"work execution not found: {source_event_id}"
            )

        allowed = {
            "state",
            "owner_task_id",
            "pm_task_id",
            "owner_run_id",
            "pm_run_id",
            "review_generation",
            "approval_required",
            "last_error",
            "lifecycle_attempt_count",
            "next_lifecycle_attempt_at",
        }

        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(
                f"unsupported work execution changes: {sorted(unknown)}"
            )

        updated = replace(current, **changes)
        self.upsert_work_execution(updated)
        return updated


    def _initialize_meeting_schema(self) -> None:
        """Atomically relax the thread mode constraint and add meeting tables."""

        expected_columns = (
            "channel_id",
            "thread_ts",
            "task_id",
            "state_version",
            "lock_version",
            "mode",
            "owner",
            "active_team",
            "advisor_stack",
            "last_human_target",
            "last_event_id",
            "updated_at",
            "expires_at",
        )
        with self._transaction():
            table_row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='router_threads'"
            ).fetchone()
            if table_row is None or not table_row[0]:
                raise StoreError("router_threads schema is missing")
            columns = tuple(
                str(row[1]) for row in self._conn.execute("PRAGMA table_info(router_threads)")
            )
            if columns != expected_columns:
                raise StoreError("router_threads columns do not match the migration contract")
            compact_sql = "".join(str(table_row[0]).lower().split())
            legacy_constraint = "check(mode='work')"
            meeting_constraint = "check(modein('work','meeting'))"
            if legacy_constraint in compact_sql:
                self._migrate_router_threads_mode_constraint(expected_columns)
            elif meeting_constraint not in compact_sql:
                raise StoreError("router_threads mode constraint is not recognized")

            meeting_ddls = (
                """
                CREATE TABLE IF NOT EXISTS router_meetings (
                    meeting_id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    thread_ts TEXT NOT NULL,
                    status TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    current_round INTEGER NOT NULL,
                    participants_json TEXT NOT NULL,
                    timeout_round_streak INTEGER NOT NULL DEFAULT 0,
                    all_pass_streak INTEGER NOT NULL DEFAULT 0,
                    return_state_json TEXT,
                    preflight_result_json TEXT,
                    block_reason TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS router_meeting_rounds (
                    meeting_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    generation INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    packet_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(meeting_id, round_id)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS router_meeting_candidates (
                    meeting_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    generation INTEGER NOT NULL,
                    participant TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason_to_speak INTEGER,
                    reason_class TEXT NOT NULL DEFAULT 'none',
                    statement TEXT NOT NULL DEFAULT '',
                    error_class TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(meeting_id, round_id, participant)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS router_meeting_publications (
                    operation_id TEXT PRIMARY KEY,
                    meeting_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    publication_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(meeting_id, round_id, publication_index)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS router_meeting_participant_injections (
                    meeting_id TEXT NOT NULL,
                    source_event_id TEXT NOT NULL,
                    participant TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','applied')),
                    created_at REAL NOT NULL,
                    applied_at REAL,
                    PRIMARY KEY(meeting_id, source_event_id, participant)
                )
                """,
            )
            for ddl in meeting_ddls:
                self._conn.execute(ddl)
            self._migrate_router_meetings_preflight_schema()
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS router_meetings_active_thread
                    ON router_meetings(channel_id, thread_ts, status, updated_at)
                """
            )
            self._conn.execute("PRAGMA user_version=2")
            if self._conn.execute("PRAGMA foreign_key_check").fetchall():
                raise StoreError("meeting schema migration failed foreign key validation")
            integrity = self._conn.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or str(integrity[0]) != "ok":
                raise StoreError("meeting schema migration failed integrity validation")

    def _migrate_router_meetings_preflight_schema(self) -> None:
        """Add preflight evidence/snapshot fields and allow terminal meeting history."""

        old_columns = (
            "meeting_id",
            "channel_id",
            "thread_ts",
            "status",
            "generation",
            "current_round",
            "participants_json",
            "timeout_round_streak",
            "all_pass_streak",
            "created_at",
            "updated_at",
        )
        new_columns = (
            *old_columns[:-2],
            "return_state_json",
            "preflight_result_json",
            "block_reason",
            *old_columns[-2:],
        )
        table_row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='router_meetings'"
        ).fetchone()
        if table_row is None or not table_row[0]:
            raise StoreError("router_meetings schema is missing")
        columns = tuple(
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(router_meetings)")
        )
        compact_sql = "".join(str(table_row[0]).lower().split())
        has_thread_unique = "unique(channel_id,thread_ts)" in compact_sql
        if columns == new_columns and not has_thread_unique:
            return
        if columns not in {old_columns, new_columns}:
            raise StoreError("router_meetings columns do not match the preflight migration contract")
        staging = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='router_meetings__preflight_v2'"
        ).fetchone()
        if staging is not None:
            raise StoreError("router_meetings preflight migration staging table already exists")
        self._conn.execute(
            """
            CREATE TABLE router_meetings__preflight_v2 (
                meeting_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                thread_ts TEXT NOT NULL,
                status TEXT NOT NULL,
                generation INTEGER NOT NULL,
                current_round INTEGER NOT NULL,
                participants_json TEXT NOT NULL,
                timeout_round_streak INTEGER NOT NULL DEFAULT 0,
                all_pass_streak INTEGER NOT NULL DEFAULT 0,
                return_state_json TEXT,
                preflight_result_json TEXT,
                block_reason TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        source_columns = ", ".join(columns)
        if columns == old_columns:
            destination_columns = ", ".join(new_columns)
            select_columns = ", ".join(
                [*old_columns[:-2], "NULL", "NULL", "NULL", *old_columns[-2:]]
            )
        else:
            destination_columns = source_columns
            select_columns = source_columns
        self._conn.execute(
            f"INSERT INTO router_meetings__preflight_v2 ({destination_columns}) "
            f"SELECT {select_columns} FROM router_meetings"
        )
        old_count = int(self._conn.execute("SELECT COUNT(*) FROM router_meetings").fetchone()[0])
        new_count = int(
            self._conn.execute("SELECT COUNT(*) FROM router_meetings__preflight_v2").fetchone()[0]
        )
        if old_count != new_count:
            raise StoreError("router_meetings preflight migration row count mismatch")
        self._conn.execute("DROP TABLE router_meetings")
        self._conn.execute(
            "ALTER TABLE router_meetings__preflight_v2 RENAME TO router_meetings"
        )

    def _migrate_router_threads_mode_constraint(self, columns: tuple[str, ...]) -> None:
        invalid_mode = self._conn.execute(
            "SELECT mode FROM router_threads WHERE mode <> 'work' LIMIT 1"
        ).fetchone()
        if invalid_mode is not None:
            raise StoreError("legacy router_threads contains a non-work mode")
        staging = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='router_threads__meeting_v2'"
        ).fetchone()
        if staging is not None:
            raise StoreError("router_threads meeting migration staging table already exists")
        custom_schema = self._conn.execute(
            """
            SELECT name FROM sqlite_master
             WHERE tbl_name='router_threads' AND type IN ('index','trigger') AND sql IS NOT NULL
            """
        ).fetchall()
        if custom_schema:
            raise StoreError("router_threads has unrecognized indexes or triggers")
        self._conn.execute(
            """
            CREATE TABLE router_threads__meeting_v2 (
                channel_id TEXT NOT NULL,
                thread_ts TEXT NOT NULL,
                task_id TEXT,
                state_version INTEGER NOT NULL,
                lock_version INTEGER NOT NULL,
                mode TEXT NOT NULL CHECK(mode IN ('work','meeting')),
                owner TEXT,
                active_team TEXT,
                advisor_stack TEXT NOT NULL,
                last_human_target TEXT,
                last_event_id TEXT,
                updated_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                PRIMARY KEY(channel_id, thread_ts)
            )
            """
        )
        column_sql = ", ".join(columns)
        self._conn.execute(
            f"INSERT INTO router_threads__meeting_v2 ({column_sql}) "
            f"SELECT {column_sql} FROM router_threads"
        )
        old_count = int(self._conn.execute("SELECT COUNT(*) FROM router_threads").fetchone()[0])
        new_count = int(
            self._conn.execute("SELECT COUNT(*) FROM router_threads__meeting_v2").fetchone()[0]
        )
        if old_count != new_count:
            raise StoreError("router_threads migration row count mismatch")
        difference_count = int(
            self._conn.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT {column_sql} FROM router_threads
                    EXCEPT
                    SELECT {column_sql} FROM router_threads__meeting_v2
                    UNION ALL
                    SELECT {column_sql} FROM router_threads__meeting_v2
                    EXCEPT
                    SELECT {column_sql} FROM router_threads
                )
                """
            ).fetchone()[0]
        )
        if difference_count:
            raise StoreError("router_threads migration content mismatch")
        self._conn.execute("DROP TABLE router_threads")
        self._conn.execute("ALTER TABLE router_threads__meeting_v2 RENAME TO router_threads")

    def integrity_check(self) -> str:
        with self._lock:
            row = self._conn.execute("PRAGMA integrity_check").fetchone()
            return str(row[0]) if row else ""

    def create_meeting(self, state: MeetingState) -> None:
        """Persist the immutable identity and initial state for one meeting."""

        try:
            with self._transaction():
                self._conn.execute(
                    """
                    INSERT INTO router_meetings
                        (meeting_id, channel_id, thread_ts, status, generation, current_round,
                         participants_json, timeout_round_streak, all_pass_streak,
                         return_state_json, preflight_result_json, block_reason,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        state.meeting_id,
                        state.channel_id,
                        state.thread_ts,
                        state.status,
                        state.generation,
                        state.current_round,
                        json.dumps(list(state.participants), ensure_ascii=False),
                        state.timeout_round_streak,
                        state.all_pass_streak,
                        (
                            json.dumps(state.return_state.to_dict(), ensure_ascii=False, sort_keys=True)
                            if state.return_state is not None
                            else None
                        ),
                        state.preflight_result_json,
                        state.block_reason,
                        state.created_at,
                        state.updated_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise StoreError("meeting identity already exists") from exc

    def get_meeting(self, meeting_id: str) -> MeetingState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (meeting_id,),
            ).fetchone()
        return _row_to_meeting(row) if row is not None else None

    def get_meeting_for_thread(self, channel_id: str, thread_ts: str) -> MeetingState | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM router_meetings
                 WHERE channel_id=? AND thread_ts=?
                   AND status NOT IN (
                       'completed', 'stopped', 'blocked_materials',
                       'blocked_preflight_error', 'blocked_all_error',
                       'blocked_report_failed', 'blocked_publication', 'blocked_recovery'
                   )
                 ORDER BY updated_at DESC, meeting_id DESC
                 LIMIT 1
                """,
                (channel_id, thread_ts),
            ).fetchone()
        return _row_to_meeting(row) if row is not None else None

    def update_meeting_status(
        self,
        meeting_id: str,
        status: str,
        *,
        expected_generation: int,
        now: float | None = None,
    ) -> MeetingState:
        if status == "stopped":
            raise StoreError("stopped transition requires the atomic meeting stop API")
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            current = _row_to_meeting(row)
            if current.generation != expected_generation:
                raise StoreError("meeting generation is stale")
            updated = replace(current, status=status, updated_at=timestamp)
            cursor = self._conn.execute(
                """
                UPDATE router_meetings SET status=?, updated_at=?
                 WHERE meeting_id=? AND generation=?
                """,
                (updated.status, updated.updated_at, meeting_id, expected_generation),
            )
            if cursor.rowcount != 1:
                raise StoreError("meeting generation is stale")
        return updated

    def has_meeting_stop_outbox(self, event_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1 FROM router_outbox
                 WHERE event_id=?
                   AND response_kind IN (
                       'meeting_stop_confirmation', 'meeting_stop_blocked_recovery'
                   )
                """,
                (event_id,),
            ).fetchone()
        return row is not None

    def outbox_allows_send(self, operation_id: str) -> bool:
        """Fence an external send unless its durable operation is attempting."""

        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM router_outbox WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
        return row is not None and str(row[0]) == "attempting"

    def stop_meeting_and_restore_thread(
        self,
        event: CanonicalEvent,
        *,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now: float | None = None,
    ) -> MeetingStopTransition:
        """Atomically stop one meeting, fence stale work, and prepare confirmation."""

        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            inbox = self._conn.execute(
                """
                SELECT status, created_at FROM router_inbox
                 WHERE event_id=? AND channel_id=? AND thread_ts=?
                """,
                (event.event_id, event.channel_id, event.thread_ts),
            ).fetchone()
            if inbox is None or str(inbox[0]) not in {"pending", "processing"}:
                raise StoreError("meeting stop event is not recoverable")

            existing_outbox = self._conn.execute(
                """
                SELECT operation_id, response_kind, content
                  FROM router_outbox WHERE event_id=?
                """,
                (event.event_id,),
            ).fetchone()
            if existing_outbox is not None:
                return self._existing_stop_transition_in_transaction(
                    event,
                    existing_outbox,
                    lease_owner,
                    lease_generation,
                    timestamp,
                )

            meeting_row = self._conn.execute(
                """
                SELECT * FROM router_meetings
                 WHERE channel_id=? AND thread_ts=?
                   AND status NOT IN (
                       'completed', 'stopped', 'blocked_materials',
                       'blocked_preflight_error', 'blocked_all_error',
                       'blocked_report_failed', 'blocked_publication', 'blocked_recovery'
                   )
                 ORDER BY updated_at DESC, meeting_id DESC LIMIT 1
                """,
                (event.channel_id, event.thread_ts),
            ).fetchone()
            if meeting_row is None:
                raise StoreError("active meeting does not exist")
            meeting = _row_to_meeting(meeting_row)
            thread_row = self._conn.execute(
                "SELECT * FROM router_threads WHERE channel_id=? AND thread_ts=?",
                (event.channel_id, event.thread_ts),
            ).fetchone()
            if thread_row is None:
                raise StoreError("meeting thread does not exist")

            owner, generation = self._resolve_stop_lease_in_transaction(
                event,
                lease_owner,
                lease_generation,
                timestamp,
                claim_pending=True,
            )
            self._terminalize_meeting_inbox_in_transaction(
                event,
                through_created_at=float(inbox[1]),
                now=timestamp,
            )
            self._conn.execute(
                """
                UPDATE router_meeting_rounds SET status='discarded', updated_at=?
                 WHERE meeting_id=? AND generation=?
                   AND status NOT IN ('completed','discarded')
                """,
                (timestamp, meeting.meeting_id, meeting.generation),
            )
            self._conn.execute(
                """
                UPDATE router_meeting_candidates SET status='discarded', updated_at=?
                 WHERE meeting_id=? AND generation=? AND status!='discarded'
                """,
                (timestamp, meeting.meeting_id, meeting.generation),
            )

            if meeting.return_state is None:
                meeting_status = "blocked_recovery"
                response_kind = "meeting_stop_blocked_recovery"
                content = (
                    "회의 중단 복구에 실패했습니다. 저장된 Work Mode 복귀 상태가 없어 "
                    "자동 복구하지 않았습니다. 운영 확인이 필요합니다."
                )
                block_reason = "missing_return_state"
            else:
                self._restore_meeting_thread_in_transaction(
                    meeting,
                    thread_row,
                    event,
                    timestamp,
                )
                meeting_status = "stopped"
                response_kind = "meeting_stop_confirmation"
                content = "회의를 중단했습니다. 이 thread는 Work Mode로 복귀했습니다."
                block_reason = f"sinclair_stop:{event.event_id}"

            meeting_cursor = self._conn.execute(
                """
                UPDATE router_meetings
                   SET status=?, generation=generation+1, block_reason=?, updated_at=?
                 WHERE meeting_id=? AND status=? AND generation=?
                """,
                (
                    meeting_status,
                    block_reason,
                    timestamp,
                    meeting.meeting_id,
                    meeting.status,
                    meeting.generation,
                ),
            )
            if meeting_cursor.rowcount != 1:
                raise StoreError("meeting stop generation is stale")

            operation_id = f"router:{event.event_id}:{response_kind}"
            self._conn.execute(
                """
                INSERT INTO router_outbox
                    (operation_id, event_id, channel_id, thread_ts, target_profile,
                     content, response_kind, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'Demian', ?, ?, 'prepared', ?, ?)
                """,
                (
                    operation_id,
                    event.event_id,
                    event.channel_id,
                    event.thread_ts,
                    content,
                    response_kind,
                    timestamp,
                    timestamp,
                ),
            )
            return MeetingStopTransition(
                meeting_id=meeting.meeting_id,
                meeting_status=meeting_status,
                meeting_generation=meeting.generation + 1,
                operation_id=operation_id,
                response_kind=response_kind,
                content=content,
                lease_owner=owner,
                lease_generation=generation,
            )

    def _existing_stop_transition_in_transaction(
        self,
        event: CanonicalEvent,
        outbox: sqlite3.Row,
        lease_owner: str | None,
        lease_generation: int | None,
        timestamp: float,
    ) -> MeetingStopTransition:
        response_kind = str(outbox[1])
        if response_kind not in {
            "meeting_stop_confirmation",
            "meeting_stop_blocked_recovery",
        }:
            raise StoreError("meeting stop event has a conflicting outbox")
        meeting_row = self._conn.execute(
            """
            SELECT * FROM router_meetings
             WHERE channel_id=? AND thread_ts=?
               AND status IN ('stopped','blocked_recovery','blocked_report_failed')
             ORDER BY updated_at DESC, meeting_id DESC LIMIT 1
            """,
            (event.channel_id, event.thread_ts),
        ).fetchone()
        if meeting_row is None:
            raise StoreError("meeting stop outbox has no terminal meeting")
        meeting = _row_to_meeting(meeting_row)
        owner, generation = self._resolve_stop_lease_in_transaction(
            event,
            lease_owner,
            lease_generation,
            timestamp,
            claim_pending=False,
        )
        return MeetingStopTransition(
            meeting_id=meeting.meeting_id,
            meeting_status=meeting.status,
            meeting_generation=meeting.generation,
            operation_id=str(outbox[0]),
            response_kind=response_kind,
            content=str(outbox[2] or ""),
            lease_owner=owner,
            lease_generation=generation,
            already_stopped=True,
        )

    def _resolve_stop_lease_in_transaction(
        self,
        event: CanonicalEvent,
        lease_owner: str | None,
        lease_generation: int | None,
        timestamp: float,
        *,
        claim_pending: bool,
    ) -> tuple[str, int]:
        if lease_owner is not None and lease_generation is not None:
            self._assert_lease_in_transaction(
                event.channel_id,
                event.thread_ts,
                lease_owner,
                lease_generation,
                timestamp,
            )
            return lease_owner, lease_generation
        lease = self._conn.execute(
            """
            SELECT generation FROM router_thread_leases
             WHERE channel_id=? AND thread_ts=?
            """,
            (event.channel_id, event.thread_ts),
        ).fetchone()
        next_generation = int(lease[0]) + 1 if lease is not None else 1
        owner = f"meeting-stop:{event.event_id}"
        self._conn.execute(
            """
            INSERT INTO router_thread_leases
                (channel_id, thread_ts, lease_owner, lease_deadline, generation)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(channel_id, thread_ts) DO UPDATE SET
                lease_owner=excluded.lease_owner,
                lease_deadline=excluded.lease_deadline,
                generation=excluded.generation
            """,
            (
                event.channel_id,
                event.thread_ts,
                owner,
                timestamp + self.lease_seconds,
                next_generation,
            ),
        )
        if claim_pending:
            cursor = self._conn.execute(
                """
                UPDATE router_inbox
                   SET status='processing', claimed_at=?, attempt_count=attempt_count+1
                 WHERE event_id=? AND status='pending'
                """,
                (timestamp, event.event_id),
            )
            if cursor.rowcount != 1:
                raise StoreError("meeting stop event could not be claimed")
        return owner, next_generation

    def _terminalize_meeting_inbox_in_transaction(
        self,
        event: CanonicalEvent,
        *,
        through_created_at: float,
        now: float,
    ) -> None:
        rows = self._conn.execute(
            """
            SELECT event_id FROM router_inbox
             WHERE channel_id=? AND thread_ts=? AND event_id!=?
               AND status IN ('pending','processing') AND created_at<=?
            """,
            (event.channel_id, event.thread_ts, event.event_id, through_created_at),
        ).fetchall()
        reason = "discarded by Sinclair meeting stop"
        for row in rows:
            discarded_event_id = str(row[0])
            self._conn.execute(
                """
                UPDATE router_inbox
                   SET status='quarantined', last_error=?, completed_at=?
                 WHERE event_id=? AND status IN ('pending','processing')
                """,
                (reason, now, discarded_event_id),
            )
            self._conn.execute(
                """
                UPDATE router_outbox
                   SET status='quarantined', attempt_evidence=?, updated_at=?
                 WHERE event_id=? AND status!='confirmed'
                """,
                (reason, now, discarded_event_id),
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO router_completions
                    (event_id, outcome, target_profile, confirmed_message_id,
                     completed_at, expires_at)
                VALUES (?, 'meeting_stop_discarded_inflight', NULL, NULL, ?, ?)
                """,
                (discarded_event_id, now, now + self.completion_ttl_seconds),
            )

    def _restore_meeting_thread_in_transaction(
        self,
        meeting: MeetingState,
        thread_row: sqlite3.Row,
        event: CanonicalEvent,
        timestamp: float,
    ) -> None:
        return_state = meeting.return_state
        if return_state is None:
            raise StoreError("meeting return state is unavailable")
        if (
            return_state.channel_id != event.channel_id
            or return_state.thread_ts != event.thread_ts
            or return_state.mode != "work"
        ):
            raise StoreError("meeting return state is invalid")
        current_state = _row_to_state(thread_row)
        restored = replace(
            return_state,
            state_version=current_state.state_version + 1,
            lock_version=current_state.lock_version + 1,
            last_event_id=event.event_id,
            updated_at=timestamp,
            expires_at=timestamp + self.thread_ttl_seconds,
        )
        cursor = self._conn.execute(
            """
            UPDATE router_threads SET task_id=?, state_version=?, lock_version=?, mode=?,
                owner=?, active_team=?, advisor_stack=?, last_human_target=?, last_event_id=?,
                updated_at=?, expires_at=?
             WHERE channel_id=? AND thread_ts=?
               AND state_version=? AND lock_version=?
            """,
            (
                restored.task_id,
                restored.state_version,
                restored.lock_version,
                restored.mode,
                restored.owner,
                restored.active_team,
                json.dumps(list(restored.advisor_stack), ensure_ascii=False),
                restored.last_human_target,
                restored.last_event_id,
                restored.updated_at,
                restored.expires_at,
                event.channel_id,
                event.thread_ts,
                current_state.state_version,
                current_state.lock_version,
            ),
        )
        if cursor.rowcount != 1:
            raise CASConflict("meeting stop thread restore CAS failed")

    def finalize_meeting_stop_confirmation(
        self,
        operation_id: str,
        *,
        message_id: str | None,
        lease_owner: str,
        lease_generation: int,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                """
                SELECT event_id, channel_id, thread_ts, response_kind, status
                  FROM router_outbox WHERE operation_id=?
                """,
                (operation_id,),
            ).fetchone()
            if row is None or str(row[4]) not in {"prepared", "attempting"}:
                raise StoreError("meeting stop outbox cannot be confirmed")
            self._assert_lease_in_transaction(
                str(row[1]),
                str(row[2]),
                lease_owner,
                lease_generation,
                timestamp,
            )
            self._conn.execute(
                "UPDATE router_outbox SET status='confirmed', updated_at=? WHERE operation_id=?",
                (timestamp, operation_id),
            )
            self._conn.execute(
                "UPDATE router_inbox SET status='completed', completed_at=? WHERE event_id=?",
                (timestamp, row[0]),
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO router_completions
                    (event_id, outcome, target_profile, confirmed_message_id,
                     completed_at, expires_at)
                VALUES (?, ?, 'Demian', ?, ?, ?)
                """,
                (
                    row[0],
                    str(row[3]),
                    message_id,
                    timestamp,
                    timestamp + self.completion_ttl_seconds,
                ),
            )

    def quarantine_meeting_stop_confirmation(
        self,
        operation_id: str,
        *,
        reason: str,
        lease_owner: str,
        lease_generation: int,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        safe_reason = reason[:200] or "stop_confirmation_failed"
        with self._transaction():
            row = self._conn.execute(
                "SELECT event_id, channel_id, thread_ts FROM router_outbox WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise StoreError("unknown meeting stop outbox")
            self._assert_lease_in_transaction(
                str(row[1]),
                str(row[2]),
                lease_owner,
                lease_generation,
                timestamp,
            )
            self._conn.execute(
                """
                UPDATE router_outbox SET status='quarantined', attempt_evidence=?, updated_at=?
                 WHERE operation_id=? AND status!='confirmed'
                """,
                (safe_reason, timestamp, operation_id),
            )
            self._conn.execute(
                """
                UPDATE router_inbox SET status='quarantined', last_error=?, completed_at=?
                 WHERE event_id=?
                """,
                (safe_reason, timestamp, row[0]),
            )
            meeting = self._conn.execute(
                """
                SELECT meeting_id FROM router_meetings
                 WHERE channel_id=? AND thread_ts=?
                   AND status IN ('stopped','blocked_recovery')
                 ORDER BY updated_at DESC, meeting_id DESC LIMIT 1
                """,
                (row[1], row[2]),
            ).fetchone()
            if meeting is not None:
                self._conn.execute(
                    """
                    UPDATE router_meetings
                       SET status='blocked_report_failed', block_reason=?, updated_at=?
                     WHERE meeting_id=? AND status IN ('stopped','blocked_recovery')
                    """,
                    (f"stop_confirmation_failed:{safe_reason}", timestamp, meeting[0]),
                )

    def finalize_meeting_preflight(
        self,
        operation_id: str,
        *,
        meeting_id: str,
        status: str,
        result_json: str | None,
        block_reason: str | None,
        participants: tuple[str, ...],
        target_profile: str | None,
        message_id: str | None,
        ready_state: RouterThreadState,
        expected_state_version: int,
        expected_lock_version: int,
        expected_generation: int,
        lease_owner: str,
        lease_generation: int,
        now: float | None = None,
    ) -> MeetingState:
        """Confirm one preflight post and atomically leave the preflight state."""

        if status not in {
            "ready",
            "awaiting_sensitive_approval",
            "blocked_materials",
            "blocked_preflight_error",
        }:
            raise StoreError("preflight terminal status is invalid")
        if status in {"ready", "awaiting_sensitive_approval", "blocked_materials"} and not result_json:
            raise StoreError("validated preflight result is required")
        if status == "blocked_preflight_error" and not block_reason:
            raise StoreError("preflight error reason is required")
        if status == "ready" and (
            not 3 <= len(participants) <= 4
            or len(set(participants)) != len(participants)
        ):
            raise StoreError("ready preflight requires three or four unique participants")
        if status != "ready" and participants:
            raise StoreError("only ready preflight may freeze participants")
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            outbox = self._conn.execute(
                "SELECT event_id, channel_id, thread_ts, status FROM router_outbox WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if outbox is None or outbox[3] not in {"prepared", "attempting"}:
                raise StoreError("preflight outbox operation cannot be confirmed")
            self._assert_lease_in_transaction(
                str(outbox[1]),
                str(outbox[2]),
                lease_owner,
                lease_generation,
                timestamp,
            )
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            meeting = _row_to_meeting(row)
            if meeting.status != "preflight" or meeting.generation != expected_generation:
                raise StoreError("meeting preflight generation is stale")
            round_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM router_meeting_rounds WHERE meeting_id=?",
                    (meeting_id,),
                ).fetchone()[0]
            )
            if round_count:
                raise StoreError("meeting preflight cannot finalize after a round exists")
            if status in {"ready", "awaiting_sensitive_approval"}:
                next_state = ready_state
                next_generation = meeting.generation
            else:
                if meeting.return_state is None:
                    raise StoreError("meeting preflight return state is unavailable")
                next_state = meeting.return_state
                next_generation = meeting.generation + 1
            if next_state.channel_id != outbox[1] or next_state.thread_ts != outbox[2]:
                raise StoreError("meeting preflight thread identity mismatch")
            if status in {"ready", "awaiting_sensitive_approval"} and next_state.mode != "meeting":
                raise StoreError("active preflight result must keep Meeting Mode")
            if status not in {"ready", "awaiting_sensitive_approval"} and next_state.mode != "work":
                raise StoreError("blocked preflight must restore Work Mode")
            current = self._conn.execute(
                "SELECT state_version, lock_version FROM router_threads WHERE channel_id=? AND thread_ts=?",
                (outbox[1], outbox[2]),
            ).fetchone()
            if current is None:
                raise StoreError("thread disappeared before preflight finalize")
            updated_thread = replace(
                next_state,
                updated_at=timestamp,
                expires_at=timestamp + self.thread_ttl_seconds,
            )
            cursor = self._conn.execute(
                """
                UPDATE router_threads SET task_id=?, state_version=?, lock_version=?, mode=?,
                    owner=?, active_team=?, advisor_stack=?, last_human_target=?, last_event_id=?,
                    updated_at=?, expires_at=?
                WHERE channel_id=? AND thread_ts=? AND state_version=? AND lock_version=?
                """,
                (
                    updated_thread.task_id,
                    expected_state_version + 1,
                    expected_lock_version + 1,
                    updated_thread.mode,
                    updated_thread.owner,
                    updated_thread.active_team,
                    json.dumps(list(updated_thread.advisor_stack), ensure_ascii=False),
                    updated_thread.last_human_target,
                    updated_thread.last_event_id,
                    updated_thread.updated_at,
                    updated_thread.expires_at,
                    outbox[1],
                    outbox[2],
                    expected_state_version,
                    expected_lock_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict("router thread state CAS failed during preflight finalize")
            meeting_cursor = self._conn.execute(
                """
                UPDATE router_meetings
                   SET status=?, generation=?, participants_json=?,
                       preflight_result_json=?, block_reason=?, updated_at=?
                 WHERE meeting_id=? AND status='preflight' AND generation=?
                """,
                (
                    status,
                    next_generation,
                    json.dumps(list(participants), ensure_ascii=False),
                    result_json,
                    block_reason,
                    timestamp,
                    meeting_id,
                    expected_generation,
                ),
            )
            if meeting_cursor.rowcount != 1:
                raise StoreError("meeting preflight generation is stale")
            self._conn.execute(
                "UPDATE router_outbox SET status='confirmed', updated_at=? WHERE operation_id=?",
                (timestamp, operation_id),
            )
            self._conn.execute(
                "UPDATE router_inbox SET status='completed', completed_at=? WHERE event_id=?",
                (timestamp, outbox[0]),
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO router_completions
                    (event_id, outcome, target_profile, confirmed_message_id, completed_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    outbox[0],
                    f"preflight_{status}",
                    target_profile,
                    message_id,
                    timestamp,
                    timestamp + self.completion_ttl_seconds,
                ),
            )
            updated_meeting = replace(
                meeting,
                status=status,
                generation=next_generation,
                participants=participants,
                preflight_result_json=result_json,
                block_reason=block_reason,
                updated_at=timestamp,
            )
        return updated_meeting

    def finalize_sensitive_lookup_approval(
        self,
        operation_id: str,
        *,
        meeting_id: str,
        message_id: str | None,
        next_state: RouterThreadState,
        expected_state_version: int,
        expected_lock_version: int,
        expected_generation: int,
        lease_owner: str,
        lease_generation: int,
        now: float | None = None,
    ) -> MeetingState:
        """Atomically record Sinclair's exact sensitive-lookup approval."""

        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            outbox = self._conn.execute(
                "SELECT event_id, channel_id, thread_ts, status FROM router_outbox WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if outbox is None or outbox[3] not in {"prepared", "attempting"}:
                raise StoreError("sensitive approval outbox operation cannot be confirmed")
            self._assert_lease_in_transaction(
                str(outbox[1]),
                str(outbox[2]),
                lease_owner,
                lease_generation,
                timestamp,
            )
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            meeting = _row_to_meeting(row)
            if (
                meeting.status != "awaiting_sensitive_approval"
                or meeting.generation != expected_generation
                or meeting.participants
            ):
                raise StoreError("sensitive approval meeting generation is stale")
            round_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM router_meeting_rounds WHERE meeting_id=?",
                    (meeting_id,),
                ).fetchone()[0]
            )
            if round_count:
                raise StoreError("sensitive approval cannot occur after a round exists")
            if (
                next_state.channel_id != outbox[1]
                or next_state.thread_ts != outbox[2]
                or next_state.mode != "meeting"
            ):
                raise StoreError("sensitive approval thread state is invalid")
            thread_cursor = self._conn.execute(
                """
                UPDATE router_threads SET task_id=?, state_version=?, lock_version=?, mode=?,
                    owner=?, active_team=?, advisor_stack=?, last_human_target=?, last_event_id=?,
                    updated_at=?, expires_at=?
                WHERE channel_id=? AND thread_ts=? AND state_version=? AND lock_version=?
                """,
                (
                    next_state.task_id,
                    expected_state_version + 1,
                    expected_lock_version + 1,
                    next_state.mode,
                    next_state.owner,
                    next_state.active_team,
                    json.dumps(list(next_state.advisor_stack), ensure_ascii=False),
                    next_state.last_human_target,
                    next_state.last_event_id,
                    timestamp,
                    timestamp + self.thread_ttl_seconds,
                    outbox[1],
                    outbox[2],
                    expected_state_version,
                    expected_lock_version,
                ),
            )
            if thread_cursor.rowcount != 1:
                raise CASConflict("router thread state CAS failed during sensitive approval")
            meeting_cursor = self._conn.execute(
                """
                UPDATE router_meetings SET status='sensitive_lookup_approved', updated_at=?
                 WHERE meeting_id=? AND status='awaiting_sensitive_approval' AND generation=?
                """,
                (timestamp, meeting_id, expected_generation),
            )
            if meeting_cursor.rowcount != 1:
                raise StoreError("sensitive approval meeting generation is stale")
            self._conn.execute(
                "UPDATE router_outbox SET status='confirmed', updated_at=? WHERE operation_id=?",
                (timestamp, operation_id),
            )
            self._conn.execute(
                "UPDATE router_inbox SET status='completed', completed_at=? WHERE event_id=?",
                (timestamp, outbox[0]),
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO router_completions
                    (event_id, outcome, target_profile, confirmed_message_id,
                     completed_at, expires_at)
                VALUES (?, 'sensitive_lookup_approved', 'Demian', ?, ?, ?)
                """,
                (
                    outbox[0],
                    message_id,
                    timestamp,
                    timestamp + self.completion_ttl_seconds,
                ),
            )
            updated = replace(
                meeting,
                status="sensitive_lookup_approved",
                updated_at=timestamp,
            )
        return updated

    def quarantine_meeting_preflight(
        self,
        operation_id: str,
        *,
        meeting_id: str,
        expected_generation: int,
        reason: str,
        lease_owner: str,
        lease_generation: int,
        now: float | None = None,
    ) -> MeetingState:
        """Atomically quarantine a failed model/send and terminally block preflight."""

        timestamp = time.time() if now is None else float(now)
        safe_reason = reason[:200] or "preflight_error"
        with self._transaction():
            outbox = self._conn.execute(
                "SELECT event_id, channel_id, thread_ts FROM router_outbox WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if outbox is None:
                raise StoreError("unknown preflight outbox operation")
            self._assert_lease_in_transaction(
                str(outbox[1]),
                str(outbox[2]),
                lease_owner,
                lease_generation,
                timestamp,
            )
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            meeting = _row_to_meeting(row)
            if meeting.status != "preflight" or meeting.generation != expected_generation:
                raise StoreError("meeting preflight generation is stale")
            if meeting.return_state is None:
                raise StoreError("meeting preflight return state is unavailable")
            round_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM router_meeting_rounds WHERE meeting_id=?",
                    (meeting_id,),
                ).fetchone()[0]
            )
            if round_count:
                raise StoreError("failed preflight cannot contain a meeting round")
            self._conn.execute(
                """
                UPDATE router_meetings
                   SET status='blocked_preflight_error', generation=generation+1,
                       preflight_result_json=NULL, block_reason=?, updated_at=?
                 WHERE meeting_id=? AND status='preflight' AND generation=?
                """,
                (safe_reason, timestamp, meeting_id, expected_generation),
            )
            self._conn.execute(
                "UPDATE router_outbox SET status='ambiguous', attempt_evidence=?, updated_at=? WHERE operation_id=?",
                (safe_reason, timestamp, operation_id),
            )
            self._conn.execute(
                "UPDATE router_outbox SET status='quarantined', updated_at=? WHERE operation_id=?",
                (timestamp, operation_id),
            )
            self._conn.execute(
                "UPDATE router_inbox SET status='quarantined', last_error=?, completed_at=? WHERE event_id=?",
                (safe_reason, timestamp, outbox[0]),
            )
            updated = replace(
                meeting,
                status="blocked_preflight_error",
                generation=meeting.generation + 1,
                preflight_result_json=None,
                block_reason=safe_reason,
                updated_at=timestamp,
            )
        return updated

    def create_meeting_round(self, state: MeetingRoundState) -> None:
        """Create one round only when its generation matches the parent meeting."""

        with self._transaction():
            row = self._conn.execute(
                "SELECT generation FROM router_meetings WHERE meeting_id=?",
                (state.meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            if int(row[0]) != state.generation:
                raise StoreError("meeting round generation is stale")
            try:
                self._conn.execute(
                    """
                    INSERT INTO router_meeting_rounds
                        (meeting_id, round_id, generation, status, packet_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        state.meeting_id,
                        state.round_id,
                        state.generation,
                        state.status,
                        state.packet_json,
                        state.created_at,
                        state.updated_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StoreError("meeting round already exists") from exc

    def get_meeting_round(self, meeting_id: str, round_id: int) -> MeetingRoundState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM router_meeting_rounds WHERE meeting_id=? AND round_id=?",
                (meeting_id, round_id),
            ).fetchone()
        return _row_to_meeting_round(row) if row is not None else None

    def start_dynamic_meeting_round(
        self,
        meeting_id: str,
        *,
        round_id: int,
        participants: tuple[str, ...],
        packet_json: str,
        expected_generation: int,
        now: float | None = None,
    ) -> MeetingState:
        """Atomically open one dynamic round and its exact participant subset."""

        if not 1 <= round_id <= 5:
            raise StoreError("dynamic meeting round_id must be between one and five")
        if not participants or len(set(participants)) != len(participants):
            raise StoreError("dynamic meeting round participants must be unique and non-empty")
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            meeting = _row_to_meeting(row)
            expected_status = "ready" if round_id == 1 else "awaiting_continue"
            if (
                meeting.status != expected_status
                or meeting.generation != expected_generation
                or meeting.current_round != round_id - 1
            ):
                raise StoreError("dynamic meeting round generation or sequence is stale")
            if any(profile not in meeting.participants for profile in participants):
                raise StoreError("dynamic meeting participant is outside the frozen meeting")
            cursor = self._conn.execute(
                """
                UPDATE router_meetings
                   SET status='round_open', current_round=?, updated_at=?
                 WHERE meeting_id=? AND status=? AND generation=? AND current_round=?
                """,
                (
                    round_id,
                    timestamp,
                    meeting_id,
                    expected_status,
                    expected_generation,
                    round_id - 1,
                ),
            )
            if cursor.rowcount != 1:
                raise StoreError("dynamic meeting round open failed")
            try:
                self._conn.execute(
                    """
                    INSERT INTO router_meeting_rounds
                        (meeting_id, round_id, generation, status, packet_json,
                         created_at, updated_at)
                    VALUES (?, ?, ?, 'collecting', ?, ?, ?)
                    """,
                    (
                        meeting_id,
                        round_id,
                        expected_generation,
                        packet_json,
                        timestamp,
                        timestamp,
                    ),
                )
                self._conn.executemany(
                    """
                    INSERT INTO router_meeting_candidates
                        (meeting_id, round_id, generation, participant, status,
                         reason_to_speak, reason_class, statement, error_class,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'pending', NULL, 'none', '', NULL, ?, ?)
                    """,
                    (
                        (
                            meeting_id,
                            round_id,
                            expected_generation,
                            participant,
                            timestamp,
                            timestamp,
                        )
                        for participant in participants
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StoreError("dynamic meeting round already exists") from exc
        return replace(
            meeting,
            status="round_open",
            current_round=round_id,
            updated_at=timestamp,
        )

    def finalize_dynamic_meeting_candidate(
        self,
        candidate: MeetingCandidate,
        *,
        now: float | None = None,
    ) -> None:
        """Finalize one pending candidate without closing its dynamic round."""

        if candidate.status not in {"submitted", "candidate_error", "timeout"}:
            raise StoreError("dynamic meeting candidate terminal state is invalid")
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (candidate.meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            meeting = _row_to_meeting(row)
            if (
                meeting.status != "round_open"
                or meeting.current_round != candidate.round_id
                or meeting.generation != candidate.generation
            ):
                raise StoreError("dynamic meeting candidate generation is stale")
            cursor = self._conn.execute(
                """
                UPDATE router_meeting_candidates
                   SET status=?, reason_to_speak=?, reason_class=?, statement=?,
                       error_class=?, updated_at=?
                 WHERE meeting_id=? AND round_id=? AND generation=?
                   AND participant=? AND status='pending'
                """,
                (
                    candidate.status,
                    None if candidate.reason_to_speak is None else int(candidate.reason_to_speak),
                    candidate.reason_class,
                    candidate.statement,
                    candidate.error_class,
                    timestamp,
                    candidate.meeting_id,
                    candidate.round_id,
                    candidate.generation,
                    candidate.participant,
                ),
            )
            if cursor.rowcount != 1:
                raise StoreError("dynamic meeting pending candidate was not found")

    def close_dynamic_meeting_round(
        self,
        meeting_id: str,
        *,
        round_id: int,
        expected_generation: int,
        now: float | None = None,
    ) -> MeetingState:
        """Close one fully submitted round and wait for Demian's next decision."""

        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            meeting = _row_to_meeting(row)
            if (
                meeting.status != "round_open"
                or meeting.current_round != round_id
                or meeting.generation != expected_generation
            ):
                raise StoreError("dynamic meeting round generation is stale")
            statuses = self._conn.execute(
                """
                SELECT status FROM router_meeting_candidates
                 WHERE meeting_id=? AND round_id=? AND generation=?
                """,
                (meeting_id, round_id, expected_generation),
            ).fetchall()
            if not statuses or any(str(item[0]) != "submitted" for item in statuses):
                raise StoreError("dynamic meeting round is not fully submitted")
            cursor = self._conn.execute(
                """
                UPDATE router_meeting_rounds SET status='closed', updated_at=?
                 WHERE meeting_id=? AND round_id=? AND generation=? AND status='collecting'
                """,
                (timestamp, meeting_id, round_id, expected_generation),
            )
            if cursor.rowcount != 1:
                raise StoreError("dynamic meeting round is not collecting")
            self._conn.execute(
                """
                UPDATE router_meetings SET status='awaiting_continue', updated_at=?
                 WHERE meeting_id=? AND status='round_open' AND generation=?
                """,
                (timestamp, meeting_id, expected_generation),
            )
        return replace(meeting, status="awaiting_continue", updated_at=timestamp)

    def enqueue_meeting_participant_injection_after_ack(
        self,
        event: CanonicalEvent,
        *,
        meeting_id: str,
        participants: tuple[str, ...],
        ack_completed: bool,
        now: float | None = None,
    ) -> bool:
        """Atomically persist one ACKed Sinclair injection as non-posting work."""

        if not ack_completed:
            raise StoreError("meeting participant injection requires completed Slack ACK")
        if not event.event_id or not event.channel_id or not event.thread_ts:
            raise StoreError("meeting participant injection identity is required")
        if not participants or len(set(participants)) != len(participants):
            raise StoreError("meeting participant injection is invalid")
        timestamp = time.time() if now is None else float(now)
        payload = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._transaction():
            meeting = self._conn.execute(
                "SELECT channel_id, thread_ts, status FROM router_meetings WHERE meeting_id=?",
                (meeting_id,),
            ).fetchone()
            if meeting is None:
                raise StoreError("meeting does not exist")
            if str(meeting[0]) != event.channel_id or str(meeting[1]) != event.thread_ts:
                raise StoreError("meeting participant injection thread does not match")
            if str(meeting[2]) in _MEETING_TERMINAL_STATUSES:
                raise StoreError("meeting participant injection is terminal")
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO router_inbox
                    (event_id, channel_id, thread_ts, payload_json, status,
                     ack_completed_at, created_at, completed_at)
                VALUES (?, ?, ?, ?, 'completed', ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.channel_id,
                    event.thread_ts,
                    payload,
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            )
            if cursor.rowcount != 1:
                return False
            self._conn.executemany(
                """
                INSERT INTO router_meeting_participant_injections
                    (meeting_id, source_event_id, participant, status, created_at, applied_at)
                VALUES (?, ?, ?, 'pending', ?, NULL)
                """,
                (
                    (meeting_id, event.event_id, participant, timestamp)
                    for participant in participants
                ),
            )
            self._conn.execute(
                """
                INSERT INTO router_completions
                    (event_id, outcome, target_profile, confirmed_message_id,
                     completed_at, expires_at)
                VALUES (?, 'meeting_participant_injection', NULL, NULL, ?, ?)
                """,
                (
                    event.event_id,
                    timestamp,
                    timestamp + self.completion_ttl_seconds,
                ),
            )
        return True

    def get_meeting_participant_injections(self, meeting_id: str) -> tuple[str, ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT participant, MIN(created_at) AS first_created
                  FROM router_meeting_participant_injections
                 WHERE meeting_id=?
                 GROUP BY participant
                 ORDER BY first_created, participant
                """,
                (meeting_id,),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def get_pending_meeting_participant_injections(
        self,
        meeting_id: str,
    ) -> tuple[str, ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT participant, MIN(created_at) AS first_created
                  FROM router_meeting_participant_injections
                 WHERE meeting_id=? AND status='pending'
                 GROUP BY participant
                 ORDER BY first_created, participant
                """,
                (meeting_id,),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def apply_meeting_participant_injections(
        self,
        meeting_id: str,
        *,
        participants: tuple[str, ...],
        expected_generation: int,
        now: float | None = None,
    ) -> MeetingState:
        """Apply validated pending injections only at the between-round boundary."""

        if not participants or len(set(participants)) != len(participants):
            raise StoreError("meeting participant injection application is invalid")
        timestamp = time.time() if now is None else float(now)
        placeholders = ",".join("?" for _ in participants)
        with self._transaction():
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            meeting = _row_to_meeting(row)
            if meeting.status != "awaiting_continue" or meeting.generation != expected_generation:
                raise StoreError("meeting participant injection application is stale")
            pending_rows = self._conn.execute(
                f"""
                SELECT DISTINCT participant
                  FROM router_meeting_participant_injections
                 WHERE meeting_id=? AND status='pending'
                   AND participant IN ({placeholders})
                """,
                (meeting_id, *participants),
            ).fetchall()
            pending = {str(item[0]) for item in pending_rows}
            if pending != set(participants):
                raise StoreError("meeting participant injection is not pending")
            merged = tuple(dict.fromkeys((*meeting.participants, *participants)))
            cursor = self._conn.execute(
                """
                UPDATE router_meetings SET participants_json=?, updated_at=?
                 WHERE meeting_id=? AND status='awaiting_continue' AND generation=?
                """,
                (
                    json.dumps(list(merged), ensure_ascii=False),
                    timestamp,
                    meeting_id,
                    expected_generation,
                ),
            )
            if cursor.rowcount != 1:
                raise StoreError("meeting participant injection application is stale")
            self._conn.execute(
                f"""
                UPDATE router_meeting_participant_injections
                   SET status='applied', applied_at=?
                 WHERE meeting_id=? AND status='pending'
                   AND participant IN ({placeholders})
                """,
                (timestamp, meeting_id, *participants),
            )
        return replace(meeting, participants=merged, updated_at=timestamp)

    def extend_dynamic_meeting_participants(
        self,
        meeting_id: str,
        *,
        participants: tuple[str, ...],
        expected_generation: int,
        now: float | None = None,
    ) -> MeetingState:
        """Append forced Staff to the durable meeting snapshot between rounds."""

        if not participants or len(set(participants)) != len(participants):
            raise StoreError("dynamic meeting participant extension is invalid")
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            meeting = _row_to_meeting(row)
            if (
                meeting.status != "awaiting_continue"
                or meeting.generation != expected_generation
            ):
                raise StoreError("dynamic meeting participant extension is stale")
            merged = tuple(dict.fromkeys((*meeting.participants, *participants)))
            self._conn.execute(
                """
                UPDATE router_meetings SET participants_json=?, updated_at=?
                 WHERE meeting_id=? AND status='awaiting_continue' AND generation=?
                """,
                (
                    json.dumps(list(merged), ensure_ascii=False),
                    timestamp,
                    meeting_id,
                    expected_generation,
                ),
            )
        return replace(meeting, participants=merged, updated_at=timestamp)

    def complete_dynamic_meeting(
        self,
        meeting_id: str,
        *,
        expected_generation: int,
        now: float | None = None,
    ) -> MeetingState:
        """Mark a dynamically converged or safety-stopped meeting complete."""

        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            meeting = _row_to_meeting(row)
            if (
                meeting.status != "awaiting_continue"
                or meeting.generation != expected_generation
            ):
                raise StoreError("dynamic meeting completion is stale")
            self._conn.execute(
                """
                UPDATE router_meetings SET status='completed', updated_at=?
                 WHERE meeting_id=? AND status='awaiting_continue' AND generation=?
                """,
                (timestamp, meeting_id, expected_generation),
            )
        return replace(meeting, status="completed", updated_at=timestamp)

    def start_stage_3a_candidate(
        self,
        meeting_id: str,
        *,
        participant: str,
        packet_json: str,
        expected_generation: int,
        now: float | None = None,
    ) -> MeetingState:
        """Atomically open round one and one pending serial candidate."""

        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            meeting = _row_to_meeting(row)
            if meeting.status != "ready" or meeting.generation != expected_generation:
                raise StoreError("stage 3a meeting generation is stale")
            if not meeting.participants or participant != meeting.participants[0]:
                raise StoreError("stage 3a candidate must be the first frozen participant")
            cursor = self._conn.execute(
                """
                UPDATE router_meetings
                   SET status='round_open', current_round=1, updated_at=?
                 WHERE meeting_id=? AND status='ready' AND generation=? AND current_round=0
                """,
                (timestamp, meeting_id, expected_generation),
            )
            if cursor.rowcount != 1:
                raise StoreError("stage 3a meeting round open failed")
            self._conn.execute(
                """
                INSERT INTO router_meeting_rounds
                    (meeting_id, round_id, generation, status, packet_json, created_at, updated_at)
                VALUES (?, 1, ?, 'collecting', ?, ?, ?)
                """,
                (meeting_id, expected_generation, packet_json, timestamp, timestamp),
            )
            self._conn.execute(
                """
                INSERT INTO router_meeting_candidates
                    (meeting_id, round_id, generation, participant, status,
                     reason_to_speak, reason_class, statement, error_class,
                     created_at, updated_at)
                VALUES (?, 1, ?, ?, 'pending', NULL, 'none', '', NULL, ?, ?)
                """,
                (meeting_id, expected_generation, participant, timestamp, timestamp),
            )
        return replace(meeting, status="round_open", current_round=1, updated_at=timestamp)

    def finalize_stage_3a_candidate(
        self,
        candidate: MeetingCandidate,
        *,
        event: CanonicalEvent | None = None,
        report_content: str | None = None,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now: float | None = None,
    ) -> Stage3aCandidateTransition:
        """Persist the sole stage 3a result and stop before selection/publication."""

        if candidate.round_id != 1 or candidate.status not in {"submitted", "candidate_error"}:
            raise StoreError("stage 3a candidate terminal state is invalid")
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                "SELECT * FROM router_meetings WHERE meeting_id=?",
                (candidate.meeting_id,),
            ).fetchone()
            if row is None:
                raise StoreError("meeting does not exist")
            meeting = _row_to_meeting(row)
            if (
                meeting.status != "round_open"
                or meeting.current_round != 1
                or meeting.generation != candidate.generation
            ):
                raise StoreError("stage 3a candidate meeting generation is stale")
            if candidate.status == "candidate_error":
                if (
                    event is None
                    or report_content is None
                    or lease_owner is None
                    or lease_generation is None
                ):
                    raise StoreError("stage 3a terminal error requires an atomic restore report")
                if (
                    event.channel_id != meeting.channel_id
                    or event.thread_ts != meeting.thread_ts
                ):
                    raise StoreError("stage 3a terminal error event does not match meeting")
                self._assert_lease_in_transaction(
                    meeting.channel_id,
                    meeting.thread_ts,
                    lease_owner,
                    lease_generation,
                    timestamp,
                )
            round_row = self._conn.execute(
                """
                SELECT status FROM router_meeting_rounds
                 WHERE meeting_id=? AND round_id=1 AND generation=?
                """,
                (candidate.meeting_id, candidate.generation),
            ).fetchone()
            if round_row is None or str(round_row[0]) != "collecting":
                raise StoreError("stage 3a candidate round is not collecting")
            cursor = self._conn.execute(
                """
                UPDATE router_meeting_candidates
                   SET status=?, reason_to_speak=?, reason_class=?, statement=?,
                       error_class=?, updated_at=?
                 WHERE meeting_id=? AND round_id=1 AND generation=?
                   AND participant=? AND status='pending'
                """,
                (
                    candidate.status,
                    None if candidate.reason_to_speak is None else int(candidate.reason_to_speak),
                    candidate.reason_class,
                    candidate.statement,
                    candidate.error_class,
                    timestamp,
                    candidate.meeting_id,
                    candidate.generation,
                    candidate.participant,
                ),
            )
            if cursor.rowcount != 1:
                raise StoreError("stage 3a pending candidate was not found")
            next_status = "selecting" if candidate.status == "submitted" else "blocked_all_error"
            round_status = "closed" if candidate.status == "submitted" else "blocked_all_error"
            self._conn.execute(
                """
                UPDATE router_meeting_rounds SET status=?, updated_at=?
                 WHERE meeting_id=? AND round_id=1 AND generation=? AND status='collecting'
                """,
                (
                    round_status,
                    timestamp,
                    candidate.meeting_id,
                    candidate.generation,
                ),
            )
            self._conn.execute(
                """
                UPDATE router_meetings SET status=?, updated_at=?
                 WHERE meeting_id=? AND status='round_open' AND generation=?
                """,
                (
                    next_status,
                    timestamp,
                    candidate.meeting_id,
                    candidate.generation,
                ),
            )
            report_operation_id = None
            if candidate.status == "candidate_error":
                if event is None or report_content is None:
                    raise StoreError("stage 3a terminal error restore inputs disappeared")
                thread_row = self._conn.execute(
                    "SELECT * FROM router_threads WHERE channel_id=? AND thread_ts=?",
                    (meeting.channel_id, meeting.thread_ts),
                ).fetchone()
                if thread_row is None:
                    raise StoreError("stage 3a terminal error thread is unavailable")
                self._restore_meeting_thread_in_transaction(
                    meeting,
                    thread_row,
                    event,
                    timestamp,
                )
                report_event_id = (
                    f"meeting-report:{meeting.meeting_id}:{meeting.generation}:blocked_all_error"
                )
                report_operation_id = f"router:{report_event_id}:meeting_blocked_all_error"
                self._conn.execute(
                    """
                    INSERT INTO router_outbox
                        (operation_id, event_id, channel_id, thread_ts, target_profile,
                         content, response_kind, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'Demian', ?, 'meeting_blocked_all_error',
                            'prepared', ?, ?)
                    """,
                    (
                        report_operation_id,
                        report_event_id,
                        meeting.channel_id,
                        meeting.thread_ts,
                        report_content,
                        timestamp,
                        timestamp,
                    ),
                )
        updated = replace(meeting, status=next_status, updated_at=timestamp)
        return Stage3aCandidateTransition(updated, report_operation_id)

    def record_meeting_candidate(self, candidate: MeetingCandidate) -> None:
        """Upsert a candidate behind the persisted round generation fence."""

        with self._transaction():
            row = self._conn.execute(
                """
                SELECT generation, status FROM router_meeting_rounds
                 WHERE meeting_id=? AND round_id=?
                """,
                (candidate.meeting_id, candidate.round_id),
            ).fetchone()
            if row is None:
                raise StoreError("meeting round does not exist")
            if int(row[0]) != candidate.generation:
                raise StoreError("meeting candidate generation is stale")
            if str(row[1]) != "collecting":
                raise StoreError("meeting round is not collecting candidates")
            self._conn.execute(
                """
                INSERT INTO router_meeting_candidates
                    (meeting_id, round_id, generation, participant, status, reason_to_speak,
                     reason_class, statement, error_class, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(meeting_id, round_id, participant) DO UPDATE SET
                    generation=excluded.generation,
                    status=excluded.status,
                    reason_to_speak=excluded.reason_to_speak,
                    reason_class=excluded.reason_class,
                    statement=excluded.statement,
                    error_class=excluded.error_class,
                    updated_at=excluded.updated_at
                """,
                (
                    candidate.meeting_id,
                    candidate.round_id,
                    candidate.generation,
                    candidate.participant,
                    candidate.status,
                    None if candidate.reason_to_speak is None else int(candidate.reason_to_speak),
                    candidate.reason_class,
                    candidate.statement,
                    candidate.error_class,
                    candidate.created_at,
                    candidate.updated_at,
                ),
            )

    def get_meeting_candidate(
        self,
        meeting_id: str,
        round_id: int,
        participant: str,
    ) -> MeetingCandidate | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM router_meeting_candidates
                 WHERE meeting_id=? AND round_id=? AND participant=?
                """,
                (meeting_id, round_id, participant),
            ).fetchone()
        return _row_to_meeting_candidate(row) if row is not None else None

    def enqueue_after_ack(self, event: CanonicalEvent, *, ack_completed: bool, now: float | None = None) -> bool:
        """Insert an event only after confirmed ACK completion.

        ``INSERT OR IGNORE`` makes the durable event_id primary key the sole
        correctness deduplication mechanism.  No process-memory dedup is used.
        """
        if not ack_completed:
            raise StoreError("durable enqueue requires completed Slack ACK")
        if not event.event_id or not event.channel_id or not event.thread_ts:
            raise StoreError("canonical event is missing durable identity fields")
        timestamp = time.time() if now is None else float(now)
        payload = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._transaction():
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO router_inbox
                    (event_id, channel_id, thread_ts, payload_json, status,
                     ack_completed_at, created_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?)
                """,
                (event.event_id, event.channel_id, event.thread_ts, payload, timestamp, timestamp),
            )
            return cursor.rowcount == 1

    def get_event(self, event_id: str) -> CanonicalEvent | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM router_inbox WHERE event_id=?", (event_id,)
            ).fetchone()
        if row is None:
            return None
        return CanonicalEvent.from_dict(json.loads(row[0]))

    def recent_thread_events(
        self,
        channel_id: str,
        thread_ts: str,
        *,
        before_event_id: str,
        limit: int = 20,
    ) -> tuple[CanonicalEvent, ...]:
        """Return prior canonical events for one exact thread, oldest first."""

        bounded_limit = max(1, min(int(limit), 1000))
        with self._lock:
            boundary = self._conn.execute(
                """
                SELECT created_at, event_id
                  FROM router_inbox
                 WHERE event_id=? AND channel_id=? AND thread_ts=?
                """,
                (before_event_id, channel_id, thread_ts),
            ).fetchone()
            if boundary is None:
                return ()
            rows = self._conn.execute(
                """
                SELECT payload_json
                  FROM router_inbox
                 WHERE channel_id=? AND thread_ts=?
                   AND (
                        created_at < ?
                        OR (created_at = ? AND event_id < ?)
                       )
              ORDER BY created_at DESC, event_id DESC
                 LIMIT ?
                """,
                (
                    channel_id,
                    thread_ts,
                    float(boundary[0]),
                    float(boundary[0]),
                    str(boundary[1]),
                    bounded_limit,
                ),
            ).fetchall()

        events: list[CanonicalEvent] = []
        for row in reversed(rows):
            try:
                events.append(CanonicalEvent.from_dict(json.loads(row[0])))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return tuple(events)

    def inbox_status(self, event_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT status FROM router_inbox WHERE event_id=?", (event_id,)).fetchone()
        return str(row[0]) if row else None

    def inbox_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM router_inbox").fetchone()
            return int(row[0])

    def processable_event_ids(
        self,
        *,
        now: float | None = None,
        limit: int = 100,
    ) -> tuple[str, ...]:
        """Return FIFO events that a worker may safely offer to ``process_once``.

        ``pending`` rows are immediately eligible. ``processing`` rows are only
        returned after their thread lease expires. Terminal and quarantined rows
        are never replayed. Only the oldest non-terminal event in each thread is
        exposed, preserving per-thread order without a process-memory queue.
        """

        timestamp = time.time() if now is None else float(now)
        bounded_limit = max(1, min(int(limit), 1000))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT inbox.event_id
                  FROM router_inbox AS inbox
             LEFT JOIN router_thread_leases AS lease
                    ON lease.channel_id=inbox.channel_id
                   AND lease.thread_ts=inbox.thread_ts
                 WHERE inbox.status IN ('pending','processing')
                   AND (lease.lease_deadline IS NULL OR lease.lease_deadline <= ?)
                   AND NOT EXISTS (
                        SELECT 1
                          FROM router_inbox AS earlier
                         WHERE earlier.channel_id=inbox.channel_id
                           AND earlier.thread_ts=inbox.thread_ts
                           AND earlier.status IN ('pending','processing')
                           AND (
                                earlier.created_at < inbox.created_at
                                OR (
                                    earlier.created_at = inbox.created_at
                                    AND earlier.event_id < inbox.event_id
                                )
                               )
                       )
              ORDER BY inbox.created_at, inbox.event_id
                 LIMIT ?
                """,
                (timestamp, bounded_limit),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def quarantine_stale_events_before(
        self,
        cutoff: float,
        *,
        now: float | None = None,
    ) -> tuple[str, ...]:
        """Quarantine pre-start backlog so a new image never sends it late."""

        timestamp = time.time() if now is None else float(now)
        threshold = float(cutoff)
        reason = "event predates Router worker startup freshness window"
        with self._transaction():
            rows = self._conn.execute(
                """
                SELECT inbox.event_id, inbox.channel_id, inbox.thread_ts
                  FROM router_inbox AS inbox
             LEFT JOIN router_thread_leases AS lease
                    ON lease.channel_id=inbox.channel_id
                   AND lease.thread_ts=inbox.thread_ts
                 WHERE inbox.created_at < ?
                   AND (
                        inbox.status='pending'
                        OR (
                            inbox.status='processing'
                            AND (
                                lease.lease_deadline IS NULL
                                OR lease.lease_deadline <= ?
                            )
                        )
                       )
              ORDER BY inbox.created_at, inbox.event_id
                """,
                (threshold, timestamp),
            ).fetchall()
            event_ids = tuple(str(row[0]) for row in rows)
            for row in rows:
                event_id = str(row[0])
                self._conn.execute(
                    """
                    UPDATE router_inbox
                       SET status='quarantined', last_error=?, completed_at=?
                     WHERE event_id=? AND status IN ('pending','processing')
                    """,
                    (reason, timestamp, event_id),
                )
                self._conn.execute(
                    """
                    UPDATE router_outbox
                       SET status='quarantined', attempt_evidence=?, updated_at=?
                     WHERE event_id=?
                       AND status IN ('prepared','attempting','ambiguous')
                    """,
                    (reason, timestamp, event_id),
                )
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO router_completions
                        (event_id, outcome, target_profile, confirmed_message_id,
                         completed_at, expires_at)
                    VALUES (?, 'startup_stale_quarantined', NULL, NULL, ?, ?)
                    """,
                    (event_id, timestamp, timestamp + self.completion_ttl_seconds),
                )
        return event_ids

    def get_thread(self, channel_id: str, thread_ts: str, *, now: float | None = None) -> RouterThreadState:
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                "SELECT * FROM router_threads WHERE channel_id=? AND thread_ts=?",
                (channel_id, thread_ts),
            ).fetchone()
            if row is None:
                state = RouterThreadState(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    updated_at=timestamp,
                    expires_at=timestamp + self.thread_ttl_seconds,
                )
                self._insert_thread(state)
                return state
            state = _row_to_state(row)
            if state.expires_at <= timestamp:
                state = replace(
                    state,
                    task_id=None,
                    owner=None,
                    active_team=None,
                    advisor_stack=(),
                    last_human_target=None,
                    last_event_id=None,
                    state_version=state.state_version + 1,
                    lock_version=state.lock_version + 1,
                    updated_at=timestamp,
                    expires_at=timestamp + self.thread_ttl_seconds,
                )
                self._replace_thread(state)
                self._conn.execute(
                    "DELETE FROM router_handoff_counts WHERE channel_id=? AND thread_ts=?",
                    (channel_id, thread_ts),
                )
            return state

    def compare_and_swap_state(
        self,
        state: RouterThreadState,
        *,
        expected_state_version: int,
        expected_lock_version: int,
        now: float | None = None,
        last_event_id: str | None = None,
    ) -> RouterThreadState:
        timestamp = time.time() if now is None else float(now)
        next_state = replace(
            state,
            state_version=expected_state_version + 1,
            lock_version=expected_lock_version + 1,
            last_event_id=last_event_id if last_event_id is not None else state.last_event_id,
            updated_at=timestamp,
            expires_at=timestamp + self.thread_ttl_seconds,
        )
        with self._transaction():
            cursor = self._conn.execute(
                """
                UPDATE router_threads
                   SET task_id=?, state_version=?, lock_version=?, mode=?, owner=?,
                       active_team=?, advisor_stack=?, last_human_target=?, last_event_id=?,
                       updated_at=?, expires_at=?
                 WHERE channel_id=? AND thread_ts=? AND state_version=? AND lock_version=?
                """,
                (
                    next_state.task_id,
                    next_state.state_version,
                    next_state.lock_version,
                    next_state.mode,
                    next_state.owner,
                    next_state.active_team,
                    json.dumps(list(next_state.advisor_stack), ensure_ascii=False),
                    next_state.last_human_target,
                    next_state.last_event_id,
                    next_state.updated_at,
                    next_state.expires_at,
                    next_state.channel_id,
                    next_state.thread_ts,
                    expected_state_version,
                    expected_lock_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict("router thread state CAS failed")
        return next_state

    def get_handoff_count(self, channel_id: str, thread_ts: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT handoff_count FROM router_handoff_counts WHERE channel_id=? AND thread_ts=?",
                (channel_id, thread_ts),
            ).fetchone()
            return int(row[0]) if row else 0

    def increment_handoff_count(self, channel_id: str, thread_ts: str) -> int:
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO router_handoff_counts(channel_id, thread_ts, handoff_count)
                VALUES (?, ?, 1)
                ON CONFLICT(channel_id, thread_ts)
                DO UPDATE SET handoff_count=handoff_count+1
                """,
                (channel_id, thread_ts),
            )
            return self.get_handoff_count(channel_id, thread_ts)

    def acquire_lease(
        self,
        channel_id: str,
        thread_ts: str,
        lease_owner: str,
        *,
        now: float | None = None,
        lease_seconds: float | None = None,
    ) -> int:
        timestamp = time.time() if now is None else float(now)
        duration = self.lease_seconds if lease_seconds is None else float(lease_seconds)
        with self._transaction():
            row = self._conn.execute(
                "SELECT lease_owner, lease_deadline, generation FROM router_thread_leases WHERE channel_id=? AND thread_ts=?",
                (channel_id, thread_ts),
            ).fetchone()
            if row is not None and row[0] != lease_owner and float(row[1]) > timestamp:
                raise LeaseBusy("router thread lease is owned by another live worker")
            generation = int(row[2]) + 1 if row is not None else 1
            self._conn.execute(
                """
                INSERT INTO router_thread_leases(channel_id, thread_ts, lease_owner, lease_deadline, generation)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(channel_id, thread_ts) DO UPDATE SET
                    lease_owner=excluded.lease_owner,
                    lease_deadline=excluded.lease_deadline,
                    generation=excluded.generation
                """,
                (channel_id, thread_ts, lease_owner, timestamp + duration, generation),
            )
            return generation

    def renew_lease(
        self,
        channel_id: str,
        thread_ts: str,
        lease_owner: str,
        lease_generation: int,
        *,
        now: float | None = None,
    ) -> float:
        """Extend one still-live generation-fenced lease.

        An expired lease is never resurrected: another worker may already be
        eligible to claim it.  The caller must treat failure as an ambiguous
        in-flight operation and stop outbound work.
        """

        timestamp = time.time() if now is None else float(now)
        deadline = timestamp + self.lease_seconds
        with self._transaction():
            cursor = self._conn.execute(
                """
                UPDATE router_thread_leases
                   SET lease_deadline=?
                 WHERE channel_id=? AND thread_ts=?
                   AND lease_owner=? AND generation=?
                   AND lease_deadline>?
                """,
                (
                    deadline,
                    channel_id,
                    thread_ts,
                    lease_owner,
                    int(lease_generation),
                    timestamp,
                ),
            )
            if cursor.rowcount != 1:
                raise LeaseFenceConflict(
                    "router lease cannot be renewed after owner, generation, or deadline changed"
                )
        return deadline

    def release_lease(
        self,
        channel_id: str,
        thread_ts: str,
        lease_owner: str,
        lease_generation: int | None = None,
    ) -> bool:
        with self._transaction():
            if lease_generation is None:
                cursor = self._conn.execute(
                    "DELETE FROM router_thread_leases WHERE channel_id=? AND thread_ts=? AND lease_owner=?",
                    (channel_id, thread_ts, lease_owner),
                )
            else:
                cursor = self._conn.execute(
                    """
                    DELETE FROM router_thread_leases
                     WHERE channel_id=? AND thread_ts=? AND lease_owner=? AND generation=?
                    """,
                    (channel_id, thread_ts, lease_owner, lease_generation),
                )
                if cursor.rowcount != 1:
                    raise LeaseFenceConflict("router lease release was fenced")
            return cursor.rowcount == 1

    def _assert_lease_in_transaction(
        self,
        channel_id: str,
        thread_ts: str,
        lease_owner: str,
        lease_generation: int,
        now: float,
    ) -> None:
        row = self._conn.execute(
            """
            SELECT lease_owner, lease_deadline, generation
              FROM router_thread_leases
             WHERE channel_id=? AND thread_ts=?
            """,
            (channel_id, thread_ts),
        ).fetchone()
        if (
            row is None
            or str(row[0]) != lease_owner
            or int(row[2]) != int(lease_generation)
            or float(row[1]) <= now
        ):
            raise LeaseFenceConflict("router lease owner or generation is stale")

    def _quarantine_in_transaction(
        self,
        event_id: str,
        reason: str,
        timestamp: float,
        *,
        outcome: str = "recovery_attempt_limit",
    ) -> None:
        self._conn.execute(
            """
            UPDATE router_inbox
               SET status='quarantined', last_error=?, completed_at=?
             WHERE event_id=?
            """,
            (reason[:500], timestamp, event_id),
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO router_completions
                (event_id, outcome, target_profile, confirmed_message_id, completed_at, expires_at)
            VALUES (?, ?, NULL, NULL, ?, ?)
            """,
            (event_id, outcome, timestamp, timestamp + self.completion_ttl_seconds),
        )

    def claim_inbox(self, event_id: str, lease_owner: str, *, now: float | None = None) -> bool:
        """Backward-compatible boolean claim wrapper."""
        return bool(self.claim_inbox_with_lease(event_id, lease_owner, now=now))

    def claim_inbox_with_lease(
        self,
        event_id: str,
        lease_owner: str,
        *,
        now: float | None = None,
    ) -> LeaseClaim | None:
        """Claim an inbox event and return the lease generation fencing token.

        A processing event whose lease deadline has passed is recovered in the
        same transaction.  Once the configured attempt cap is reached, the
        event is durably quarantined instead of being requeued indefinitely.
        """
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                """
                SELECT channel_id, thread_ts, status, attempt_count
                  FROM router_inbox
                 WHERE event_id=?
                """,
                (event_id,),
            ).fetchone()
            if row is None:
                return None

            channel_id, thread_ts, status = str(row[0]), str(row[1]), str(row[2])
            attempt_count = int(row[3] or 0)
            lease = self._conn.execute(
                """
                SELECT lease_owner, lease_deadline, generation
                  FROM router_thread_leases
                 WHERE channel_id=? AND thread_ts=?
                """,
                (channel_id, thread_ts),
            ).fetchone()

            if status == "processing":
                lease_live = lease is not None and float(lease[1]) > timestamp
                if lease_live:
                    return None
                outbox = self._conn.execute(
                    "SELECT operation_id, status FROM router_outbox WHERE event_id=?",
                    (event_id,),
                ).fetchone()
                if outbox is not None and str(outbox[1]) in {
                    "attempting",
                    "ambiguous",
                    "quarantined",
                    "confirmed",
                }:
                    reason = (
                        "processing lease expired after outbound attempt; "
                        "automatic retry prohibited"
                    )
                    self._quarantine_in_transaction(
                        event_id,
                        reason,
                        timestamp,
                        outcome="ambiguous_outbound_recovery",
                    )
                    self._conn.execute(
                        """
                        UPDATE router_outbox
                           SET status='quarantined', attempt_evidence=?, updated_at=?
                         WHERE operation_id=? AND status!='confirmed'
                        """,
                        (reason, timestamp, str(outbox[0])),
                    )
                    return LeaseClaim(
                        event_id,
                        channel_id,
                        thread_ts,
                        lease_owner,
                        int(lease[2]) if lease is not None else 0,
                        attempt_count,
                        status="quarantined",
                    )
                if attempt_count >= self.max_processing_attempts:
                    reason = (
                        "processing lease expired; recovery attempt limit "
                        f"reached ({self.max_processing_attempts})"
                    )
                    self._quarantine_in_transaction(event_id, reason, timestamp)
                    return LeaseClaim(
                        event_id,
                        channel_id,
                        thread_ts,
                        lease_owner,
                        int(lease[2]) if lease is not None else 0,
                        attempt_count,
                        status="quarantined",
                    )
                self._conn.execute(
                    """
                    UPDATE router_inbox
                       SET status='pending', claimed_at=NULL,
                           last_error='recovered after lease expiry'
                     WHERE event_id=? AND status='processing'
                    """,
                    (event_id,),
                )
                status = "pending"

            if status != "pending":
                return None

            next_attempt = attempt_count + 1
            generation = self._acquire_lease_in_transaction(
                channel_id,
                thread_ts,
                lease_owner,
                timestamp,
            )
            cursor = self._conn.execute(
                """
                UPDATE router_inbox
                   SET status='processing', claimed_at=?, attempt_count=?, last_error=NULL
                 WHERE event_id=? AND status='pending'
                """,
                (timestamp, next_attempt, event_id),
            )
            if cursor.rowcount != 1:
                return None
            return LeaseClaim(
                event_id,
                channel_id,
                thread_ts,
                lease_owner,
                generation,
                next_attempt,
            )

    def prepare_outbox(
        self,
        event: CanonicalEvent,
        decision: RouteDecision,
        *,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now: float | None = None,
    ) -> str:
        timestamp = time.time() if now is None else float(now)
        operation_id = f"router:{event.event_id}:{decision.action.response_kind}"
        with self._transaction():
            if lease_owner is not None and lease_generation is not None:
                self._assert_lease_in_transaction(
                    event.channel_id,
                    event.thread_ts,
                    lease_owner,
                    lease_generation,
                    timestamp,
                )
            row = self._conn.execute(
                "SELECT status, operation_id FROM router_outbox WHERE event_id=?", (event.event_id,)
            ).fetchone()
            if row is not None:
                return str(row[1])
            self._conn.execute(
                """
                INSERT INTO router_outbox
                    (operation_id, event_id, channel_id, thread_ts, target_profile,
                     content, response_kind, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?)
                """,
                (
                    operation_id,
                    event.event_id,
                    event.channel_id,
                    event.thread_ts,
                    decision.action.target_profile,
                    decision.action.content,
                    decision.action.response_kind,
                    timestamp,
                    timestamp,
                ),
            )
        return operation_id

    def delegation_handoff_reserved(self, event_id: str) -> bool:
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM router_outbox WHERE event_id=? AND status='prepared' "
                "AND attempt_evidence='delegation_handoff_reserved'", (event_id,),
            ).fetchone() is not None

    def reserve_delegation_handoff(
        self, operation_id: str, *, lease_owner: str, lease_generation: int,
    ) -> None:
        """Consume one batch hop before child side effects, once across recovery."""
        with self._transaction():
            row = self._conn.execute(
                "SELECT channel_id, thread_ts, status, attempt_evidence FROM router_outbox WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if row is None or row[2] != "prepared":
                raise StoreError("delegation parent is not prepared")
            self._assert_lease_in_transaction(
                str(row[0]), str(row[1]), lease_owner, lease_generation, time.time(),
            )
            if row[3] == "delegation_handoff_reserved":
                return
            self._conn.execute(
                "INSERT INTO router_handoff_counts(channel_id, thread_ts, handoff_count) VALUES (?, ?, 1) "
                "ON CONFLICT(channel_id, thread_ts) DO UPDATE SET handoff_count=handoff_count+1",
                (row[0], row[1]),
            )
            self._conn.execute(
                "UPDATE router_outbox SET attempt_evidence='delegation_handoff_reserved' WHERE operation_id=?",
                (operation_id,),
            )

    def outbox_content(self, operation_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT content FROM router_outbox WHERE operation_id=?", (operation_id,)
            ).fetchone()
            return str(row[0]) if row and row[0] is not None else None

    def record_delegation_result(
        self, operation_id: str, content: str, *, lease_owner: str, lease_generation: int,
    ) -> None:
        """Save a bounded execution receipt before confirming its child outbox."""
        with self._transaction():
            row = self._conn.execute(
                "SELECT channel_id, thread_ts FROM router_outbox WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise StoreError("unknown delegation operation")
            self._assert_lease_in_transaction(
                str(row[0]), str(row[1]), lease_owner, lease_generation, time.time(),
            )
            cursor = self._conn.execute(
                "UPDATE router_outbox SET content=? WHERE operation_id=? AND status='attempting'",
                (content, operation_id),
            )
            if cursor.rowcount != 1:
                raise StoreError("delegation operation is not attempting")

    def set_prepared_outbox_content(
        self,
        operation_id: str,
        content: str,
        *,
        lease_owner: str,
        lease_generation: int,
        now: float | None = None,
    ) -> None:
        """Persist deterministic content before allowing its sole send attempt."""

        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                "SELECT channel_id, thread_ts FROM router_outbox WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise StoreError("unknown outbox operation")
            self._assert_lease_in_transaction(
                str(row[0]),
                str(row[1]),
                lease_owner,
                lease_generation,
                timestamp,
            )
            cursor = self._conn.execute(
                """
                UPDATE router_outbox SET content=?, updated_at=?
                 WHERE operation_id=? AND status='prepared'
                """,
                (content, timestamp, operation_id),
            )
            if cursor.rowcount != 1:
                raise StoreError("outbox operation is not prepared")

    def mark_outbox_attempting(
        self,
        operation_id: str,
        evidence: str,
        *,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            if lease_owner is not None and lease_generation is not None:
                row = self._conn.execute(
                    "SELECT channel_id, thread_ts FROM router_outbox WHERE operation_id=?",
                    (operation_id,),
                ).fetchone()
                if row is None:
                    raise StoreError("unknown outbox operation")
                self._assert_lease_in_transaction(
                    str(row[0]),
                    str(row[1]),
                    lease_owner,
                    lease_generation,
                    timestamp,
                )
            cursor = self._conn.execute(
                """
                UPDATE router_outbox SET status='attempting', attempt_evidence=?, updated_at=?
                WHERE operation_id=? AND status='prepared'
                """,
                (evidence, timestamp, operation_id),
            )
            if cursor.rowcount != 1:
                raise StoreError("outbox operation is not prepared")

    def finalize_terminal_report(
        self,
        operation_id: str,
        *,
        message_id: str | None,
        lease_owner: str,
        lease_generation: int,
        now: float | None = None,
    ) -> None:
        """Confirm a meeting terminal report without reopening its completed inbox event."""

        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                """
                SELECT event_id, channel_id, thread_ts, target_profile, response_kind, status
                  FROM router_outbox WHERE operation_id=?
                """,
                (operation_id,),
            ).fetchone()
            if row is None or str(row[5]) != "attempting":
                raise StoreError("terminal report outbox cannot be confirmed")
            self._assert_lease_in_transaction(
                str(row[1]),
                str(row[2]),
                lease_owner,
                lease_generation,
                timestamp,
            )
            self._conn.execute(
                "UPDATE router_outbox SET status='confirmed', updated_at=? WHERE operation_id=?",
                (timestamp, operation_id),
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO router_completions
                    (event_id, outcome, target_profile, confirmed_message_id,
                     completed_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row[0]),
                    str(row[4]),
                    row[3],
                    message_id,
                    timestamp,
                    timestamp + self.completion_ttl_seconds,
                ),
            )

    def finalize_confirmed(
        self,
        operation_id: str,
        *,
        target_profile: str | None,
        message_id: str | None,
        next_state: RouterThreadState | None,
        expected_state_version: int | None = None,
        expected_lock_version: int | None = None,
        handoff_count_delta: int = 0,
        outcome: str = "confirmed",
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                "SELECT event_id, channel_id, thread_ts, status FROM router_outbox WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if row is None or row[3] not in {"prepared", "attempting"}:
                raise StoreError("outbox operation cannot be confirmed")
            if lease_owner is not None and lease_generation is not None:
                self._assert_lease_in_transaction(
                    str(row[1]),
                    str(row[2]),
                    lease_owner,
                    lease_generation,
                    timestamp,
                )
            self._conn.execute(
                "UPDATE router_outbox SET status='confirmed', updated_at=? WHERE operation_id=?",
                (timestamp, operation_id),
            )
            if next_state is not None:
                current = self._conn.execute(
                    "SELECT state_version, lock_version FROM router_threads WHERE channel_id=? AND thread_ts=?",
                    (row[1], row[2]),
                ).fetchone()
                if current is None:
                    raise StoreError("thread disappeared before outbox finalize")
                expected_state_version = current[0] if expected_state_version is None else expected_state_version
                expected_lock_version = current[1] if expected_lock_version is None else expected_lock_version
                updated = replace(next_state, updated_at=timestamp, expires_at=timestamp + self.thread_ttl_seconds)
                cursor = self._conn.execute(
                    """
                    UPDATE router_threads SET task_id=?, state_version=?, lock_version=?, mode=?,
                        owner=?, active_team=?, advisor_stack=?, last_human_target=?, last_event_id=?,
                        updated_at=?, expires_at=?
                    WHERE channel_id=? AND thread_ts=? AND state_version=? AND lock_version=?
                    """,
                    (
                        updated.task_id,
                        expected_state_version + 1,
                        expected_lock_version + 1,
                        updated.mode,
                        updated.owner,
                        updated.active_team,
                        json.dumps(list(updated.advisor_stack), ensure_ascii=False),
                        updated.last_human_target,
                        updated.last_event_id,
                        updated.updated_at,
                        updated.expires_at,
                        row[1],
                        row[2],
                        expected_state_version,
                        expected_lock_version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CASConflict("router thread state CAS failed during finalize")
            if handoff_count_delta:
                self._conn.execute(
                    """
                    INSERT INTO router_handoff_counts(channel_id, thread_ts, handoff_count)
                    VALUES (?, ?, ?)
                    ON CONFLICT(channel_id, thread_ts)
                    DO UPDATE SET handoff_count=handoff_count+excluded.handoff_count
                    """,
                    (row[1], row[2], handoff_count_delta),
                )
            self._conn.execute(
                "UPDATE router_inbox SET status='completed', completed_at=? WHERE event_id=?",
                (timestamp, row[0]),
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO router_completions
                    (event_id, outcome, target_profile, confirmed_message_id, completed_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row[0],
                    outcome,
                    target_profile,
                    message_id,
                    timestamp,
                    timestamp + self.completion_ttl_seconds,
                ),
            )

    def complete_silence(
        self,
        event_id: str,
        *,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now: float | None = None,
    ) -> None:
        self._complete_inbox(
            event_id,
            "silence",
            None,
            lease_owner=lease_owner,
            lease_generation=lease_generation,
            now=now,
        )

    def complete_without_outbox(
        self,
        event_id: str,
        next_state: RouterThreadState,
        *,
        expected_state_version: int,
        expected_lock_version: int,
        outcome: str,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now: float | None = None,
    ) -> None:
        """Commit a non-posting state transition and terminal inbox outcome."""
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            if lease_owner is not None and lease_generation is not None:
                self._assert_lease_in_transaction(
                    next_state.channel_id,
                    next_state.thread_ts,
                    lease_owner,
                    lease_generation,
                    timestamp,
                )
            cursor = self._conn.execute(
                """
                UPDATE router_threads SET task_id=?, state_version=?, lock_version=?, mode=?,
                    owner=?, active_team=?, advisor_stack=?, last_human_target=?, last_event_id=?,
                    updated_at=?, expires_at=?
                WHERE channel_id=? AND thread_ts=? AND state_version=? AND lock_version=?
                """,
                (
                    next_state.task_id,
                    expected_state_version + 1,
                    expected_lock_version + 1,
                    next_state.mode,
                    next_state.owner,
                    next_state.active_team,
                    json.dumps(list(next_state.advisor_stack), ensure_ascii=False),
                    next_state.last_human_target,
                    next_state.last_event_id,
                    timestamp,
                    timestamp + self.thread_ttl_seconds,
                    next_state.channel_id,
                    next_state.thread_ts,
                    expected_state_version,
                    expected_lock_version,
                ),
            )
            if cursor.rowcount != 1:
                raise CASConflict("router thread state CAS failed during non-posting finalize")
            self._conn.execute(
                "UPDATE router_inbox SET status='completed', completed_at=? WHERE event_id=?",
                (timestamp, event_id),
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO router_completions
                    (event_id, outcome, target_profile, confirmed_message_id, completed_at, expires_at)
                VALUES (?, ?, NULL, NULL, ?, ?)
                """,
                (event_id, outcome, timestamp, timestamp + self.completion_ttl_seconds),
            )

    def quarantine(
        self,
        event_id: str,
        *,
        reason: str,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            if lease_owner is not None and lease_generation is not None:
                row = self._conn.execute(
                    "SELECT channel_id, thread_ts FROM router_inbox WHERE event_id=?",
                    (event_id,),
                ).fetchone()
                if row is None:
                    raise StoreError("unknown inbox event")
                self._assert_lease_in_transaction(
                    str(row[0]),
                    str(row[1]),
                    lease_owner,
                    lease_generation,
                    timestamp,
                )
            self._conn.execute(
                "UPDATE router_inbox SET status='quarantined', last_error=?, completed_at=? WHERE event_id=?",
                (reason[:500], timestamp, event_id),
            )
            self._conn.execute(
                "UPDATE router_outbox SET status='quarantined', attempt_evidence=?, updated_at=? WHERE event_id=? AND status IN ('prepared','attempting','ambiguous')",
                (reason[:500], timestamp, event_id),
            )

    def mark_ambiguous_and_quarantine(
        self,
        operation_id: str,
        *,
        reason: str,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            row = self._conn.execute(
                "SELECT event_id, channel_id, thread_ts FROM router_outbox WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise StoreError("unknown outbox operation")
            if lease_owner is not None and lease_generation is not None:
                self._assert_lease_in_transaction(
                    str(row[1]),
                    str(row[2]),
                    lease_owner,
                    lease_generation,
                    timestamp,
                )
            self._conn.execute(
                "UPDATE router_outbox SET status='ambiguous', attempt_evidence=?, updated_at=? WHERE operation_id=?",
                (reason[:500], timestamp, operation_id),
            )
            self._conn.execute(
                "UPDATE router_outbox SET status='quarantined', updated_at=? WHERE operation_id=?",
                (timestamp, operation_id),
            )
            self._conn.execute(
                "UPDATE router_inbox SET status='quarantined', last_error=?, completed_at=? WHERE event_id=?",
                (reason[:500], timestamp, row[0]),
            )

    def cleanup_expired_completions(self, *, now: float | None = None) -> int:
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            cursor = self._conn.execute(
                """
                DELETE FROM router_completions
                 WHERE expires_at <= ?
                   AND NOT EXISTS (
                       SELECT 1 FROM router_inbox i
                       WHERE i.event_id=router_completions.event_id
                         AND i.status IN ('pending','processing')
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM router_outbox o
                       WHERE o.event_id=router_completions.event_id
                         AND o.status IN ('prepared','attempting','ambiguous')
                   )
                """,
                (timestamp,),
            )
            return cursor.rowcount

    def outbox_status(self, operation_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT status FROM router_outbox WHERE operation_id=?", (operation_id,)).fetchone()
            return str(row[0]) if row else None

    def _complete_inbox(
        self,
        event_id: str,
        outcome: str,
        target: str | None,
        *,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now: float | None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        with self._transaction():
            if lease_owner is not None and lease_generation is not None:
                row = self._conn.execute(
                    "SELECT channel_id, thread_ts FROM router_inbox WHERE event_id=?",
                    (event_id,),
                ).fetchone()
                if row is None:
                    raise StoreError("unknown inbox event")
                self._assert_lease_in_transaction(
                    str(row[0]),
                    str(row[1]),
                    lease_owner,
                    lease_generation,
                    timestamp,
                )
            self._conn.execute(
                "UPDATE router_inbox SET status='completed', completed_at=? WHERE event_id=?",
                (timestamp, event_id),
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO router_completions
                    (event_id, outcome, target_profile, confirmed_message_id, completed_at, expires_at)
                VALUES (?, ?, ?, NULL, ?, ?)
                """,
                (event_id, outcome, target, timestamp, timestamp + self.completion_ttl_seconds),
            )

    def _acquire_lease_in_transaction(self, channel_id: str, thread_ts: str, owner: str, now: float) -> int:
        row = self._conn.execute(
            "SELECT lease_owner, lease_deadline, generation FROM router_thread_leases WHERE channel_id=? AND thread_ts=?",
            (channel_id, thread_ts),
        ).fetchone()
        if row is not None and row[0] != owner and float(row[1]) > now:
            raise LeaseBusy("router thread lease is owned by another live worker")
        generation = int(row[2]) + 1 if row else 1
        self._conn.execute(
            """
            INSERT INTO router_thread_leases(channel_id, thread_ts, lease_owner, lease_deadline, generation)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(channel_id, thread_ts) DO UPDATE SET
                lease_owner=excluded.lease_owner, lease_deadline=excluded.lease_deadline,
                generation=excluded.generation
            """,
            (channel_id, thread_ts, owner, now + self.lease_seconds, generation),
        )
        return generation

    def _insert_thread(self, state: RouterThreadState) -> None:
        self._conn.execute(
            """
            INSERT INTO router_threads
                (channel_id, thread_ts, task_id, state_version, lock_version, mode,
                 owner, active_team, advisor_stack, last_human_target, last_event_id,
                 updated_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.channel_id,
                state.thread_ts,
                state.task_id,
                state.state_version,
                state.lock_version,
                state.mode,
                state.owner,
                state.active_team,
                json.dumps(list(state.advisor_stack), ensure_ascii=False),
                state.last_human_target,
                state.last_event_id,
                state.updated_at,
                state.expires_at,
            ),
        )

    def _replace_thread(self, state: RouterThreadState) -> None:
        self._conn.execute(
            """
            UPDATE router_threads SET task_id=?, state_version=?, lock_version=?, mode=?,
                owner=?, active_team=?, advisor_stack=?, last_human_target=?, last_event_id=?,
                updated_at=?, expires_at=? WHERE channel_id=? AND thread_ts=?
            """,
            (
                state.task_id,
                state.state_version,
                state.lock_version,
                state.mode,
                state.owner,
                state.active_team,
                json.dumps(list(state.advisor_stack), ensure_ascii=False),
                state.last_human_target,
                state.last_event_id,
                state.updated_at,
                state.expires_at,
                state.channel_id,
                state.thread_ts,
            ),
        )

    class _Transaction:
        def __init__(self, store: "RouterStore") -> None:
            self.store = store

        def __enter__(self) -> "RouterStore._Transaction":
            self.store._lock.acquire()
            self.store._conn.execute("BEGIN IMMEDIATE")
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            try:
                self.store._conn.execute("ROLLBACK" if exc_type else "COMMIT")
            finally:
                self.store._lock.release()

    def _transaction(self) -> "RouterStore._Transaction":
        return self._Transaction(self)


def _row_to_meeting(row: sqlite3.Row) -> MeetingState:
    try:
        participants = json.loads(row["participants_json"])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StoreError("meeting participants_json is malformed") from exc
    if not isinstance(participants, list) or not all(isinstance(item, str) for item in participants):
        raise StoreError("meeting participants_json is malformed")
    return_state = None
    raw_return_state = row["return_state_json"]
    if raw_return_state is not None:
        try:
            return_value = json.loads(raw_return_state)
            if not isinstance(return_value, dict):
                raise ValueError("return state must be an object")
            return_state = RouterThreadState.from_dict(return_value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StoreError("meeting return_state_json is malformed") from exc
    return MeetingState(
        meeting_id=str(row["meeting_id"]),
        channel_id=str(row["channel_id"]),
        thread_ts=str(row["thread_ts"]),
        status=str(row["status"]),
        generation=int(row["generation"]),
        current_round=int(row["current_round"]),
        participants=tuple(participants),
        timeout_round_streak=int(row["timeout_round_streak"]),
        all_pass_streak=int(row["all_pass_streak"]),
        return_state=return_state,
        preflight_result_json=row["preflight_result_json"],
        block_reason=row["block_reason"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _row_to_meeting_round(row: sqlite3.Row) -> MeetingRoundState:
    return MeetingRoundState(
        meeting_id=str(row["meeting_id"]),
        round_id=int(row["round_id"]),
        generation=int(row["generation"]),
        status=str(row["status"]),
        packet_json=str(row["packet_json"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _row_to_meeting_candidate(row: sqlite3.Row) -> MeetingCandidate:
    raw_reason = row["reason_to_speak"]
    reason_to_speak = None if raw_reason is None else bool(int(raw_reason))
    return MeetingCandidate(
        meeting_id=str(row["meeting_id"]),
        round_id=int(row["round_id"]),
        generation=int(row["generation"]),
        participant=str(row["participant"]),
        status=str(row["status"]),
        reason_to_speak=reason_to_speak,
        reason_class=str(row["reason_class"]),
        statement=str(row["statement"]),
        error_class=row["error_class"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _row_to_state(row: sqlite3.Row) -> RouterThreadState:
    try:
        advisors = json.loads(row["advisor_stack"])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StoreError("router thread advisor_stack is malformed") from exc
    if not isinstance(advisors, list) or any(not isinstance(item, str) for item in advisors):
        raise StoreError("router thread advisor_stack is malformed")
    return RouterThreadState(
        channel_id=str(row["channel_id"]),
        thread_ts=str(row["thread_ts"]),
        task_id=row["task_id"],
        state_version=int(row["state_version"]),
        lock_version=int(row["lock_version"]),
        mode=str(row["mode"]),
        owner=row["owner"],
        active_team=row["active_team"],
        advisor_stack=tuple(advisors),
        last_human_target=row["last_human_target"],
        last_event_id=row["last_event_id"],
        updated_at=float(row["updated_at"]),
        expires_at=float(row["expires_at"]),
    )
