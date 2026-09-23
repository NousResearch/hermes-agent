"""Profile-local durable journal for safe connected-tool mutation projections."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Callable, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

Title = Annotated[StrictStr, Field(min_length=1, max_length=128)]
Description = Annotated[StrictStr, Field(min_length=1, max_length=1_000)]
Context = Annotated[StrictStr, Field(min_length=1, max_length=2_000)]
Provider = Annotated[StrictStr, Field(min_length=1, max_length=80)]
Operation = Annotated[StrictStr, Field(min_length=1, max_length=80)]
Destination = Annotated[StrictStr, Field(min_length=1, max_length=128)]


class _StrictActionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MutationStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class MutationType(StrEnum):
    CALENDAR = "calendar"
    TODOIST = "todoist"
    NOTE = "note"
    HOME_AUTOMATION = "home_automation"
    NETWORK = "network"
    CONNECTED_TOOL_CHANGE = "connected_tool_change"


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Mutation timestamps must include a timezone")
    return value


class MutationEvent(_StrictActionModel):
    """The only payload that may cross the private dashboard bridge."""

    source_event_key: UUID
    status: MutationStatus
    action_type: MutationType
    title: Title
    description: Description
    provider: Provider
    operation: Operation
    destination: Destination | None = None
    occurred_at: datetime
    context: Context
    requires_receipt: StrictBool = False

    _validate_occurred_at = field_validator("occurred_at")(_require_aware)


class MutationPage(_StrictActionModel):
    schema_version: Annotated[StrictStr, Field(pattern=r"^1$")] = "1"
    events: list[MutationEvent] = Field(max_length=100)
    next_cursor: Annotated[StrictStr, Field(min_length=1, max_length=512)] | None = None


class MutationCursorInvalid(ValueError):
    """The caller supplied a cursor that cannot be used for this profile."""


class MutationConflict(ValueError):
    """The same idempotency key was used for a different safe event."""


class OneShotInProgress(ValueError):
    """A one-shot key is claimed by an unfinished execution."""


class ActionJournal:
    """SQLite-backed append-only journal scoped to one Hermes profile."""

    _SCHEMA_VERSION = 1
    _MAX_CURSOR_CHARS = 512

    def __init__(
        self,
        path: str | Path,
        *,
        profile_key: str,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = str(path)
        raw_profile_key = str(profile_key).strip()
        if not raw_profile_key:
            raise ValueError("profile_key is required")
        # The profile selector is only an isolation input.  Store a stable
        # digest so absolute home paths, usernames, or deployment labels never
        # land in the SQLite file or cursor payload.
        self.profile_key = hashlib.sha256(raw_profile_key.encode("utf-8")).hexdigest()
        self._now = now or datetime.now
        self._lock = threading.RLock()
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._configure()
        self._create_schema()

    def _configure(self) -> None:
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        if self.path != ":memory:":
            # SQLite does not apply busy_timeout while changing the journal
            # mode. A concurrent profile-local opener may hold the schema
            # lock for a few milliseconds; retry the idempotent pragma so
            # startup does not turn a safe duplicate append into a flaky
            # failure.
            for attempt in range(20):
                try:
                    self._connection.execute("PRAGMA journal_mode = WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).casefold() or attempt == 19:
                        raise
                    time.sleep(0.01)

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS mutation_journal (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_key TEXT NOT NULL,
                source_event_key TEXT NOT NULL,
                status TEXT NOT NULL,
                action_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                provider TEXT NOT NULL,
                operation TEXT NOT NULL,
                destination TEXT,
                occurred_at TEXT NOT NULL,
                context TEXT NOT NULL,
                requires_receipt INTEGER NOT NULL CHECK (requires_receipt IN (0, 1)),
                UNIQUE (profile_key, source_event_key)
            );
            CREATE INDEX IF NOT EXISTS mutation_journal_profile_sequence
                ON mutation_journal (profile_key, sequence);
            CREATE TABLE IF NOT EXISTS one_shot_requests (
                profile_key TEXT NOT NULL,
                request_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('pending', 'complete')),
                result_json TEXT,
                PRIMARY KEY (profile_key, request_key)
            );
            CREATE TABLE IF NOT EXISTS start_loop_requests (
                profile_key TEXT NOT NULL,
                request_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('pending', 'complete')),
                progress_json TEXT,
                result_json TEXT,
                PRIMARY KEY (profile_key, request_key)
            );
            """
        )
        # The operation ledger was introduced before staged topic creation was
        # durable. Keep existing profile databases forward-compatible without
        # rewriting or exposing their mutation rows.
        columns = {
            row["name"]
            for row in self._connection.execute(
                "PRAGMA table_info(start_loop_requests)"
            ).fetchall()
        }
        if "progress_json" not in columns:
            self._connection.execute(
                "ALTER TABLE start_loop_requests ADD COLUMN progress_json TEXT"
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def append(self, event: MutationEvent) -> tuple[MutationEvent, bool]:
        if not isinstance(event, MutationEvent):
            event = MutationEvent.model_validate(event)
        with self._lock:
            key = str(event.source_event_key)
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT * FROM mutation_journal
                    WHERE profile_key = ? AND source_event_key = ?
                    """,
                    (self.profile_key, key),
                ).fetchone()
                if row is not None:
                    existing = self._event_from_row(row)
                    if existing != event:
                        raise MutationConflict("source_event_key already has another event")
                    self._connection.execute("COMMIT")
                    return existing, False
                self._connection.execute(
                    """
                    INSERT INTO mutation_journal (
                        profile_key, source_event_key, status, action_type, title,
                        description, provider, operation, destination, occurred_at,
                        context, requires_receipt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.profile_key,
                        key,
                        event.status.value,
                        event.action_type.value,
                        event.title,
                        event.description,
                        event.provider,
                        event.operation,
                        event.destination,
                        event.occurred_at.isoformat(),
                        event.context,
                        int(event.requires_receipt),
                    ),
                )
                self._connection.execute("COMMIT")
                return event, True
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def list(self, *, after_cursor: str | None, limit: int = 100) -> MutationPage:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._lock:
            sequence = 0 if after_cursor is None else self._decode_cursor(after_cursor)
            rows = self._connection.execute(
                """
                SELECT * FROM mutation_journal
                WHERE profile_key = ? AND sequence > ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                (self.profile_key, sequence, limit + 1),
            ).fetchall()
            visible = rows[:limit]
            # The dashboard persists this value after every poll. Keep a
            # high-water cursor even on a terminal page; otherwise a normal
            # empty poll would reset the dashboard to sequence zero and replay
            # the complete journal on its next request. An empty first poll
            # has no high-water mark, while an empty continuation echoes the
            # caller's already-validated cursor.
            next_cursor = (
                self._encode_cursor(int(visible[-1]["sequence"]))
                if visible
                else after_cursor
            )
            return MutationPage(
                schema_version="1",
                events=[self._event_from_row(row) for row in visible],
                next_cursor=next_cursor,
            )

    def get(self, source_event_key: UUID | str) -> MutationEvent | None:
        """Return one event for observer replay/idempotency checks."""
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM mutation_journal
                WHERE profile_key = ? AND source_event_key = ?
                """,
                (self.profile_key, str(source_event_key)),
            ).fetchone()
            return None if row is None else self._event_from_row(row)

    def claim_one_shot(self, request_key: UUID | str, fingerprint: str) -> str | None:
        """Claim a one-shot request, returning a stored safe result on replay."""
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("invalid one-shot fingerprint")
        with self._lock:
            key = str(request_key)
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT fingerprint, state, result_json FROM one_shot_requests
                    WHERE profile_key = ? AND request_key = ?
                    """,
                    (self.profile_key, key),
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        """
                        INSERT INTO one_shot_requests
                            (profile_key, request_key, fingerprint, state)
                        VALUES (?, ?, ?, 'pending')
                        """,
                        (self.profile_key, key, fingerprint),
                    )
                    self._connection.execute("COMMIT")
                    return None
                if row["fingerprint"] != fingerprint:
                    raise MutationConflict("one-shot request key has another payload")
                if row["state"] == "pending":
                    raise OneShotInProgress("one-shot request is still pending")
                result_json = row["result_json"]
                if not isinstance(result_json, str):
                    raise OneShotInProgress("one-shot request result is incomplete")
                self._connection.execute("COMMIT")
                return result_json
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def complete_one_shot(self, request_key: UUID | str, result_json: str) -> None:
        """Store a validated, safe one-shot result for durable replay."""
        if not isinstance(result_json, str) or not result_json:
            raise ValueError("invalid one-shot result")
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE one_shot_requests
                SET state = 'complete', result_json = ?
                WHERE profile_key = ? AND request_key = ? AND state = 'pending'
                """,
                (result_json, self.profile_key, str(request_key)),
            )
            if cursor.rowcount != 1:
                raise OneShotInProgress("one-shot request is not pending")

    def claim_start_loop(self, request_key: UUID | str, fingerprint: str) -> str | None:
        return self._claim_operation(
            "start_loop_requests", request_key=request_key, fingerprint=fingerprint
        )

    def complete_start_loop(self, request_key: UUID | str, result_json: str) -> None:
        self._complete_operation(
            "start_loop_requests", request_key=request_key, result_json=result_json
        )

    def _claim_operation(
        self, table: str, *, request_key: UUID | str, fingerprint: str
    ) -> str | None:
        if table not in {"start_loop_requests"}:
            raise ValueError("invalid operation table")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("invalid operation fingerprint")
        with self._lock:
            key = str(request_key)
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    f"SELECT fingerprint, state, progress_json, result_json FROM {table} "
                    "WHERE profile_key = ? AND request_key = ?",
                    (self.profile_key, key),
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        f"INSERT INTO {table} "
                        "(profile_key, request_key, fingerprint, state) "
                        "VALUES (?, ?, ?, 'pending')",
                        (self.profile_key, key, fingerprint),
                    )
                    self._connection.execute("COMMIT")
                    return None
                if row["fingerprint"] != fingerprint:
                    raise MutationConflict("operation key has another payload")
                if row["state"] == "pending":
                    # A start-loop retry after a gateway restart may resume a
                    # staged operation.  A process-local concurrent retry with
                    # no recorded stage still fails closed.
                    if table == "start_loop_requests" and row["progress_json"]:
                        self._connection.execute("COMMIT")
                        return None
                    raise OneShotInProgress("operation is still pending")
                result_json = row["result_json"]
                if not isinstance(result_json, str):
                    raise OneShotInProgress("operation result is incomplete")
                self._connection.execute("COMMIT")
                return result_json
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def _complete_operation(
        self, table: str, *, request_key: UUID | str, result_json: str
    ) -> None:
        if table not in {"start_loop_requests"}:
            raise ValueError("invalid operation table")
        if not isinstance(result_json, str) or not result_json:
            raise ValueError("invalid operation result")
        with self._lock:
            cursor = self._connection.execute(
                f"UPDATE {table} SET state = 'complete', result_json = ? "
                "WHERE profile_key = ? AND request_key = ? AND state = 'pending'",
                (result_json, self.profile_key, str(request_key)),
            )
            if cursor.rowcount != 1:
                raise OneShotInProgress("operation is not pending")

    def get_start_loop_progress(self, request_key: UUID | str) -> dict[str, object] | None:
        """Return private staged start-loop state for crash reconciliation."""
        with self._lock:
            row = self._connection.execute(
                """
                SELECT progress_json FROM start_loop_requests
                WHERE profile_key = ? AND request_key = ?
                """,
                (self.profile_key, str(request_key)),
            ).fetchone()
            if row is None or not isinstance(row["progress_json"], str):
                return None
            try:
                value = json.loads(row["progress_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
            return value if isinstance(value, dict) else None

    def update_start_loop_progress(
        self, request_key: UUID | str, progress: dict[str, object]
    ) -> None:
        """Persist bounded private stage output while a start-loop is pending."""
        if not isinstance(progress, dict) or not progress:
            raise ValueError("start-loop progress must be a non-empty object")
        encoded = json.dumps(progress, sort_keys=True, separators=(",", ":"))
        if len(encoded) > 4_000:
            raise ValueError("start-loop progress is too large")
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE start_loop_requests
                SET progress_json = ?
                WHERE profile_key = ? AND request_key = ? AND state = 'pending'
                """,
                (encoded, self.profile_key, str(request_key)),
            )
            if cursor.rowcount != 1:
                raise OneShotInProgress("start-loop request is not pending")

    def _encode_cursor(self, sequence: int) -> str:
        profile_digest = hashlib.sha256(self.profile_key.encode("utf-8")).hexdigest()[:16]
        raw = json.dumps(
            {"v": self._SCHEMA_VERSION, "s": sequence, "p": profile_digest},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def _decode_cursor(self, cursor: str) -> int:
        if not isinstance(cursor, str) or not cursor or len(cursor) > self._MAX_CURSOR_CHARS:
            raise MutationCursorInvalid("invalid cursor")
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
            value = json.loads(decoded)
        except (ValueError, TypeError, json.JSONDecodeError):
            raise MutationCursorInvalid("invalid cursor") from None
        expected_profile = hashlib.sha256(self.profile_key.encode("utf-8")).hexdigest()[:16]
        if (
            not isinstance(value, dict)
            or set(value) != {"v", "s", "p"}
            or value.get("v") != self._SCHEMA_VERSION
            or value.get("p") != expected_profile
            or isinstance(value.get("s"), bool)
            or not isinstance(value.get("s"), int)
            or value["s"] < 0
        ):
            raise MutationCursorInvalid("invalid cursor")
        maximum = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM mutation_journal WHERE profile_key = ?",
            (self.profile_key,),
        ).fetchone()[0]
        if value["s"] > int(maximum):
            raise MutationCursorInvalid("invalid cursor")
        return value["s"]

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> MutationEvent:
        return MutationEvent.model_validate(
            {
                "source_event_key": row["source_event_key"],
                "status": row["status"],
                "action_type": row["action_type"],
                "title": row["title"],
                "description": row["description"],
                "provider": row["provider"],
                "operation": row["operation"],
                "destination": row["destination"],
                "occurred_at": row["occurred_at"],
                "context": row["context"],
                "requires_receipt": bool(row["requires_receipt"]),
            }
        )


__all__ = [
    "ActionJournal",
    "MutationConflict",
    "MutationCursorInvalid",
    "MutationEvent",
    "MutationPage",
    "MutationStatus",
    "MutationType",
    "OneShotInProgress",
]
