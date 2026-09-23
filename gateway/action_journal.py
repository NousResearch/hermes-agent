"""Profile-local durable journal for safe connected-tool mutation projections."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
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
        self.profile_key = str(profile_key).strip()
        if not self.profile_key:
            raise ValueError("profile_key is required")
        self._now = now or datetime.now
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
            """
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def append(self, event: MutationEvent) -> tuple[MutationEvent, bool]:
        if not isinstance(event, MutationEvent):
            event = MutationEvent.model_validate(event)
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
        has_more = len(rows) > limit
        visible = rows[:limit]
        next_cursor = (
            self._encode_cursor(int(visible[-1]["sequence"]))
            if has_more and visible
            else None
        )
        return MutationPage(
            schema_version="1",
            events=[self._event_from_row(row) for row in visible],
            next_cursor=next_cursor,
        )

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
]
