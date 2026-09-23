from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from gateway.action_journal import (
    ActionJournal,
    MutationConflict,
    MutationCursorInvalid,
    MutationEvent,
    OneShotInProgress,
)

NOW = datetime(2026, 9, 22, 18, 0, tzinfo=UTC)


def event(*, key: UUID | None = None, status: str = "succeeded") -> MutationEvent:
    return MutationEvent.model_validate(
        {
            "source_event_key": key or uuid4(),
            "status": status,
            "action_type": "calendar",
            "title": "Added dentist appointment",
            "description": "Dentist appointment added for tomorrow at 2:00 PM.",
            "provider": "Google Calendar",
            "operation": "create_event",
            "destination": "Personal calendar",
            "occurred_at": NOW,
            "context": "A dentist appointment was added for tomorrow at 2:00 PM.",
            "requires_receipt": True,
        }
    )


def test_mutation_event_rejects_raw_payload_unknown_fields_and_naive_time() -> None:
    payload = event().model_dump()
    with pytest.raises(ValidationError):
        MutationEvent.model_validate({**payload, "raw_output": "secret"})
    with pytest.raises(ValidationError):
        MutationEvent.model_validate({**payload, "occurred_at": NOW.replace(tzinfo=None)})


def test_mutation_event_bounds_safe_fields_and_uses_closed_enums() -> None:
    payload = event().model_dump()
    with pytest.raises(ValidationError):
        MutationEvent.model_validate({**payload, "title": "x" * 129})
    with pytest.raises(ValidationError):
        MutationEvent.model_validate({**payload, "action_type": "unknown"})
    with pytest.raises(ValidationError):
        MutationEvent.model_validate({**payload, "requires_receipt": 1})


def test_event_survives_repository_restart_and_hides_profile_column(tmp_path: Path) -> None:
    path = tmp_path / "actions.sqlite3"
    first = ActionJournal(path, profile_key="becky", now=lambda: NOW)
    stored, created = first.append(event())
    assert created is True
    first.close()

    second = ActionJournal(path, profile_key="becky", now=lambda: NOW + timedelta(seconds=1))
    page = second.list(after_cursor=None, limit=10)
    assert page.events == [stored]
    assert "profile_key" not in stored.model_dump()
    assert "raw_output" not in stored.model_dump()
    second.close()


def test_profile_isolation_rejects_cursor_from_another_profile(tmp_path: Path) -> None:
    path = tmp_path / "actions.sqlite3"
    becky = ActionJournal(path, profile_key="becky")
    _, _ = becky.append(event())
    cursor = becky.list(after_cursor=None, limit=1).next_cursor
    assert cursor is not None

    # A cursor from a populated profile cannot be accepted by another profile.
    _, _ = becky.append(event())
    cursor = becky.list(after_cursor=None, limit=1).next_cursor
    assert cursor is not None
    sanchez = ActionJournal(path, profile_key="sanchez")
    with pytest.raises(MutationCursorInvalid):
        sanchez.list(after_cursor=cursor, limit=10)


def test_cursor_is_opaque_bounded_and_pages_are_stable(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="becky")
    keys = [uuid4() for _ in range(3)]
    for key in keys:
        journal.append(event(key=key))

    first = journal.list(after_cursor=None, limit=2)
    assert [item.source_event_key for item in first.events] == keys[:2]
    assert first.next_cursor is not None
    decoded = json.loads(base64.urlsafe_b64decode(first.next_cursor + "=="))
    assert set(decoded) == {"v", "s", "p"}
    second = journal.list(after_cursor=first.next_cursor, limit=2)
    assert [item.source_event_key for item in second.events] == keys[2:]
    assert second.next_cursor == journal.list(
        after_cursor=first.next_cursor, limit=2
    ).next_cursor
    assert journal.list(after_cursor=first.next_cursor, limit=2) == second

    empty = journal.list(after_cursor=second.next_cursor, limit=2)
    assert empty.events == []
    assert empty.next_cursor == second.next_cursor


def test_invalid_cursor_and_limit_fail_without_database_write(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="becky")
    with pytest.raises(ValueError):
        journal.list(after_cursor=None, limit=0)
    with pytest.raises(MutationCursorInvalid):
        journal.list(after_cursor="not-a-cursor", limit=10)
    with pytest.raises(MutationCursorInvalid):
        journal.list(after_cursor=base64.urlsafe_b64encode(b'{"v":1,"s":99,"p":"x"}').decode(), limit=10)
    assert journal.list(after_cursor=None, limit=10).events == []


def test_duplicate_event_is_idempotent_and_conflicting_payload_is_rejected(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="becky")
    key = uuid4()
    first, created = journal.append(event(key=key))
    replay, replay_created = journal.append(event(key=key))
    assert replay == first
    assert created is True
    assert replay_created is False
    with pytest.raises(MutationConflict):
        journal.append(event(key=key, status="failed"))


def test_concurrent_duplicate_append_creates_one_event(tmp_path: Path) -> None:
    path = tmp_path / "actions.sqlite3"
    key = uuid4()

    def write() -> tuple[MutationEvent, bool]:
        journal = ActionJournal(path, profile_key="becky")
        try:
            return journal.append(event(key=key))
        finally:
            journal.close()

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: write(), range(6)))
    assert sum(created for _, created in results) == 1
    journal = ActionJournal(path, profile_key="becky")
    assert len(journal.list(after_cursor=None, limit=100).events) == 1


def test_concurrent_unique_append_on_shared_connection_is_safe() -> None:
    journal = ActionJournal(":memory:", profile_key="becky")
    events = [event() for _ in range(100)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(journal.append, events))
    assert all(created for _, created in results)
    assert len(journal.list(after_cursor=None, limit=100).events) == 100


def test_start_loop_progress_allows_restart_reconciliation(tmp_path: Path) -> None:
    path = tmp_path / "actions.sqlite3"
    key = uuid4()
    journal = ActionJournal(path, profile_key="becky")
    fingerprint = "a" * 64
    assert journal.claim_start_loop(key, fingerprint) is None
    journal.update_start_loop_progress(
        key,
        {"stage": "topic_created", "thread_id": "12345"},
    )
    journal.close()

    reopened = ActionJournal(path, profile_key="becky")
    # A restart may resume a staged operation, while a duplicate request with
    # no durable stage remains fail-closed.
    assert reopened.claim_start_loop(key, fingerprint) is None
    assert reopened.get_start_loop_progress(key) == {
        "stage": "topic_created",
        "thread_id": "12345",
    }
    other = uuid4()
    assert reopened.claim_start_loop(other, fingerprint) is None
    with pytest.raises(OneShotInProgress):
        reopened.claim_start_loop(other, fingerprint)
