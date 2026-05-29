import json
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli import calendar_db
from tools import calendar_tools


@pytest.fixture
def sqlite_calendar(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_CALENDAR_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_POSTGRES_DSN", raising=False)
    return hermes_home


def test_parse_dt_requires_timezone():
    with pytest.raises(ValueError, match="timezone"):
        calendar_db._parse_dt("2026-05-27T10:00:00", field="scheduled_at")


def test_parse_dt_accepts_zulu():
    dt = calendar_db._parse_dt("2026-05-27T10:00:00Z", field="scheduled_at")

    assert dt == datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc)


def test_normalize_recurrence_accepts_simple_values():
    assert calendar_db._normalize_recurrence("daily") == "daily"
    assert calendar_db._normalize_recurrence(" WEEKLY ") == "weekly"
    assert calendar_db._normalize_recurrence(None) is None


def test_normalize_recurrence_rejects_rrule_for_v1():
    with pytest.raises(ValueError, match="daily, weekly, monthly"):
        calendar_db._normalize_recurrence("FREQ=DAILY")


def test_advance_monthly_clamps_end_of_month():
    dt = datetime(2026, 1, 31, 8, 0, tzinfo=timezone.utc)

    assert calendar_db._advance(dt, "monthly") == datetime(2026, 2, 28, 8, 0, tzinfo=timezone.utc)


def test_calendar_add_creates_sqlite_db_and_parses_json_fields(sqlite_calendar):
    result = json.loads(
        calendar_tools._handle_add(
            {
                "title": "Wake",
                "scheduled_at": "2026-05-27T10:00:00Z",
                "tags": ["memoire", "judy"],
                "context": {"source": "test"},
            }
        )
    )

    assert result["ok"] is True
    event = result["result"]
    assert event["context"] == {"source": "test"}
    assert event["tags"] == ["memoire", "judy"]
    assert (sqlite_calendar / "data" / "judy_calendar.db").exists()


def test_calendar_list_sqlite_supports_status_date_and_tag_filters(sqlite_calendar):
    calendar_db.add_event(
        title="First",
        scheduled_at="2026-05-27T10:00:00Z",
        tags=["memoire"],
        context={"rank": 1},
    )
    second = calendar_db.add_event(
        title="Second",
        scheduled_at="2026-05-28T10:00:00Z",
        tags=["health"],
    )
    calendar_db.cancel_event(second["id"])

    pending = calendar_db.list_events(
        status="pending",
        from_="2026-05-27T00:00:00Z",
        to="2026-05-27T23:59:59Z",
        tags=["memoire"],
    )

    assert [event["title"] for event in pending] == ["First"]
    assert pending[0]["context"] == {"rank": 1}
    assert calendar_db.list_events(status="cancelled", tags=["health"])[0]["title"] == "Second"


def test_calendar_done_cancel_and_update_sqlite(sqlite_calendar):
    event = calendar_db.add_event(title="Draft", scheduled_at="2026-05-27T10:00:00Z")

    updated = calendar_db.update_event(
        event["id"],
        title="Updated",
        scheduled_at="2026-05-27T11:00:00Z",
        description="Details",
        tags=["focus"],
        context={"changed": True},
    )
    assert updated["title"] == "Updated"
    assert updated["scheduled_at"] == "2026-05-27T11:00:00Z"
    assert updated["tags"] == ["focus"]
    assert updated["context"] == {"changed": True}

    done = calendar_db.mark_done(event["id"], notes="ok", session_id="session-1")
    assert done["status"] == "done"
    assert done["notes"] == "ok"
    assert done["session_id"] == "session-1"

    cancelled = calendar_db.cancel_event(event["id"])
    assert cancelled["status"] == "cancelled"


def test_claim_due_events_marks_pending_rows_as_firing(sqlite_calendar):
    due = calendar_db.add_event(title="Due", scheduled_at="2026-05-27T10:00:00Z")
    calendar_db.add_event(title="Future", scheduled_at="2026-05-27T12:00:00Z")

    claimed = calendar_db.claim_due_events(now="2026-05-27T10:30:00Z", limit=5)

    assert [event["id"] for event in claimed] == [due["id"]]
    assert claimed[0]["status"] == "firing"
    assert calendar_db.get_event(due["id"])["status"] == "firing"


def test_requeue_stale_firing_returns_events_to_pending(sqlite_calendar):
    event = calendar_db.add_event(title="Due", scheduled_at="2026-05-27T10:00:00Z")
    calendar_db.claim_due_events(now="2026-05-27T10:30:00Z")

    with calendar_db._connect() as conn:
        conn.execute(
            "UPDATE judy_calendar SET fired_at = ? WHERE id = ?",
            ((datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace("+00:00", "Z"), event["id"]),
        )

    assert calendar_db.requeue_stale_firing(older_than_seconds=60) == 1
    requeued = calendar_db.get_event(event["id"])
    assert requeued["status"] == "pending"
    assert requeued["fired_at"] is None


def test_recurring_mark_done_advances_scheduled_at_and_stays_pending(sqlite_calendar):
    event = calendar_db.add_event(
        title="Daily",
        scheduled_at="2026-05-27T10:00:00Z",
        recurrence="daily",
    )

    updated = calendar_db.mark_done(event["id"])

    assert updated["status"] == "pending"
    assert updated["scheduled_at"] > datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    assert updated["completed_at"] is not None
