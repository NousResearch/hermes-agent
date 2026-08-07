# SPDX-License-Identifier: Apache-2.0
"""Tests for the reminders domain module.

Focus on invariants, not on any concrete store implementation:

* calculate_next_due_at is a pure function — daily/weekly/monthly all
  produce the strictly-next matching wall-clock instant.
* Same-minute one-off for the same user merges (BaiLongma parity).
* Past-time one-offs are rejected.
* Recurrence config errors surface as ValueError.
* emit_event hook receives payloads for create / merged / cancelled;
  a broken hook does NOT lose state changes.
* Manager time is injectable via now_fn — no wall-clock dependence.
* advance_recurring updates due_at using the SAME recurrence config
  the row was created with (not whatever the caller happens to send).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import pytest

from agent.reminders import (
    Reminder,
    ReminderCreateRequest,
    ReminderManager,
    calculate_next_due_at,
    format_reminder_row,
)


# ── In-memory store ────────────────────────────────────────────────


class InMemoryReminderStore:
    """A dict-backed reference implementation of ReminderStore."""

    def __init__(self) -> None:
        self._rows: dict[int, Reminder] = {}
        self._next_id = 1

    # Write-side ------------------------------------------------------

    def create(self, reminder: Reminder) -> Reminder:
        new = replace(reminder, id=self._next_id)
        self._rows[self._next_id] = new
        self._next_id += 1
        return new

    def find_mergeable_one_off(
        self, *, user_id: str, minute_key: str
    ) -> Optional[Reminder]:
        for r in sorted(self._rows.values(), key=lambda x: x.id):
            if (
                r.status == "pending"
                and r.recurrence_type is None
                and r.user_id == user_id
                and r.due_at[:16] == minute_key
            ):
                return r
        return None

    def append_task(
        self, *, reminder_id: int, additional_task: str
    ) -> Optional[Reminder]:
        row = self._rows.get(reminder_id)
        if row is None or row.status != "pending":
            return None
        merged = replace(row, task=f"{row.task}; {additional_task}")
        self._rows[reminder_id] = merged
        return merged

    def get_by_id(self, reminder_id: int) -> Optional[Reminder]:
        return self._rows.get(reminder_id)

    def cancel(
        self, *, reminder_id: int, cancelled_at: str
    ) -> Optional[Reminder]:
        row = self._rows.get(reminder_id)
        if row is None or row.status != "pending":
            return None
        updated = replace(row, status="cancelled", cancelled_at=cancelled_at)
        self._rows[reminder_id] = updated
        return updated

    def mark_fired(
        self, *, reminder_id: int, fired_at: str
    ) -> Optional[Reminder]:
        row = self._rows.get(reminder_id)
        if row is None or row.status != "pending":
            return None
        updated = replace(row, status="fired", fired_at=fired_at)
        self._rows[reminder_id] = updated
        return updated

    def advance_due_at(
        self, *, reminder_id: int, next_due_at: str
    ) -> Optional[Reminder]:
        row = self._rows.get(reminder_id)
        if row is None or row.status != "pending":
            return None
        updated = replace(row, due_at=next_due_at)
        self._rows[reminder_id] = updated
        return updated

    # Read-side -------------------------------------------------------

    def list_pending(self, limit: int = 50) -> Sequence[Reminder]:
        rows = [r for r in self._rows.values() if r.status == "pending"]
        return sorted(rows, key=lambda r: (r.due_at, r.id))[:limit]

    def list_due(self, *, now_iso: str, limit: int = 20) -> Sequence[Reminder]:
        rows = [
            r
            for r in self._rows.values()
            if r.status == "pending" and r.due_at <= now_iso
        ]
        return sorted(rows, key=lambda r: (r.due_at, r.id))[:limit]

    def next_pending(self) -> Optional[Reminder]:
        pending = self.list_pending(limit=1)
        return pending[0] if pending else None


def _fixed_now(dt: datetime):
    return lambda: dt


# ── Recurrence engine ──────────────────────────────────────────────


def test_calculate_next_due_at_daily_wraps_to_next_day() -> None:
    now = datetime(2026, 4, 21, 12, 0, 0, tzinfo=timezone.utc)
    result = calculate_next_due_at("daily", {"time": "09:00"}, from_dt=now)
    assert result == datetime(2026, 4, 22, 9, 0, tzinfo=timezone.utc)


def test_calculate_next_due_at_daily_uses_today_if_time_still_ahead() -> None:
    now = datetime(2026, 4, 21, 5, 0, 0, tzinfo=timezone.utc)
    result = calculate_next_due_at("daily", {"time": "09:00"}, from_dt=now)
    assert result == datetime(2026, 4, 21, 9, 0, tzinfo=timezone.utc)


def test_calculate_next_due_at_weekly_uses_sunday_zero_convention() -> None:
    """BaiLongma follows JS getDay(): 0=Sunday. 2026-04-21 is a Tuesday
    (JS getDay=2). Asking for weekday=2 with a time still in the future
    should land on the same day."""
    now = datetime(2026, 4, 21, 7, 0, 0, tzinfo=timezone.utc)
    result = calculate_next_due_at(
        "weekly", {"time": "09:00", "weekday": 2}, from_dt=now
    )
    assert result == datetime(2026, 4, 21, 9, 0, tzinfo=timezone.utc)


def test_calculate_next_due_at_weekly_advances_seven_days_when_slot_passed() -> None:
    """Same weekday, but the slot has already passed → next week."""
    now = datetime(2026, 4, 21, 12, 0, 0, tzinfo=timezone.utc)  # Tue
    result = calculate_next_due_at(
        "weekly", {"time": "09:00", "weekday": 2}, from_dt=now
    )
    assert result == datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)


def test_calculate_next_due_at_weekly_rejects_invalid_weekday() -> None:
    now = datetime(2026, 4, 21, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="weekday must be"):
        calculate_next_due_at(
            "weekly", {"time": "09:00", "weekday": 7}, from_dt=now
        )


def test_calculate_next_due_at_monthly_skips_short_months() -> None:
    """Asking for day 31 in a January that's already past its 31st
    should return the next month with 31 days (March)."""
    now = datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)
    result = calculate_next_due_at(
        "monthly", {"time": "09:00", "day_of_month": 31}, from_dt=now
    )
    assert result == datetime(2026, 3, 31, 9, 0, tzinfo=timezone.utc)


def test_calculate_next_due_at_monthly_rejects_invalid_day() -> None:
    now = datetime(2026, 4, 21, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="day_of_month must be"):
        calculate_next_due_at(
            "monthly", {"time": "09:00", "day_of_month": 32}, from_dt=now
        )


def test_calculate_next_due_at_rejects_bad_time_format() -> None:
    now = datetime(2026, 4, 21, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="HH:MM"):
        calculate_next_due_at(
            "daily", {"time": "9-00"}, from_dt=now
        )


def test_calculate_next_due_at_rejects_unknown_kind() -> None:
    now = datetime(2026, 4, 21, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="Unknown recurrence kind"):
        calculate_next_due_at("hourly", {"time": "09:00"}, from_dt=now)


# ── Manager.create — one-off ───────────────────────────────────────


def _mk_manager(now: datetime, emit=None):
    return ReminderManager(
        store=InMemoryReminderStore(), now_fn=_fixed_now(now), emit_event=emit
    )


def test_create_one_off_stores_row() -> None:
    now = datetime(2026, 4, 21, 6, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now)
    created = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="drink water", due_at="2026-04-21T08:00:00+00:00"
        )
    )
    assert created.id == 1
    assert created.user_id == "u1"
    assert created.task == "drink water"
    assert created.due_at.startswith("2026-04-21T08:00")
    assert created.status == "pending"
    assert created.recurrence_type is None


def test_create_one_off_rejects_empty_task() -> None:
    mgr = _mk_manager(datetime(2026, 4, 21, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="task must be"):
        mgr.create(
            ReminderCreateRequest(
                user_id="u1", task="   ", due_at="2026-04-21T08:00Z"
            )
        )


def test_create_one_off_rejects_past_due_at() -> None:
    now = datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now)
    with pytest.raises(ValueError, match="strictly in the future"):
        mgr.create(
            ReminderCreateRequest(
                user_id="u1",
                task="ping",
                due_at="2026-04-21T06:00:00+00:00",
            )
        )


def test_create_one_off_rejects_missing_due_at() -> None:
    mgr = _mk_manager(datetime(2026, 4, 21, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="due_at is required"):
        mgr.create(ReminderCreateRequest(user_id="u1", task="ping"))


def test_create_one_off_rejects_malformed_due_at() -> None:
    mgr = _mk_manager(datetime(2026, 4, 21, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="valid ISO 8601"):
        mgr.create(
            ReminderCreateRequest(
                user_id="u1", task="ping", due_at="not-a-date"
            )
        )


def test_create_one_off_merges_same_minute_same_user() -> None:
    now = datetime(2026, 4, 21, 6, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now)
    first = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="alpha", due_at="2026-04-21T08:00:00Z"
        )
    )
    second = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="beta", due_at="2026-04-21T08:00:59Z"
        )
    )
    assert second.id == first.id
    assert second.task == "alpha; beta"
    # Only one row stored.
    assert len(mgr.list_pending()) == 1


def test_create_one_off_different_user_does_not_merge() -> None:
    now = datetime(2026, 4, 21, 6, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now)
    a = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="alpha", due_at="2026-04-21T08:00:00Z"
        )
    )
    b = mgr.create(
        ReminderCreateRequest(
            user_id="u2", task="beta", due_at="2026-04-21T08:00:00Z"
        )
    )
    assert a.id != b.id
    assert len(mgr.list_pending()) == 2


def test_create_one_off_different_minute_does_not_merge() -> None:
    now = datetime(2026, 4, 21, 6, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now)
    a = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="alpha", due_at="2026-04-21T08:00:00Z"
        )
    )
    b = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="beta", due_at="2026-04-21T08:01:00Z"
        )
    )
    assert a.id != b.id


# ── Manager.create — recurrence ────────────────────────────────────


def test_create_daily_computes_next_slot() -> None:
    now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now)
    row = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="wake up", kind="daily", time="09:00"
        )
    )
    assert row.recurrence_type == "daily"
    assert row.recurrence_config == {"time": "09:00"}
    assert row.due_at.startswith("2026-04-22T09:00")


def test_create_weekly_requires_weekday() -> None:
    mgr = _mk_manager(datetime(2026, 4, 21, tzinfo=timezone.utc))
    with pytest.raises(ValueError):
        mgr.create(
            ReminderCreateRequest(
                user_id="u1", task="review", kind="weekly", time="10:00"
            )
        )


def test_create_monthly_stores_config() -> None:
    now = datetime(2026, 4, 21, tzinfo=timezone.utc)
    mgr = _mk_manager(now)
    row = mgr.create(
        ReminderCreateRequest(
            user_id="u1",
            task="rent",
            kind="monthly",
            time="09:00",
            day_of_month=25,
        )
    )
    assert row.recurrence_type == "monthly"
    assert row.recurrence_config == {"time": "09:00", "day_of_month": 25}


def test_create_rejects_unknown_kind() -> None:
    mgr = _mk_manager(datetime(2026, 4, 21, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="Unknown kind"):
        mgr.create(
            ReminderCreateRequest(
                user_id="u1", task="x", kind="hourly", time="09:00"
            )
        )


# ── cancel / list / mark_fired / advance_recurring ─────────────────


def test_cancel_marks_row_cancelled() -> None:
    now = datetime(2026, 4, 21, 6, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now)
    created = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="ping", due_at="2026-04-21T08:00Z"
        )
    )
    cancelled = mgr.cancel(created.id)
    assert cancelled.status == "cancelled"
    assert cancelled.cancelled_at is not None
    assert mgr.list_pending() == []


def test_cancel_rejects_unknown_id() -> None:
    mgr = _mk_manager(datetime(2026, 4, 21, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="not found"):
        mgr.cancel(999)


def test_cancel_rejects_already_fired_row() -> None:
    now = datetime(2026, 4, 21, 6, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now)
    created = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="ping", due_at="2026-04-21T08:00Z"
        )
    )
    mgr.mark_fired(created.id)
    with pytest.raises(ValueError, match="fired"):
        mgr.cancel(created.id)


def test_due_now_returns_only_past_due_pending() -> None:
    now = datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now)
    # One in the past (past due), one in the future (not due)
    mgr._now_fn = _fixed_now(  # bypass "future" check for setup
        datetime(2026, 4, 21, 5, 0, tzinfo=timezone.utc)
    )
    r1 = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="past", due_at="2026-04-21T06:00Z"
        )
    )
    r2 = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="future", due_at="2026-04-21T10:00Z"
        )
    )
    mgr._now_fn = _fixed_now(now)  # restore
    due = mgr.due_now()
    assert [r.id for r in due] == [r1.id]
    # r2 is still pending, just not due yet.
    assert r2.id in {r.id for r in mgr.list_pending()}


def test_mark_fired_rejects_unknown_or_non_pending() -> None:
    mgr = _mk_manager(datetime(2026, 4, 21, tzinfo=timezone.utc))
    with pytest.raises(RuntimeError):
        mgr.mark_fired(999)


def test_advance_recurring_reschedules_daily() -> None:
    now = datetime(2026, 4, 21, 5, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now)
    row = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="wake", kind="daily", time="09:00"
        )
    )
    # first fire: due_at = 2026-04-21 09:00 (today's future)
    assert row.due_at.startswith("2026-04-21T09:00")

    # Simulate firing at 09:05 and advancing.
    mgr._now_fn = _fixed_now(
        datetime(2026, 4, 21, 9, 5, tzinfo=timezone.utc)
    )
    advanced = mgr.advance_recurring(row.id)
    assert advanced.due_at.startswith("2026-04-22T09:00")


def test_advance_recurring_rejects_one_off() -> None:
    now = datetime(2026, 4, 21, 5, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now)
    row = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="ping", due_at="2026-04-21T08:00Z"
        )
    )
    with pytest.raises(ValueError, match="not recurring"):
        mgr.advance_recurring(row.id)


# ── emit_event ─────────────────────────────────────────────────────


def test_emit_event_fires_on_create_and_cancel() -> None:
    hits: list[tuple[str, dict]] = []
    now = datetime(2026, 4, 21, 6, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now, emit=lambda n, p: hits.append((n, p)))
    row = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="alpha", due_at="2026-04-21T08:00Z"
        )
    )
    mgr.cancel(row.id)
    assert [h[0] for h in hits] == ["reminder_created", "reminder_cancelled"]
    assert hits[0][1]["task"] == "alpha"
    assert hits[1][1]["id"] == row.id


def test_emit_event_fires_reminder_merged_on_same_minute_second_create() -> None:
    hits: list[tuple[str, dict]] = []
    now = datetime(2026, 4, 21, 6, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now, emit=lambda n, p: hits.append((n, p)))
    mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="alpha", due_at="2026-04-21T08:00Z"
        )
    )
    mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="beta", due_at="2026-04-21T08:00Z"
        )
    )
    event_names = [h[0] for h in hits]
    assert event_names == ["reminder_created", "reminder_merged"]
    assert hits[1][1]["task"] == "alpha; beta"


def test_emit_event_hook_error_does_not_lose_state_change() -> None:
    def broken(name, payload):  # noqa: ARG001
        raise RuntimeError("hook down")

    now = datetime(2026, 4, 21, 6, 0, tzinfo=timezone.utc)
    mgr = _mk_manager(now, emit=broken)
    row = mgr.create(
        ReminderCreateRequest(
            user_id="u1", task="ping", due_at="2026-04-21T08:00Z"
        )
    )
    assert row.id == 1
    assert mgr.get(row.id) is not None


# ── formatter ─────────────────────────────────────────────────────


def test_format_reminder_row_one_off() -> None:
    row = Reminder(
        id=42,
        user_id="u1",
        due_at="2026-04-21T08:00:00Z",
        task="ping",
    )
    text = format_reminder_row(row)
    assert text.startswith("#42 [once]")
    assert "2026-04-21T08:00:00Z" in text
    assert "u1" in text
    assert "ping" in text


def test_format_reminder_row_weekly_uses_zh_weekday() -> None:
    row = Reminder(
        id=7,
        user_id="u1",
        due_at="2026-04-27T09:00:00Z",
        task="review",
        recurrence_type="weekly",
        recurrence_config={"time": "09:00", "weekday": 1},
    )
    text = format_reminder_row(row)
    assert "[weekly]" in text
    assert "每周一" in text
    assert "09:00" in text
