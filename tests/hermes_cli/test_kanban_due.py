"""Kanban due dates: input parsing, the additive migration, list windows, CLI, dispatcher.

``tasks.due_at`` is epoch seconds. Wall-clock input is read in the profile's configured
timezone, so these tests pin one (``HERMES_TIMEZONE``) instead of trusting the host clock.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

import hermes_time
from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_due as kd

def _utc(*args) -> int:
    return int(datetime(*args, tzinfo=UTC).timestamp())


# 2026-11-19 12:00:00 UTC == 2026-11-19 07:00:00 in New York (EST, UTC-5).
NOON_UTC = _utc(2026, 11, 19, 12)


@pytest.fixture
def new_york(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/New_York")
    hermes_time.reset_cache()
    yield
    hermes_time.reset_cache()


@pytest.fixture
def kanban_home(tmp_path, monkeypatch, new_york):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# --- parse_due: one grammar for every surface --------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("2026-11-20T17:00:00Z", _utc(2026, 11, 20, 17)),
    ("2026-11-20T17:00:00+01:00", _utc(2026, 11, 20, 16)),
    # No offset = the configured zone (EST), the cron one-shot rule.
    ("2026-11-20T17:00", _utc(2026, 11, 20, 22)),
    # A bare date is due by the END of that local day.
    ("2026-11-20", _utc(2026, 11, 21, 4, 59, 59)),
    # Same rule across a DST boundary: July is EDT (UTC-4).
    ("2026-07-01", _utc(2026, 7, 2, 3, 59, 59)),
    ("in 2h", NOON_UTC + 7200),
    ("in 3d", NOON_UTC + 3 * 86400),
    (str(NOON_UTC), NOON_UTC),
    (NOON_UTC, NOON_UTC),
    (None, None),
    ("", None),
    ("none", None),
    ("  clear ", None),
])
def test_parse_due_forms(new_york, raw, expected):
    assert kd.parse_due(raw, now=NOON_UTC) == expected


@pytest.mark.parametrize("raw", [
    "tomorrow", "next friday", "2026-13-01", "in soon", "in -5m", True,
    # A JS millisecond timestamp must not be stored as a date 50,000 years out.
    1795608000000,
    -1,
])
def test_parse_due_rejects_ambiguous_or_junk_input(new_york, raw):
    with pytest.raises(ValueError):
        kd.parse_due(raw, now=NOON_UTC)


def test_today_window_follows_configured_timezone(new_york):
    start, end = kd.today_window(NOON_UTC)
    assert (start, end) == (_utc(2026, 11, 19, 5), _utc(2026, 11, 20, 5))
    assert kd.due_iso(start) == "2026-11-19T00:00:00-05:00"


# --- DST: America/New_York springs forward 2026-03-08 02:00, falls back 2026-11-01 02:00 ---

@pytest.mark.parametrize("raw", ["2026-03-08T02:00", "2026-03-08T02:30", "2026-03-08T02:59:59"])
def test_naive_time_skipped_by_spring_forward_is_refused(new_york, raw):
    with pytest.raises(ValueError, match="does not exist"):
        kd.parse_due(raw)


@pytest.mark.parametrize("raw", ["2026-11-01T01:00", "2026-11-01T01:30", "2026-11-01T01:59:59"])
def test_naive_time_repeated_by_fall_back_needs_an_offset(new_york, raw):
    with pytest.raises(ValueError, match="occurs twice"):
        kd.parse_due(raw)


@pytest.mark.parametrize("raw, expected", [
    # The two readings of the repeated hour, made explicit.
    ("2026-11-01T01:30-04:00", _utc(2026, 11, 1, 5, 30)),
    ("2026-11-01T01:30-05:00", _utc(2026, 11, 1, 6, 30)),
    # Unambiguous neighbours of both transitions read as their single local instant.
    ("2026-03-08T01:59", _utc(2026, 3, 8, 6, 59)),
    ("2026-03-08T03:00", _utc(2026, 3, 8, 7)),
    ("2026-11-01T00:59", _utc(2026, 11, 1, 4, 59)),
    ("2026-11-01T02:00", _utc(2026, 11, 1, 7)),
    # A bare date is still due by its last local second on both transition days.
    ("2026-03-08", _utc(2026, 3, 9, 3, 59, 59)),
    ("2026-11-01", _utc(2026, 11, 2, 4, 59, 59)),
])
def test_unambiguous_times_around_dst_transitions(new_york, raw, expected):
    assert kd.parse_due(raw) == expected


@pytest.mark.parametrize("now, start, end", [
    # Spring forward: local midnight is EST, the next one EDT -> a 23-hour day.
    (_utc(2026, 3, 8, 12), _utc(2026, 3, 8, 5), _utc(2026, 3, 9, 4)),
    # Fall back: local midnight is EDT, the next one EST -> a 25-hour day.
    (_utc(2026, 11, 1, 12), _utc(2026, 11, 1, 4), _utc(2026, 11, 2, 5)),
    # The first and last instants of each transition day map to that same day.
    (_utc(2026, 3, 8, 5), _utc(2026, 3, 8, 5), _utc(2026, 3, 9, 4)),
    (_utc(2026, 3, 9, 4) - 1, _utc(2026, 3, 8, 5), _utc(2026, 3, 9, 4)),
    (_utc(2026, 11, 1, 4), _utc(2026, 11, 1, 4), _utc(2026, 11, 2, 5)),
    (_utc(2026, 11, 2, 5) - 1, _utc(2026, 11, 1, 4), _utc(2026, 11, 2, 5)),
])
def test_today_window_across_dst_boundaries(new_york, now, start, end):
    assert kd.today_window(now) == (start, end)
    # The "today" list filter and the dashboard badge agree at the edges.
    assert kd.due_fields(start, "done", now=now)["due_state"] == "today"
    assert kd.due_fields(end - 1, "done", now=now)["due_state"] == "today"
    assert kd.due_fields(start - 1, "done", now=now)["due_state"] == "later"
    assert kd.due_fields(end, "done", now=now)["due_state"] == "later"


@pytest.mark.parametrize("raw", ["infh", "-infm", "nans", "nand", "1e400h"])
def test_non_finite_durations_are_value_errors(raw):
    with pytest.raises(ValueError, match="finite"):
        kd.parse_duration(raw)
    with pytest.raises(ValueError):
        kd.parse_due(f"in {raw}", now=NOON_UTC)


# --- Migration ----------------------------------------------------------------

def test_legacy_board_gains_nullable_due_at_without_losing_rows(tmp_path, monkeypatch, new_york):
    """A board created before the column existed opens cleanly: the column and its index
    are added, existing rows read as "no due date", and they become editable."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db_path = home / "kanban.db"
    # Build a current board, then strip due_at to reproduce the pre-feature schema.
    kb.init_db(db_path)
    with kbc.connect_closing(db_path) as conn:
        legacy_id = kb.create_task(conn, title="pre-existing", assignee="alice", priority=3)
    raw = sqlite3.connect(db_path)
    raw.execute("DROP INDEX IF EXISTS idx_tasks_due_at")
    raw.execute("ALTER TABLE tasks DROP COLUMN due_at")
    raw.commit()
    assert "due_at" not in {r[1] for r in raw.execute("PRAGMA table_info(tasks)")}
    raw.close()

    kb.init_db(db_path)  # the same path every connect() takes on first open

    raw = sqlite3.connect(db_path)
    cols = {r[1]: r for r in raw.execute("PRAGMA table_info(tasks)")}
    indexes = {r[1] for r in raw.execute("PRAGMA index_list(tasks)")}
    raw.close()
    assert cols["due_at"][2] == "INTEGER" and cols["due_at"][3] == 0  # nullable, no default
    assert "idx_tasks_due_at" in indexes
    with kbc.connect_closing(db_path) as conn:
        task = kb.get_task(conn, legacy_id)
        assert (task.title, task.priority, task.due_at) == ("pre-existing", 3, None)
        assert kb.edit_task(conn, legacy_id, due_at="2026-11-20")
        assert kb.get_task(conn, legacy_id).due_at == _utc(2026, 11, 21, 4, 59, 59)
    kb.init_db(db_path)  # idempotent re-run


# --- DB layer: storage, edit, list windows ------------------------------------

def test_create_edit_and_clear_due_date_records_events(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="file taxes", due_at="2026-11-20T17:00:00Z")
        assert kb.get_task(conn, tid).due_at == _utc(2026, 11, 20, 17)
        created = next(e for e in kb.list_events(conn, tid) if e.kind == "created")
        assert created.payload["due_at"] == _utc(2026, 11, 20, 17)

        assert kb.edit_task(conn, tid, due_at="in 1d")
        moved = kb.get_task(conn, tid).due_at
        assert moved > time.time()
        assert kb.edit_task(conn, tid, clear_due=True)
        assert kb.get_task(conn, tid).due_at is None
        edits = [e.payload for e in kb.list_events(conn, tid) if e.kind == "edited"]
        assert edits == [{"fields": ["due_at"], "due_at": moved}, {"fields": ["due_at"], "due_at": None}]


def test_invalid_due_date_writes_nothing(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", due_at=NOON_UTC)
        with pytest.raises(ValueError):
            kb.edit_task(conn, tid, title="renamed", due_at="next tuesday")
        task = kb.get_task(conn, tid)
        assert (task.title, task.due_at) == ("t", NOON_UTC)
        with pytest.raises(ValueError):
            kb.create_task(conn, title="bad", due_at="soonish")
        assert [t.title for t in kb.list_tasks(conn)] == ["t"]


def test_list_due_windows_and_ordering(kanban_home):
    now = time.time()
    start, end = kd.today_window()
    with kbc.connect_closing() as conn:
        overdue = kb.create_task(conn, title="overdue", due_at=int(now) - 3 * 86400)
        done_late = kb.create_task(conn, title="done late", due_at=int(now) - 86400)
        kb.complete_task(conn, done_late, result="ok", force=True)
        today = kb.create_task(conn, title="today", due_at=end - 1)
        later = kb.create_task(conn, title="later", due_at=end + 5 * 86400)
        undated = kb.create_task(conn, title="undated", priority=9)

        def ids(**kw):
            return [t.id for t in kb.list_tasks(conn, **kw)]

        # Overdue = past due and not finished; a completed task is never overdue.
        assert ids(due="overdue") == [overdue]
        assert ids(due="today") == [today]
        assert set(ids(due="any")) == {overdue, done_late, today, later}
        assert ids(due="none") == [undated]
        # Soonest first, undated last; the reverse keeps undated last too.
        assert ids(order_by="due") == [overdue, done_late, today, later, undated]
        assert ids(order_by="due-desc") == [later, today, done_late, overdue, undated]
        # The default order (priority, then age) is untouched by due dates.
        assert ids()[0] == undated
        assert ids(due="today", order_by="due", status="ready") == [today]
        with pytest.raises(ValueError):
            kb.list_tasks(conn, due="someday")
    assert start <= end - 1 < end


# --- CLI ----------------------------------------------------------------------

def test_cli_create_edit_list_show_round_trip(kanban_home):
    created = json.loads(kc.run_slash("create 'renew passport' --due 2026-11-20 --json"))
    tid = created["id"]
    assert created["due_at"] == _utc(2026, 11, 21, 4, 59, 59)
    assert created["due_at_iso"] == "2026-11-20T23:59:59-05:00"

    kc.run_slash(f"edit {tid} --due 2026-11-25T09:30")
    shown = json.loads(kc.run_slash(f"show {tid} --json"))["task"]
    assert shown["due_at_iso"] == "2026-11-25T09:30:00-05:00"
    assert "due:" in kc.run_slash(f"show {tid}")

    kc.run_slash("create 'no deadline'")
    listed = json.loads(kc.run_slash("list --sort due --json"))
    assert [row["title"] for row in listed] == ["renew passport", "no deadline"]
    assert [row["title"] for row in json.loads(kc.run_slash("list --due none --json"))] == ["no deadline"]

    kc.run_slash(f"edit {tid} --due none")
    assert json.loads(kc.run_slash(f"show {tid} --json"))["task"]["due_at"] is None


def test_cli_rejects_unparseable_due_without_writing(kanban_home):
    out = kc.run_slash("create errand --due 'next week'")
    assert "--due" in out and "invalid due date" in out
    with kbc.connect_closing() as conn:
        assert kb.list_tasks(conn) == []
        tid = kb.create_task(conn, title="keep", due_at=NOON_UTC)
    assert "invalid due date" in kc.run_slash(f"edit {tid} --due someday")
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).due_at == NOON_UTC


# --- Dispatcher: due dates never cause execution ------------------------------

def test_past_due_date_never_spawns_a_non_profile_assignee(kanban_home, monkeypatch):
    """A human / control-plane assignee stays parked however overdue the card is."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "worker")
    spawned: list[str] = []

    def fake_spawn(task, workspace, **_kw):
        spawned.append(task.id)
        return 4242

    with kbc.connect_closing() as conn:
        human = kb.create_task(conn, title="call the bank", assignee="me", due_at=int(time.time()) - 86400)
        unassigned = kb.create_task(conn, title="no owner", due_at=int(time.time()) - 86400)
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn, dry_run=False)
        assert human in res.skipped_nonspawnable
        assert unassigned in res.skipped_unassigned
        assert spawned == []
        assert kb.get_task(conn, human).status == "ready"


def test_due_date_does_not_change_profile_dispatch(kanban_home, monkeypatch):
    """Dispatch of a real profile's cards is identical with and without due dates:
    the dispatcher's own order (priority, age) decides, never the deadline."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "worker")
    order: list[str] = []

    def fake_spawn(task, workspace, **_kw):
        order.append(task.title)
        return 4242

    with kbc.connect_closing() as conn:
        kb.create_task(conn, title="old, no deadline", assignee="worker")
        kb.create_task(conn, title="new, overdue", assignee="worker", due_at=int(time.time()) - 86400)
        kbd.dispatch_once(conn, spawn_fn=fake_spawn, dry_run=False, max_spawn=1)
    assert order == ["old, no deadline"]


def test_overdue_scheduled_card_stays_parked(kanban_home, monkeypatch):
    """A card parked in Scheduled waits for an unblock, not for its deadline: an overdue
    due date on a spawnable profile's card never releases it."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "worker")
    spawned: list[str] = []

    def fake_spawn(task, workspace, **_kw):
        spawned.append(task.id)
        return 4242

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="renew lease", assignee="worker", due_at=int(time.time()) - 86400)
        assert kb.schedule_task(conn, tid, reason="after the 1st")
        for _ in range(2):
            kbd.dispatch_once(conn, spawn_fn=fake_spawn, dry_run=False)
        task = kb.get_task(conn, tid)
        assert [t.id for t in kb.list_tasks(conn, due="overdue")] == [tid]  # still reported overdue
    assert spawned == []
    assert task.status == "scheduled"


# --- Overflow and float input: always a ValueError (400 / exit 2), never Overflow/500 -----

@pytest.mark.parametrize("raw", ["1e308d", "1e305d"])
def test_duration_overflowing_after_scaling_is_a_value_error(raw):
    """``1e308`` and ``1e305`` are finite floats; times 86400 they overflow to inf."""
    with pytest.raises(ValueError, match="finite"):
        kd.parse_duration(raw)
    with pytest.raises(ValueError):
        kd.parse_due(f"in {raw}", now=NOON_UTC)


def test_huge_integer_duration_is_out_of_range_not_overflow():
    with pytest.raises(ValueError, match="out of range"):
        kd.parse_due("in " + "9" * 400, now=NOON_UTC)


@pytest.mark.parametrize("raw", [1.5, float(NOON_UTC), float("inf"), float("nan")])
def test_float_due_values_are_rejected_not_truncated(raw):
    with pytest.raises(ValueError, match="whole epoch seconds"):
        kd.parse_due(raw)


@pytest.mark.parametrize("args", ["--due 'in 1e308d'", "--due 'in 1e305d'", "--max-runtime 1e308d"])
def test_cli_overflowing_input_is_a_clean_error(kanban_home, args):
    out = kc.run_slash(f"create errand {args}")
    assert "malformed duration" in out and "Traceback" not in out
    with kbc.connect_closing() as conn:
        assert kb.list_tasks(conn) == []
