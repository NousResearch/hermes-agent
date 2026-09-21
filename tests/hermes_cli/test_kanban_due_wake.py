"""Time-gated cards: ``due_at``, the dispatcher's due-card waker, and the
surfaces that arm a card.

The contract these lock down: ``hermes kanban schedule <id> --due <when>`` (and
the board API) parks a card with a due time, and then the DISPATCHER TICK wakes
it -- Woke by the daemon alone, no external cron. A reserved execution band
holds the wake until the band closes, and every failure to wake is visible
instead of silent.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbdd
from hermes_cli import kanban_diagnostics as kd
from hermes_cli import kanban_due as kdue


@pytest.fixture(autouse=True)
def _writable_board(monkeypatch):
    """A dispatched worker inherits the delegated-child marker, which opens the
    board read-only. These tests write to it."""
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.delenv("HERMES_EXECUTION_WINDOWS_MAP", raising=False)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kbc.connect()
    yield c
    c.close()


def _task(conn, title="card") -> str:
    return kb.create_task(conn, title=title, assignee="smoke")


def _get(conn, tid) -> kb.Task:
    task = kb.get_task(conn, tid)
    assert task is not None, tid
    return task


def _utc_today() -> int:
    return datetime.now(timezone.utc).isoweekday()


def _map_file(tmp_path: Path, *, days, start, end, key="band") -> str:
    """A band map in the live prod shape (see
    /opt/hermes_prod/yoyodine-web-services/execution_windows.json)."""
    path = tmp_path / f"{key}.json"
    path.write_text(json.dumps({
        "timezone": "UTC",
        "version": 1,
        "windows": [{
            "key": key, "label": f"test {key}", "priority": "P0",
            "protection": "reserved", "days": list(days),
            "start": start, "end": end, "members": ["smoke"],
            "rationale": "test band",
        }],
        "external": [],
    }))
    return str(path)


def _covering_map(tmp_path: Path) -> str:
    """A reserved band that contains any instant (all days, all minutes)."""
    return _map_file(tmp_path, days=range(1, 8), start="00:00", end="23:59",
                     key="covers-now")


def _elsewhere_map(tmp_path: Path) -> str:
    """A band that contains no instant today, whatever hour the suite runs."""
    other = _utc_today() % 7 + 1
    return _map_file(tmp_path, days=[other], start="00:00", end="23:59",
                     key="other-day")


# ---------------------------------------------------------------------------
# Arming: schedule_task
# ---------------------------------------------------------------------------


def test_schedule_with_due_at_parks_the_card_and_records_the_time(conn):
    tid = _task(conn)
    due = int(time.time()) + 600

    assert kb.schedule_task(conn, tid, reason="wait for the band", due_at=due) is True

    task = _get(conn, tid)
    assert task.status == "scheduled"
    assert task.due_at == due
    assert task.due_window_policy == "defer"
    event = [e for e in kb.list_events(conn, tid) if e.kind == "scheduled"][-1]
    assert event.payload == {"reason": "wait for the band", "due_at": due,
                             "window_policy": "defer"}


def test_schedule_without_due_at_keeps_an_existing_due_time(conn):
    """``due_at`` omitted (kb.UNSET) must not silently disarm a self-waking card."""
    tid = _task(conn)
    due = int(time.time()) + 600
    kb.schedule_task(conn, tid, reason="armed", due_at=due)

    kb.schedule_task(conn, tid, reason="re-parked with a new note")

    assert _get(conn, tid).due_at == due


def test_schedule_due_none_clears_the_wake_time(conn):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) + 600)

    kb.schedule_task(conn, tid, reason="park by hand", due_at=None)

    task = _get(conn, tid)
    assert task.status == "scheduled"
    assert task.due_at is None and task.due_window_policy is None


def test_window_policy_needs_a_due_time(conn):
    """A band policy with nothing to apply it to is dead state: refuse it."""
    tid = _task(conn)
    with pytest.raises(ValueError, match="window_policy"):
        kb.schedule_task(conn, tid, reason="policy only", window_policy="ambient")


def test_window_policy_can_be_re_polished_on_an_armed_card(conn):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) + 600)

    assert kb.schedule_task(conn, tid, reason="tick around the clock",
                            window_policy="ambient") is True

    task = _get(conn, tid)
    assert task.due_window_policy == "ambient"


def test_unknown_window_policy_is_refused(conn):
    with pytest.raises(ValueError, match="window_policy"):
        kb.schedule_task(conn, _task(conn), reason="typo", due_at=1,
                         window_policy="deferr")


def test_schedule_unknown_card_reports_failure(conn):
    assert kb.schedule_task(conn, "t_missing", reason="x") is False


# ---------------------------------------------------------------------------
# The waker
# ---------------------------------------------------------------------------


def test_waker_wakes_a_due_card_and_clears_the_due_time(conn, tmp_path):
    tid = _task(conn)
    due = int(time.time()) - 60
    kb.schedule_task(conn, tid, reason="armed", due_at=due)

    out = kdue.wake_due_cards(conn, map_path=_elsewhere_map(tmp_path))

    assert out.woken == [tid] and out.problems == []
    task = _get(conn, tid)
    assert task.status == "ready"
    assert task.due_at is None and task.due_window_policy is None
    # The wake has to say who did it: a card that came back on its own is
    # otherwise indistinguishable from one a human unblocked.
    event = [e for e in kb.list_events(conn, tid) if e.kind == "unblocked"][-1]
    assert event.payload["woken_by"] == "due-card-waker"
    assert event.payload["due_at"] == due


def test_waker_leaves_a_card_that_is_not_due_yet(conn, tmp_path):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) + 3600)

    out = kdue.wake_due_cards(conn, map_path=_elsewhere_map(tmp_path))

    assert out.woken == [] and out.deferred == []
    assert _get(conn, tid).status == "scheduled"


def test_waker_defers_out_of_a_reserved_band_and_says_so(conn, tmp_path):
    """A wake hands the card to the spawn pass in the same tick, so a due time
    inside a reserved band must slide to the band's end -- visibly."""
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) - 60)

    out = kdue.wake_due_cards(conn, map_path=_covering_map(tmp_path))

    assert out.woken == [] and out.problems == []
    assert [d.task_id for d in out.deferred] == [tid]
    task = _get(conn, tid)
    assert task.status == "scheduled"
    assert task.due_at > int(time.time()), "the deferral must move the card forward"
    comments = [c.body for c in kb.list_comments(conn, tid)]
    assert any("band" in (b or "") for b in comments), comments


def test_deferral_survives_the_band_closing(conn, tmp_path):
    """The deferral parks the card at the band end, so the NEXT pass -- with the
    band gone -- wakes it rather than deferring it again."""
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) - 60)
    kdue.wake_due_cards(conn, map_path=_covering_map(tmp_path))
    deferred_to = _get(conn, tid).due_at

    # Same board, a map whose band no longer exists, and the clock past the
    # deferral: the card must come back.
    out = kdue.wake_due_cards(conn, map_path=_elsewhere_map(tmp_path),
                              now=deferred_to + 1)

    assert out.woken == [tid]
    assert _get(conn, tid).status == "ready"


def test_waker_fails_closed_when_the_map_is_unreadable(conn, tmp_path):
    tid = _task(conn)
    due = int(time.time()) - 60
    kb.schedule_task(conn, tid, reason="armed", due_at=due)

    out = kdue.wake_due_cards(conn, map_path=str(tmp_path / "nope.json"))

    assert out.woken == [] and out.problems
    task = _get(conn, tid)
    assert task.status == "scheduled" and task.due_at == due


def test_ambient_policy_wakes_inside_a_band(conn, tmp_path):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="lightweight check",
                     due_at=int(time.time()) - 60, window_policy="ambient")

    out = kdue.wake_due_cards(conn, map_path=_covering_map(tmp_path))

    assert out.woken == [tid]
    assert _get(conn, tid).status == "ready"


def test_waker_records_a_heartbeat(conn, tmp_path):
    """The heartbeat is what lets the overdue diagnostic tell a dead dispatcher
    apart from a wake that was refused."""
    assert kb.get_meta_int(conn, kb.META_DUE_WAKER_LAST_TICK) is None

    kdue.wake_due_cards(conn, map_path=_elsewhere_map(tmp_path))

    tick = kb.get_meta_int(conn, kb.META_DUE_WAKER_LAST_TICK)
    assert tick is not None and abs(tick - int(time.time())) < 60


def test_dry_run_changes_nothing(conn, tmp_path):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) - 60)

    out = kdue.wake_due_cards(conn, map_path=_elsewhere_map(tmp_path), dry_run=True)

    assert out.woken == [tid]
    assert _get(conn, tid).status == "scheduled"
    assert kb.get_meta_int(conn, kb.META_DUE_WAKER_LAST_TICK) is None


# ---------------------------------------------------------------------------
# The dispatcher tick (the acceptance path: the daemon alone wakes the card)
# ---------------------------------------------------------------------------


def test_dispatcher_tick_wakes_a_due_card(conn, tmp_path, monkeypatch):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) - 60)
    monkeypatch.setenv(kdue.ENV_MAP_PATH, _elsewhere_map(tmp_path))

    result = kbdd.dispatch_once(conn, dry_run=False, board=None, spawn_fn=None)

    assert list(result.due_woken) == [tid]
    assert result.due_deferred == [] and result.due_problems == []
    assert _get(conn, tid).status == "ready"


def test_dispatcher_tick_defers_inside_a_band(conn, tmp_path, monkeypatch):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) - 60)
    monkeypatch.setenv(kdue.ENV_MAP_PATH, _covering_map(tmp_path))

    result = kbdd.dispatch_once(conn, dry_run=False, board=None, spawn_fn=None)

    assert list(result.due_woken) == []
    assert [entry[0] for entry in result.due_deferred] == [tid]
    assert [entry[1] for entry in result.due_deferred] == ["covers-now"]
    assert [entry[2] for entry in result.due_deferred] == [_get(conn, tid).due_at]
    assert _get(conn, tid).status == "scheduled"


def test_dispatcher_tick_survives_a_broken_map(conn, tmp_path, monkeypatch):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) - 60)
    monkeypatch.setenv(kdue.ENV_MAP_PATH, str(tmp_path / "missing.json"))

    result = kbdd.dispatch_once(conn, dry_run=False, board=None, spawn_fn=None)

    assert result.due_problems and result.due_woken == []
    assert _get(conn, tid).status == "scheduled"


# ---------------------------------------------------------------------------
# Diagnostics: a card that should have woken and did not
# ---------------------------------------------------------------------------


def _overdue_diags(conn, tid, **cfg):
    task = _get(conn, tid)
    return [d for d in kd.compute_task_diagnostics(
        task, kb.list_events(conn, tid), kb.list_runs(conn, tid), config=cfg)
        if d.kind == "overdue_scheduled"]


def test_overdue_scheduled_card_is_reported(conn):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) - 7200)

    diags = _overdue_diags(conn, tid, due_waker_last_tick=int(time.time()))

    assert len(diags) == 1
    assert diags[0].severity == "warning"
    assert "2h0m" in diags[0].title
    assert diags[0].actions and "unblock" in diags[0].actions[0].payload["command"]


def test_overdue_diagnostic_names_a_dead_dispatcher(conn):
    """No heartbeat means no tick has run: the card will never come back."""
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) - 7200)

    diags = _overdue_diags(conn, tid)

    assert len(diags) == 1
    assert "dispatcher" in diags[0].detail


def test_overdue_diagnostic_escalates_when_long_stuck(conn):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) - 30 * 3600)

    diags = _overdue_diags(conn, tid)

    assert diags and diags[0].severity == "error"


def test_recently_due_card_is_not_noisy(conn):
    """The wake runs on the tick: up to one interval late is normal."""
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) - 5)

    assert _overdue_diags(conn, tid) == []


def test_woken_card_is_not_reported(conn, tmp_path):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) - 60)
    kdue.wake_due_cards(conn, map_path=_elsewhere_map(tmp_path))

    assert _overdue_diags(conn, tid) == []


def test_scheduled_card_without_a_due_time_is_not_reported(conn):
    """Parking on a human is a first-class state, not a fault."""
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="waiting on rob")

    assert _overdue_diags(conn, tid) == []


def test_diagnostic_kind_is_published(conn):
    assert "overdue_scheduled" in kd.DIAGNOSTIC_KINDS
    assert kd.DEFAULT_CONFIG["due_grace_seconds"] > 0
    assert kd.DEFAULT_CONFIG["due_stale_seconds"] > kd.DEFAULT_CONFIG["due_grace_seconds"]


# ---------------------------------------------------------------------------
# parse_due / CLI / show
# ---------------------------------------------------------------------------


def test_parse_due_accepts_iso_relative_and_epoch():
    now = 1_800_000_000
    assert kdue.parse_due("+30m", now=now) == now + 1800
    assert kdue.parse_due("+2h", now=now) == now + 7200
    assert kdue.parse_due("+1d", now=now) == now + 86400
    assert kdue.parse_due("+45s", now=now) == now + 45
    assert kdue.parse_due(str(now)) == now
    assert kdue.parse_due("2026-09-16T01:40") % 60 == 0
    assert kdue.parse_due("2026-09-16T01:40") == kdue.parse_due("2026-09-16 01:40")


def test_parse_due_rejects_nonsense():
    with pytest.raises(ValueError):
        kdue.parse_due("tomorrow")


def test_cli_schedule_due_arms_the_card(conn):
    tid = _task(conn)

    out = kc.run_slash(f"schedule {tid} boarding the window --due +2h")

    assert "due" in out.lower(), out
    task = _get(conn, tid)
    assert task.status == "scheduled"
    assert task.due_window_policy == "defer"
    assert task.due_at is not None and task.due_at > time.time()


def test_cli_schedule_due_and_clear_due_are_exclusive(conn):
    tid = _task(conn)

    out = kc.run_slash(f"schedule {tid} --due +2h --clear-due")

    assert "mutually exclusive" in out
    assert _get(conn, tid).status != "scheduled"


def test_cli_schedule_rejects_an_unparseable_due_time(conn):
    tid = _task(conn)

    out = kc.run_slash(f"schedule {tid} --due tomorrow")

    assert "due time" in out
    assert _get(conn, tid).status != "scheduled"


def test_cli_schedule_clear_due_disarms(conn):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) + 600)

    kc.run_slash(f"schedule {tid} parked by hand --clear-due")

    task = _get(conn, tid)
    assert task.due_at is None and task.status == "scheduled"


def test_cli_schedule_window_policy_needs_a_due_time(conn):
    tid = _task(conn)

    out = kc.run_slash(f"schedule {tid} no due given --window-policy ambient")

    assert "due" in out.lower()
    assert _get(conn, tid).due_at is None


def test_cli_show_renders_the_due_time(conn):
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="armed", due_at=int(time.time()) + 3600)

    out = kc.run_slash(f"show {tid}")

    assert "due" in out.lower(), out


def test_cli_show_flags_a_parked_card_with_no_wake_time(conn):
    """A scheduled card with no due time wakes only by hand -- say so."""
    tid = _task(conn)
    kb.schedule_task(conn, tid, reason="waiting on rob")

    out = kc.run_slash(f"show {tid}")

    assert "no due time" in out.lower(), out
