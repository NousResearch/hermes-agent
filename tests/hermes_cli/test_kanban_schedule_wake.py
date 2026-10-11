"""Timed wake for ``scheduled`` cards + the scheduled-without-wake diagnostic.

A ``scheduled`` card had no timed exit: nothing in the dispatcher wakes it, and
``promote`` (the verb operators reach for) refuses ``scheduled`` without naming
``unblock``. A card parked on a date therefore sat there until someone
remembered it. Pinned here:

1. ``schedule --at/--now`` stamps a timed wake; the dispatcher's next tick
   returns the card to ``ready`` (``todo`` while parents are open) and records
   ``schedule_elapsed``.
2. ``scheduled_without_wake`` fires on a NULL-gated scheduled card parked
   >= 24h; a 1h-old one stays silent; a card WITH a wake time never fires.
3. ``promote`` on a scheduled card names the exit verbs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_diagnostics as kd

HOUR = 3600


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _args(**kw) -> argparse.Namespace:
    base = {"reason": [], "ids": None, "at": None, "now": False}
    base.update(kw)
    return argparse.Namespace(**base)


def _scheduled(conn, title="gated") -> str:
    tid = kb.create_task(conn, title=title, assignee="worker")
    assert kb.schedule_task(conn, tid, reason="wait for the readback window")
    task = kb.get_task(conn, tid)
    assert task.status == "scheduled" and task.next_eligible_at is None
    return tid


def _kinds(conn, tid) -> list[str]:
    return [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (tid,))]


def _diag(task, events, now):
    return [d for d in kd.compute_task_diagnostics(task, events, [], now=now)
            if d.kind == "scheduled_without_wake"]


# --- diagnostic ---------------------------------------------------------------


def test_null_gated_scheduled_25h_old_fires():
    now = 1_000_000
    task = {"id": "t_x", "status": "scheduled", "next_eligible_at": None, "created_at": now - 30 * HOUR}
    diags = _diag(task, [{"kind": "scheduled", "created_at": now - 25 * HOUR, "payload": None}], now)
    assert len(diags) == 1
    assert diags[0].data["age_hours"] == 25.0
    commands = [a.payload["command"] for a in diags[0].actions]
    assert "hermes kanban unblock t_x" in commands


def test_null_gated_scheduled_1h_old_is_silent():
    now = 1_000_000
    task = {"id": "t_x", "status": "scheduled", "next_eligible_at": None, "created_at": now - 30 * HOUR}
    assert _diag(task, [{"kind": "scheduled", "created_at": now - HOUR, "payload": None}], now) == []


def test_timed_scheduled_card_never_fires():
    now = 1_000_000
    task = {"id": "t_x", "status": "scheduled", "next_eligible_at": now + HOUR, "created_at": now - 30 * HOUR}
    assert _diag(task, [{"kind": "scheduled", "created_at": now - 25 * HOUR, "payload": None}], now) == []


def test_diagnostic_round_trips_a_real_row(kanban_home):
    """The rule must see ``next_eligible_at`` on a real ``Task`` from the DB."""
    with kbc.connect_closing() as conn:
        tid = _scheduled(conn)
        task = kb.get_task(conn, tid)
        events = conn.execute("SELECT * FROM task_events WHERE task_id=?", (tid,)).fetchall()
        assert len(_diag(task, events, int(time.time()) + 25 * HOUR)) == 1
        assert kb.set_schedule_wake(conn, tid, wake_at=int(time.time()) + 48 * HOUR, actor="op")[0]
        task = kb.get_task(conn, tid)
        assert _diag(task, events, int(time.time()) + 25 * HOUR) == []


# --- timed wake ---------------------------------------------------------------


def test_cli_schedule_now_wakes_on_next_tick(kanban_home, capsys):
    with kbc.connect_closing() as conn:
        tid = _scheduled(conn)
    assert kb_cli._cmd_schedule(_args(task_id=tid, now=True)) == 0, capsys.readouterr().err
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "scheduled" and task.next_eligible_at is not None
        assert "schedule_wake_set" in _kinds(conn, tid)

        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: None, max_spawn=0)
        assert res.woken_scheduled == [tid]
        task = kb.get_task(conn, tid)
        assert task.status == "ready" and task.next_eligible_at is None
        kinds = _kinds(conn, tid)
        assert "unblocked" in kinds and "schedule_elapsed" in kinds


def test_dry_run_tick_does_not_wake(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _scheduled(conn)
        assert kb.set_schedule_wake(conn, tid, wake_at=int(time.time()) - 1, actor="op")[0]
        res = kbd.dispatch_once(conn, dry_run=True, spawn_fn=lambda *a, **k: None)
        assert res.woken_scheduled == []
        assert kb.get_task(conn, tid).status == "scheduled"


def test_future_wake_holds_until_due(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _scheduled(conn)
        wake = int(time.time()) + 2 * HOUR
        assert kb.set_schedule_wake(conn, tid, wake_at=wake, actor="op")[0]
        assert kb.wake_due_scheduled(conn) == []
        assert kb.get_task(conn, tid).status == "scheduled"
        assert kb.wake_due_scheduled(conn, now=wake) == [tid]
        assert kb.get_task(conn, tid).status == "ready"


def test_wake_regates_on_open_parents(kanban_home):
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(conn, title="child", parents=[parent], assignee="worker")
        assert kb.schedule_task(conn, child, reason="x")
        assert kb.set_schedule_wake(conn, child, wake_at=int(time.time()), actor="op")[0]
        assert kb.wake_due_scheduled(conn) == [child]
        assert kb.get_task(conn, child).status == "todo"


def test_cli_schedule_at_parks_and_stamps(kanban_home, capsys):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="to park", assignee="worker")
    wake = int(time.time()) + 5 * HOUR
    assert kb_cli._cmd_schedule(_args(task_id=tid, reason=["readback"], at=str(wake))) == 0, \
        capsys.readouterr().err
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "scheduled" and task.next_eligible_at == wake


def test_parse_wake_at():
    assert kb_cli._parse_wake_at("2026-09-27T00:00:00+00:00") == 1790467200
    assert kb_cli._parse_wake_at("1790467200") == 1790467200
    with pytest.raises(ValueError):
        kb_cli._parse_wake_at("tomorrow-ish")


def test_wake_refused_on_non_scheduled(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="ready", assignee="worker")
        ok, err = kb.set_schedule_wake(conn, tid, wake_at=0, actor="op")
        assert not ok and "only applies to 'scheduled'" in err


def test_manual_unblock_clears_wake_stamp(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _scheduled(conn)
        assert kb.set_schedule_wake(conn, tid, wake_at=int(time.time()) + HOUR, actor="op")[0]
        assert kb.unblock_task(conn, tid)
        task = kb.get_task(conn, tid)
        assert task.status == "ready" and task.next_eligible_at is None


def test_promote_refusal_names_the_scheduled_exit(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _scheduled(conn)
        ok, err = kb.promote_task(conn, tid, actor="op")
        assert not ok
        assert "unblock" in err and "schedule" in err and "--now" in err
