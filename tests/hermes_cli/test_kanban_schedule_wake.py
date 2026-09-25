"""Scheduled parks out of review and the timed wake (promote_due_scheduled).

A review card had no supported park: schedule_task refused ``review``, so a
reviewer waiting on something external left the card in ``review`` and the
next tick re-claimed it. A park that did land resumed into ``ready``, losing
the review phase, and a ``wake_at`` had nowhere to live.
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


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _park_review_card(conn, *, wake_at=None, title="review park"):
    tid = kb.create_task(conn, title=title, assignee="impl")
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    assert kb.request_review(conn, tid, summary="please review", expected_run_id=run_id)
    assert kb.get_task(conn, tid).status == "review"
    assert kb.schedule_task(conn, tid, reason="waiting on CI", wake_at=wake_at) is True
    return tid


def _events(conn, tid, kind):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


def test_review_card_parks_and_is_not_reclaimable(conn):
    wake = int(time.time()) + 3600
    tid = _park_review_card(conn, wake_at=wake)
    task = kb.get_task(conn, tid)
    assert task.status == "scheduled"
    assert task.scheduled_wake_at == wake
    assert kb.claim_review_task(conn, tid) is None
    assert kb.claim_task(conn, tid) is None
    ev = _events(conn, tid, "scheduled")[-1].payload
    assert ev["source_status"] == "review"
    assert ev["wake_at"] == wake


def test_unblock_restores_review_phase_and_clears_wake(conn):
    tid = _park_review_card(conn, wake_at=int(time.time()) + 3600)
    assert kb.unblock_task(conn, tid) is True
    task = kb.get_task(conn, tid)
    assert task.status == "review"
    assert task.scheduled_wake_at is None


def test_ready_park_without_wake_still_resumes_ready(conn):
    tid = kb.create_task(conn, title="plain", assignee="impl")
    assert kb.schedule_task(conn, tid, reason="next week") is True
    ev = _events(conn, tid, "scheduled")[-1].payload
    assert ev["source_status"] == "ready" and ev["wake_at"] is None
    assert kb.promote_due_scheduled(conn) == []
    assert kb.unblock_task(conn, tid) is True
    assert kb.get_task(conn, tid).status == "ready"


def test_promote_due_scheduled_only_wakes_past_due(conn):
    due = _park_review_card(conn, wake_at=int(time.time()) - 5, title="due")
    later = _park_review_card(conn, wake_at=int(time.time()) + 3600, title="later")
    manual = _park_review_card(conn, wake_at=None, title="manual")

    assert kb.promote_due_scheduled(conn) == [due]
    assert kb.get_task(conn, due).status == "review"
    assert kb.get_task(conn, due).scheduled_wake_at is None
    assert kb.get_task(conn, later).status == "scheduled"
    assert kb.get_task(conn, manual).status == "scheduled"
    assert _events(conn, due, "scheduled_wake")[-1].payload["status"] == "review"


def test_repeated_scheduled_waits_do_not_trip_the_block_loop_breaker(conn):
    tid = _park_review_card(conn, wake_at=int(time.time()) - 5)
    assert kb.promote_due_scheduled(conn) == [tid]
    assert kb.schedule_task(conn, tid, reason="waiting on CI", wake_at=int(time.time()) - 5)
    assert kb.promote_due_scheduled(conn) == [tid]
    task = kb.get_task(conn, tid)
    assert task.status == "review"
    assert task.block_recurrences == 0


def test_dispatcher_tick_wakes_due_cards(conn):
    tid = kb.create_task(conn, title="tick", assignee="impl")
    assert kb.schedule_task(conn, tid, wake_at=int(time.time()) - 1)
    result = kbd.dispatch_once(conn, dry_run=True)
    assert result.woken_scheduled == 1
    assert kb.get_task(conn, tid).status in ("ready", "running")


def test_cli_schedule_wake_at(kanban_home, capsys):
    with kbc.connect() as c:
        tid = kb.create_task(c, title="cli", assignee="impl")
    ns = argparse.Namespace(task_id=tid, reason=["later"], ids=None, wake_at="2999-01-01T00:00:00")
    assert kb_cli._cmd_schedule(ns) == 0
    with kbc.connect() as c:
        task = kb.get_task(c, tid)
    assert task.status == "scheduled"
    assert task.scheduled_wake_at == kb_cli._parse_wake_at("2999-01-01T00:00:00")
    assert kb_cli._parse_wake_at("1700000000") == 1700000000
    bad = argparse.Namespace(task_id=tid, reason=[], ids=None, wake_at="tomorrow-ish")
    assert kb_cli._cmd_schedule(bad) != 0
