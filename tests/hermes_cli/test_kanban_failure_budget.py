#!/usr/bin/env python3
"""The retry budget is per task, not per profile.

``assign_task`` used to clear ``consecutive_failures`` / ``last_failure_error`` whenever the
assignee changed, under the comment "the failure streak is per task/profile". In practice a
dispatcher handoff is ordinary drift, so a card that fails everywhere got a fresh budget on
every handoff and the circuit breaker could never trip: across 49 live boards every task sat
at ``consecutive_failures = 0``.

These tests pin the budget to the card. They call ``_record_task_failure`` directly, the way
``tests/hermes_cli/test_kanban_worker_exit_trailer.py`` does, and reuse the fixture pattern
from ``test_kanban_blocked_sticky.py``.

Kör: python3 -m pytest tests/hermes_cli/test_kanban_failure_budget.py -q
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_db_connect as kbc  # noqa: E402


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    with kbc.connect() as c:
        yield c


def _streak(c, task_id: str) -> int:
    return int(c.execute(
        "SELECT consecutive_failures FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()[0] or 0)


def _status(c, task_id: str) -> str:
    return c.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0]


def _fail_once(c, task_id: str, *, release_claim: bool) -> bool:
    return kb._record_task_failure(
        c, task_id, "synthetic failure",
        outcome="spawn_failed" if release_claim else "timed_out",
        failure_limit=9, release_claim=release_claim, end_run=not release_claim,
    )


def test_profile_handoff_preserves_failure_streak(conn):
    """A dispatcher handoff must not hand the card a fresh retry budget."""
    tid = kb.create_task(conn, title="handoff keeps the streak", board="test", max_retries=9)
    kb.assign_task(conn, tid, "w")
    kb.claim_task(conn, tid)
    assert _fail_once(conn, tid, release_claim=True) is False
    assert _streak(conn, tid) == 1

    kb.assign_task(conn, tid, "w2")

    assert _streak(conn, tid) == 1, "assign_task cleared the streak on a profile handoff"


def test_max_retries_one_blocks_after_single_failure(conn):
    """max_retries=1 blocks on the first failure (card requirement 3)."""
    tid = kb.create_task(conn, title="single strike", board="test", max_retries=1)
    kb.assign_task(conn, tid, "w")
    kb.claim_task(conn, tid)

    assert _fail_once(conn, tid, release_claim=False) is True

    assert _status(conn, tid) == "blocked"
    assert _streak(conn, tid) == 1
    kinds = [r[0] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ?", (tid,)
    ).fetchall()]
    assert "gave_up" in kinds


def test_two_profile_handoffs_trip_breaker(conn):
    """The budget survives two handoffs, so the third failure blocks."""
    tid = kb.create_task(conn, title="survives handoffs", board="test", max_retries=2)
    for assignee in ("w", "w2", "w3"):
        kb.assign_task(conn, tid, assignee)
        kb.claim_task(conn, tid)
        tripped = _fail_once(conn, tid, release_claim=True)
    assert tripped is True
    assert _status(conn, tid) == "blocked"
    assert _streak(conn, tid) == 2


def test_complete_task_clears_streak(conn):
    """Progress still resets the budget — a card that succeeded starts fresh."""
    tid = kb.create_task(conn, title="progress resets", board="test", max_retries=9)
    kb.assign_task(conn, tid, "w")
    kb.claim_task(conn, tid)
    _fail_once(conn, tid, release_claim=True)
    assert _streak(conn, tid) == 1

    kb.claim_task(conn, tid)
    kb.complete_task(conn, tid, result="klar")

    assert _streak(conn, tid) == 0