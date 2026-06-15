"""Tests for native iteration-budget escalation (auto-heal replacement).

next_budget_tier (pure ladder) + escalate_and_requeue (a running worker that
exhausted its budget but is below the cap gets a higher max_iterations and is
re-queued to `ready` WITHOUT counting a failure).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ── pure ladder ─────────────────────────────────────────────────────

def test_next_budget_tier_ladder():
    assert kb.next_budget_tier(None) == 120   # default 90 -> first tier
    assert kb.next_budget_tier(90) == 120
    assert kb.next_budget_tier(120) == 150
    assert kb.next_budget_tier(150) == 200
    assert kb.next_budget_tier(200) is None    # cap -> block for human
    assert kb.next_budget_tier(999) is None
    assert kb.next_budget_tier("bad") is None


# ── escalate_and_requeue ────────────────────────────────────────────

def test_escalate_running_task_requeues_with_higher_budget(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="big", assignee="worker", max_iterations=90)
        kb.claim_task(conn, tid)
        assert kb.get_task(conn, tid).status == "running"

        ok = kb.escalate_and_requeue(conn, tid, 120, reason="iteration budget exhausted")
        assert ok is True

        t = kb.get_task(conn, tid)
        assert t.status == "ready"             # re-queued for a fresh spawn
        assert t.max_iterations == 120         # higher budget persisted
        assert t.claim_lock is None            # claim released
        # NOT counted as a failure (breaker must not advance).
        assert (t.consecutive_failures or 0) == 0
        # Auditable event emitted.
        assert any(e.kind == "budget_escalated" for e in kb.list_events(conn, tid))
    finally:
        conn.close()


def test_escalate_does_not_trip_circuit_breaker(kanban_home):
    # Two escalations in a row must NOT auto-block (failure_limit default 2).
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker", max_iterations=90)
        kb.claim_task(conn, tid)
        assert kb.escalate_and_requeue(conn, tid, 120) is True
        kb.claim_task(conn, tid)               # re-spawn claims it
        assert kb.escalate_and_requeue(conn, tid, 150) is True
        t = kb.get_task(conn, tid)
        assert t.status == "ready"             # still ready, NOT blocked
        assert t.max_iterations == 150
    finally:
        conn.close()


def test_escalate_non_running_returns_false(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker", max_iterations=90)
        # never claimed -> not running
        assert kb.escalate_and_requeue(conn, tid, 120) is False
        assert kb.get_task(conn, tid).max_iterations == 90  # untouched
    finally:
        conn.close()


def test_escalate_declines_when_not_strict_increase(kanban_home):
    # P0 guard: if the task's STORED budget already >= the requested tier
    # (e.g. config pins the effective budget so the caller keeps computing the
    # same top tier), decline -> block path, instead of looping forever.
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker", max_iterations=200)
        kb.claim_task(conn, tid)
        assert kb.escalate_and_requeue(conn, tid, 200) is False   # equal, not higher
        assert kb.escalate_and_requeue(conn, tid, 150) is False   # lower
        t = kb.get_task(conn, tid)
        assert t.status == "running" and t.max_iterations == 200  # untouched
    finally:
        conn.close()


def test_escalate_invalid_budget_returns_false(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker", max_iterations=90)
        kb.claim_task(conn, tid)
        assert kb.escalate_and_requeue(conn, tid, 0) is False
        assert kb.escalate_and_requeue(conn, tid, "bad") is False
        assert kb.get_task(conn, tid).status == "running"  # untouched
    finally:
        conn.close()
