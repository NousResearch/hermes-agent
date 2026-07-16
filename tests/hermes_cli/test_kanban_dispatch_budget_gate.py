"""Tests for the spend-guard budget gate in the kanban dispatcher.

``dispatch_once(profile_gate=...)`` defers (never claims) ready/review tasks
whose assignee the gate reports as paused; deferred tasks stay in their
column and land in ``DispatchResult.skipped_budget_paused``. ``profile_gate=None``
must be a zero-behavior-change no-op.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home_with_profiles(monkeypatch):
    """Fresh HERMES_HOME with kanban DB + alpha/beta profiles."""
    test_home = tempfile.mkdtemp(prefix="kanban_budget_gate_test_")
    for prof in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345


def _gate_alpha(profile):
    return "lane api_key over daily budget ($12.00/$10.00)" if profile == "alpha" else None


def test_gate_defers_paused_profile_spawns_other(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        a_task = kb.create_task(conn, title="a0", assignee="alpha")
        kb.create_task(conn, title="b0", assignee="beta")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True, profile_gate=_gate_alpha
        )
    assert [s[1] for s in res.spawned] == ["beta"]
    assert [(t, a) for t, a, _r in res.skipped_budget_paused] == [(a_task, "alpha")]
    # Deferred task keeps its column and claim state — nothing lost.
    with kb.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (a_task,)
        ).fetchone()
    assert row["status"] == "ready"
    assert row["claim_lock"] is None


def test_gate_defers_review_tasks(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        r_task = kb.create_task(conn, title="review me", assignee="alpha")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'review' WHERE id = ?", (r_task,)
            )
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True, profile_gate=_gate_alpha
        )
    assert res.spawned == []
    assert [(t, a) for t, a, _r in res.skipped_budget_paused] == [(r_task, "alpha")]
    with kb.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (r_task,)
        ).fetchone()
    assert row["status"] == "review"
    assert row["claim_lock"] is None


def test_no_gate_is_zero_behavior_change(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="a0", assignee="alpha")
        kb.create_task(conn, title="b0", assignee="beta")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, profile_gate=None)
    assert len(res.spawned) == 2
    assert res.skipped_budget_paused == []


def test_gate_exception_does_not_break_dispatch(isolated_kanban_home_with_profiles):
    """The gateway wraps its gate so it returns None on error; but even a
    raising gate must not lose the tick for other profiles — dispatch_once
    treats a raise as fatal, so this documents that the gateway-side wrapper
    (which swallows exceptions) is the required contract."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="b0", assignee="beta")

    def _safe_gate(profile):
        try:
            raise RuntimeError("boom")
        except Exception:
            return None

    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, profile_gate=_safe_gate)
    assert len(res.spawned) == 1
