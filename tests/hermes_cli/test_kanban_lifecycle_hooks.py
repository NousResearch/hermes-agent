"""Tests for kanban lifecycle plugin hooks.

Verifies that create / claim / complete / block / unblock transitions fire the
kanban_task_created / kanban_task_claimed / kanban_task_completed /
kanban_task_blocked / kanban_task_unblocked plugin hooks AFTER the board DB
change is committed (observer hooks), and that kanban_pre_complete fires
BEFORE the write txn and can veto the transition (governance hooks).

Also verifies that a misbehaving hook callback never breaks the transition.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_hooks(monkeypatch):
    """Register capturing callbacks for all six kanban lifecycle hooks.

    Patches the plugin manager's _hooks dict directly (the same registry
    invoke_hook reads) and restores it afterward.
    """
    mgr = get_plugin_manager()
    events: list[tuple[str, dict]] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    for hook in (
        "kanban_task_created",
        "kanban_task_claimed",
        "kanban_task_completed",
        "kanban_task_blocked",
        "kanban_task_unblocked",
    ):
        mgr._hooks.setdefault(hook, []).append(
            lambda _h=hook, **kw: events.append((_h, kw))
        )
    try:
        yield events
    finally:
        mgr._hooks = saved


# ── VALID_HOOKS registration ────────────────────────────────────────────


def test_hooks_are_registered_as_valid():
    """All six kanban lifecycle hook names are part of VALID_HOOKS."""
    assert "kanban_task_created" in VALID_HOOKS
    assert "kanban_task_claimed" in VALID_HOOKS
    assert "kanban_task_completed" in VALID_HOOKS
    assert "kanban_task_blocked" in VALID_HOOKS
    assert "kanban_task_unblocked" in VALID_HOOKS
    assert "kanban_pre_complete" in VALID_HOOKS


# ── Observer hooks ──────────────────────────────────────────────────────


def test_create_fires_hook(kanban_home, captured_hooks):
    """create_task fires kanban_task_created with the task id and assignee."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="test create", assignee="worker")
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_created"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "worker"
    assert "profile_name" in kw


def test_claim_fires_hook(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_claimed"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "worker"
    assert "profile_name" in kw
    assert kw["run_id"] is not None


def test_complete_fires_hook_with_summary(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.complete_task(conn, tid, summary="all done")
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_completed"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["summary"] == "all done"
    assert kw["assignee"] == "worker"


def test_block_fires_hook_with_reason(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.block_task(conn, tid, reason="needs human")
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_blocked"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["reason"] == "needs human"


def test_unblock_fires_hook(kanban_home, captured_hooks):
    """unblock_task fires kanban_task_unblocked with the task id and assignee."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="test unblock", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.block_task(conn, tid, reason="test block")
        assert kb.unblock_task(conn, tid)
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_unblocked"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "worker"
    assert "profile_name" in kw


# ── Governance hooks (kanban_pre_complete) ──────────────────────────────


def test_pre_complete_allows_completion(kanban_home, monkeypatch):
    """A pre_complete hook returning None/allow permits the transition."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    calls: list[dict] = []

    def _allow(**kw):
        calls.append(kw)
        return {"action": "allow"}

    mgr._hooks.setdefault("kanban_pre_complete", []).append(_allow)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            kb.claim_task(conn, tid)
            assert kb.complete_task(conn, tid, summary="ok") is True
            assert kb.get_task(conn, tid).status == "done"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved

    assert len(calls) == 1
    assert calls[0]["task_id"] == tid
    assert calls[0]["summary"] == "ok"
    assert calls[0]["assignee"] == "worker"


def test_pre_complete_blocks_completion(kanban_home, monkeypatch):
    """A pre_complete hook returning {"action": "block"} vetoes the
    transition — the task stays in its current state and complete_task
    returns False."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _veto(**kw):
        return {"action": "block", "reason": "attestation missing"}

    mgr._hooks.setdefault("kanban_pre_complete", []).append(_veto)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            kb.claim_task(conn, tid)
            # The completion is vetoed.
            assert kb.complete_task(conn, tid, summary="should fail") is False
            # Task is still in its pre-completion state (running).
            assert kb.get_task(conn, tid).status == "running"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


def test_pre_complete_block_is_auditable(kanban_home, monkeypatch):
    """When kanban_pre_complete blocks, a completion_blocked_governance
    event is recorded in the event log."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _veto(**kw):
        return {"action": "block", "reason": "attestation missing"}

    mgr._hooks.setdefault("kanban_pre_complete", []).append(_veto)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            kb.claim_task(conn, tid)
            assert kb.complete_task(conn, tid, summary="blocked") is False

            events = kb.list_events(conn, tid)
            blocked_events = [
                e for e in events
                if e.kind == "completion_blocked_governance"
            ]
            assert len(blocked_events) == 1
            payload = blocked_events[0].payload
            assert payload["reason"] == "attestation missing"
            assert payload["source"] == "kanban_pre_complete"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


def test_pre_complete_hook_error_non_fatal(kanban_home, monkeypatch):
    """A pre_complete callback that raises does NOT block completion —
    the exception is caught and the transition proceeds normally."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    mgr._hooks.setdefault("kanban_pre_complete", []).append(_boom)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            kb.claim_task(conn, tid)
            # Completion succeeds despite the raising hook.
            assert kb.complete_task(conn, tid, summary="ok") is True
            assert kb.get_task(conn, tid).status == "done"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


# ── Failed transition → no hook ─────────────────────────────────────────


def test_no_hook_on_failed_transition(kanban_home, captured_hooks):
    """complete_task on an unclaimed/nonexistent task fires no hook."""
    conn = kb.connect()
    try:
        # Completing a task that doesn't exist returns False without firing.
        assert kb.complete_task(conn, "t_doesnotexist", summary="x") is False
    finally:
        conn.close()
    assert [e for e in captured_hooks if e[0] == "kanban_task_completed"] == []


# ── Observer hook resilience ────────────────────────────────────────────


def test_misbehaving_hook_does_not_break_transition(kanban_home, monkeypatch):
    """A hook callback that raises must not break the board transition."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    mgr._hooks.setdefault("kanban_task_completed", []).append(_boom)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            kb.claim_task(conn, tid)
            # Despite the raising hook, completion succeeds and persists.
            assert kb.complete_task(conn, tid, summary="ok") is True
            assert kb.get_task(conn, tid).status == "done"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved
