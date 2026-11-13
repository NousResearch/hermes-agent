"""Tests for the ``pre_kanban_complete`` directive hook.

Verifies that ``complete_task`` fires the pre-commit ``pre_kanban_complete``
hook BEFORE any gate or state mutation, with the documented kwargs, and that a
plugin's ``{"action": "block", "message"}`` return vetoes the close (raising
``PreKanbanCompleteVetoError``) while ``None`` allows it. A raising hook must
fail open (never block the transition).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
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
def hook_registry(monkeypatch):
    """Patch the plugin manager's ``_hooks`` dict so we can register callbacks
    for ``pre_kanban_complete`` and restore it afterward (same technique as
    ``test_kanban_lifecycle_hooks.py``)."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    try:
        yield mgr
    finally:
        mgr._hooks = saved


def _claimed_task(conn) -> str:
    """Create + claim a task so it is ``running`` with no live worker (claim is
    not live without a spawned PID), so ``complete_task`` can close it."""
    tid = kb.create_task(conn, title="t", assignee="worker")
    assert kb.claim_task(conn, tid) is not None
    return tid


def test_hook_is_registered_in_valid_hooks():
    assert "pre_kanban_complete" in VALID_HOOKS


def test_hook_fires_before_commit_with_documented_kwargs(kanban_home, hook_registry):
    events: list[dict] = []
    hook_registry._hooks.setdefault("pre_kanban_complete", []).append(
        lambda **kw: events.append(kw)
    )
    conn = kbc.connect()
    try:
        tid = _claimed_task(conn)
        assert kb.complete_task(conn, tid, summary="ok", result="commit abc1234") is True
    finally:
        conn.close()
    assert len(events) == 1
    kw = events[0]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "worker"
    assert kw["status"] == "running"  # fired BEFORE the status flipped to done
    assert kw["summary"] == "ok"
    assert kw["result"] == "commit abc1234"
    assert "profile_name" in kw
    assert "board" in kw
    assert kw["run_id"] is not None


def test_block_directive_vetoes_close(kanban_home, hook_registry):
    hook_registry._hooks.setdefault("pre_kanban_complete", []).append(
        lambda **kw: {"action": "block", "message": "no receipt attached"}
    )
    conn = kbc.connect()
    try:
        tid = _claimed_task(conn)
        with pytest.raises(kb.PreKanbanCompleteVetoError) as exc:
            kb.complete_task(conn, tid, summary="done", result="commit abc1234")
        assert exc.value.task_id == tid
        assert exc.value.reason == "no receipt attached"
        # Nothing mutated: still running, no completed event.
        assert kb.get_task(conn, tid).status == "running"
        kinds = [r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
        ).fetchall()]
        assert "completed" not in kinds
    finally:
        conn.close()


def test_reason_alias_vetoes_close(kanban_home, hook_registry):
    hook_registry._hooks.setdefault("pre_kanban_complete", []).append(
        lambda **kw: {"action": "block", "reason": "missing provenance"}
    )
    conn = kbc.connect()
    try:
        tid = _claimed_task(conn)
        with pytest.raises(kb.PreKanbanCompleteVetoError) as exc:
            kb.complete_task(conn, tid, summary="done", result="commit abc1234")
        assert exc.value.reason == "missing provenance"
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_none_return_allows_close(kanban_home, hook_registry):
    hook_registry._hooks.setdefault("pre_kanban_complete", []).append(
        lambda **kw: None
    )
    conn = kbc.connect()
    try:
        tid = _claimed_task(conn)
        assert kb.complete_task(conn, tid, summary="ok") is True
        assert kb.get_task(conn, tid).status == "done"
    finally:
        conn.close()


def test_non_block_directive_does_not_veto(kanban_home, hook_registry):
    hook_registry._hooks.setdefault("pre_kanban_complete", []).append(
        lambda **kw: {"action": "allow"}
    )
    conn = kbc.connect()
    try:
        tid = _claimed_task(conn)
        assert kb.complete_task(conn, tid, summary="ok") is True
        assert kb.get_task(conn, tid).status == "done"
    finally:
        conn.close()


def test_block_without_message_is_ignored(kanban_home, hook_registry):
    # A block directive with no message is not a valid veto (mirrors pre_tool_call).
    hook_registry._hooks.setdefault("pre_kanban_complete", []).append(
        lambda **kw: {"action": "block"}
    )
    conn = kbc.connect()
    try:
        tid = _claimed_task(conn)
        assert kb.complete_task(conn, tid, summary="ok") is True
        assert kb.get_task(conn, tid).status == "done"
    finally:
        conn.close()


def test_misbehaving_hook_fails_open(kanban_home, hook_registry):
    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    hook_registry._hooks.setdefault("pre_kanban_complete", []).append(_boom)
    conn = kbc.connect()
    try:
        tid = _claimed_task(conn)
        # A raising hook must never block the close.
        assert kb.complete_task(conn, tid, summary="ok") is True
        assert kb.get_task(conn, tid).status == "done"
    finally:
        conn.close()
