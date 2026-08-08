"""Tests for kanban lifecycle plugin hooks.

Verifies that claim/complete/block transitions fire the
kanban_task_claimed / kanban_task_completed / kanban_task_blocked plugin
hooks AFTER the board DB change is committed, with the documented kwargs,
and that a misbehaving hook callback never breaks the transition.
"""

from __future__ import annotations

import sqlite3
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
    """Register capturing callbacks for the three kanban lifecycle hooks.

    Patches the plugin manager's _hooks dict directly (the same registry
    invoke_hook reads) and restores it afterward.
    """
    mgr = get_plugin_manager()
    events: list[tuple[str, dict]] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    for hook in ("kanban_task_claimed", "kanban_task_completed", "kanban_task_blocked"):
        mgr._hooks.setdefault(hook, []).append(
            lambda _h=hook, **kw: events.append((_h, kw))
        )
    try:
        yield events
    finally:
        mgr._hooks = saved


def test_dependency_block_hook_waits_for_full_transaction_boundary(
    kanban_home, monkeypatch
):
    """Dependency hooks observe committed state and not invariant-failed writes."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    observed: list[tuple[bool, str, list[str], str]] = []

    def _observe_committed_state(*, task_id, **_kwargs):
        origin_in_transaction = conn.in_transaction
        with kb.connect_closing() as observer:
            task = kb.get_task(observer, task_id)
            assert task is not None
            kinds = [event.kind for event in kb.list_events(observer, task_id)]
            callback_task_id = kb.create_task(
                observer, title="callback writer", assignee="observer"
            )
        observed.append(
            (origin_in_transaction, task.status, kinds, callback_task_id)
        )

    try:
        conn = kb.connect()
        try:
            first = kb.create_task(conn, title="first", assignee="worker")
            assert kb.claim_task(conn, first) is not None
            mgr._hooks.setdefault("kanban_task_blocked", []).append(
                _observe_committed_state
            )

            assert kb.block_task(
                conn,
                first,
                reason="waiting for parent",
                kind="dependency",
            ) is True
            assert len(observed) == 1
            in_txn, status, kinds, callback_task_id = observed[0]
            assert in_txn is False
            assert status == "todo"
            assert kinds == ["created", "claimed", "dependency_wait"]
            assert kb.get_task(conn, callback_task_id) is not None

            second = kb.create_task(conn, title="second", assignee="worker")
            assert kb.claim_task(conn, second) is not None

            def _invariant_failed(_conn):
                raise RuntimeError("forced post-commit invariant failure")

            monkeypatch.setattr(kb, "_check_file_length_invariant", _invariant_failed)
            with pytest.raises(RuntimeError, match="forced post-commit invariant failure"):
                kb.block_task(
                    conn,
                    second,
                    reason="waiting for parent",
                    kind="dependency",
                )

            assert len(observed) == 1
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


def test_dependency_block_failures_suppress_hook_and_transition(
    kanban_home, monkeypatch
):
    """Body and terminal commit failures roll back without dispatch."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    fired: list[dict] = []
    real_get_task = kb.get_task
    real_boundary = kb._execute_boundary_with_retry

    try:
        with kb.connect_closing() as conn:
            body_task = kb.create_task(conn, title="body failure", assignee="worker")
            commit_task = kb.create_task(conn, title="commit failure", assignee="worker")
            assert kb.claim_task(conn, body_task) is not None
            assert kb.claim_task(conn, commit_task) is not None
            mgr._hooks.setdefault("kanban_task_blocked", []).append(
                lambda **kwargs: fired.append(kwargs)
            )

            def _fail_after_append(candidate_conn, task_id):
                if candidate_conn is conn and task_id == body_task:
                    raise RuntimeError("forced body failure")
                return real_get_task(candidate_conn, task_id)

            monkeypatch.setattr(kb, "get_task", _fail_after_append)
            with pytest.raises(RuntimeError, match="forced body failure"):
                kb.block_task(
                    conn, body_task, reason="body", kind="dependency"
                )
            monkeypatch.setattr(kb, "get_task", real_get_task)

            assert fired == []
            body_after = real_get_task(conn, body_task)
            assert body_after is not None
            assert body_after.status == "running"
            assert "dependency_wait" not in {
                event.kind for event in kb.list_events(conn, body_task)
            }

            def _fail_commit(candidate_conn, sql):
                if candidate_conn is conn and sql == "COMMIT":
                    raise sqlite3.OperationalError("forced terminal commit failure")
                return real_boundary(candidate_conn, sql)

            monkeypatch.setattr(kb, "_execute_boundary_with_retry", _fail_commit)
            with pytest.raises(
                sqlite3.OperationalError, match="forced terminal commit failure"
            ):
                kb.block_task(
                    conn, commit_task, reason="commit", kind="dependency"
                )

            assert fired == []
            assert conn.in_transaction is False
            commit_after = real_get_task(conn, commit_task)
            assert commit_after is not None
            assert commit_after.status == "running"
            assert "dependency_wait" not in {
                event.kind for event in kb.list_events(conn, commit_task)
            }
    finally:
        mgr._hooks = saved


def test_dependency_block_payload_and_observer_failure_isolation(kanban_home):
    """Dependency callback kwargs stay stable and observers remain fail-open."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    seen: list[dict] = []

    def _boom(**_kwargs):
        raise RuntimeError("observer exploded")

    try:
        with kb.connect_closing() as conn:
            task_id = kb.create_task(conn, title="payload", assignee="worker")
            claimed = kb.claim_task(conn, task_id)
            assert claimed is not None
            mgr._hooks.setdefault("kanban_task_blocked", []).extend(
                [_boom, lambda **kwargs: seen.append(kwargs)]
            )

            assert kb.block_task(
                conn,
                task_id,
                reason="waiting",
                kind="dependency",
                expected_run_id=claimed.current_run_id,
            ) is True

            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.status == "todo"
            assert [event.kind for event in kb.list_events(conn, task_id)][-1] == (
                "dependency_wait"
            )

        assert len(seen) == 1
        assert seen[0]["task_id"] == task_id
        assert seen[0]["assignee"] == "worker"
        assert seen[0]["run_id"] == claimed.current_run_id
        assert seen[0]["reason"] == "waiting"
        assert isinstance(seen[0]["board"], str)
        assert isinstance(seen[0]["profile_name"], str)
    finally:
        mgr._hooks = saved


def test_sibling_lifecycle_hooks_remain_post_commit(kanban_home):
    """Claim, complete, hard-block, and loop paths retain post-commit timing."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    observed: list[tuple[str, bool, str, str]] = []

    try:
        with kb.connect_closing() as conn:
            def _observe(hook_name):
                def _callback(*, task_id, **_kwargs):
                    in_transaction = conn.in_transaction
                    with kb.connect_closing() as observer:
                        task = kb.get_task(observer, task_id)
                        assert task is not None
                        last_kind = kb.list_events(observer, task_id)[-1].kind
                    observed.append(
                        (hook_name, in_transaction, task.status, last_kind)
                    )
                return _callback

            for hook_name in (
                "kanban_task_claimed",
                "kanban_task_completed",
                "kanban_task_blocked",
            ):
                mgr._hooks.setdefault(hook_name, []).append(_observe(hook_name))

            loop_task = kb.create_task(conn, title="loop", assignee="worker")
            assert kb.claim_task(conn, loop_task) is not None
            assert kb.block_task(
                conn, loop_task, reason="same", kind="capability"
            ) is True
            assert kb.unblock_task(conn, loop_task) is True
            assert kb.claim_task(conn, loop_task) is not None
            assert kb.block_task(
                conn, loop_task, reason="same", kind="capability"
            ) is True

            complete_task = kb.create_task(
                conn, title="complete", assignee="worker"
            )
            assert kb.claim_task(conn, complete_task) is not None
            assert kb.complete_task(conn, complete_task, summary="done") is True

        assert observed == [
            ("kanban_task_claimed", False, "running", "claimed"),
            ("kanban_task_blocked", False, "blocked", "blocked"),
            ("kanban_task_claimed", False, "running", "claimed"),
            ("kanban_task_blocked", False, "triage", "block_loop_detected"),
            ("kanban_task_claimed", False, "running", "claimed"),
            ("kanban_task_completed", False, "done", "completed"),
        ]
    finally:
        mgr._hooks = saved


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
