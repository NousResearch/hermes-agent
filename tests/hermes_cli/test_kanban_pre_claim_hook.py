"""Behavior contract for the synchronous kanban pre-claim plugin hook."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db(board="default")
    with kb.connect(board="default") as conn:
        yield conn, tmp_path


@pytest.fixture
def pre_claim_hooks():
    manager = get_plugin_manager()
    saved = {name: list(callbacks) for name, callbacks in manager._hooks.items()}
    manager._hooks["kanban_task_pre_claim"] = []
    try:
        yield manager._hooks["kanban_task_pre_claim"]
    finally:
        manager._hooks = saved


def _task(conn, root: Path, *, status="ready", max_retries=None):
    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    task_id = kb.create_task(
        conn,
        title=f"{status} task",
        assignee="worker",
        workspace_kind="dir",
        workspace_path=str(workspace),
        max_retries=max_retries,
    )
    if status != "ready":
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
        conn.commit()
    return task_id


def _events(conn, task_id, kind):
    return [event for event in kb.list_events(conn, task_id) if event.kind == kind]


def _runs(conn, task_id):
    return [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (task_id,)
        )
    ]


def test_hook_is_registered_and_no_hook_preserves_spawn(
    board, pre_claim_hooks, all_assignees_spawnable
):
    conn, root = board
    assert "kanban_task_pre_claim" in VALID_HOOKS
    task_id = _task(conn, root)
    calls = []

    result = kb.dispatch_once(
        conn, spawn_fn=lambda task, workspace: calls.append(task.id) or None
    )

    assert calls == [task_id]
    assert result.spawned[0][0] == task_id
    assert result.pre_claim_dispositions == []
    assert result.pre_claim_errors == []


@pytest.mark.parametrize(
    ("decision", "expected_status", "event_kind"),
    [
        (
            {"action": "defer", "reason": "maintenance"},
            "ready",
            "kanban_pre_claim_deferred",
        ),
        (
            {"action": "block", "reason": "needs token", "kind": "needs_input"},
            "blocked",
            "blocked",
        ),
        (
            {
                "action": "complete",
                "summary": "already shipped",
                "evidence": {"pr": 42},
            },
            "done",
            "completed",
        ),
    ],
)
def test_non_allow_actions_skip_workspace_and_spawn_and_persist_disposition(
    board,
    pre_claim_hooks,
    all_assignees_spawnable,
    monkeypatch,
    decision,
    expected_status,
    event_kind,
):
    conn, root = board
    task_id = _task(conn, root)
    conn.execute(
        "UPDATE tasks SET workspace_kind='worktree', workspace_path=? WHERE id=?",
        (str(root / "does-not-exist"), task_id),
    )
    conn.commit()
    pre_claim_hooks.append(lambda **kwargs: decision)
    monkeypatch.setattr(
        kb,
        "_resolve_worktree_workspace",
        lambda *args, **kwargs: pytest.fail("workspace resolver must not run"),
    )
    spawn = lambda *args, **kwargs: pytest.fail("spawn must not run")

    result = kb.dispatch_once(conn, spawn_fn=spawn)
    task = kb.get_task(conn, task_id)

    assert task.status == expected_status
    assert task.consecutive_failures == 0
    assert result.pre_claim_dispositions == [(task_id, decision["action"])]
    assert _events(conn, task_id, event_kind)
    assert not _events(conn, task_id, "spawn_failed")

    if decision["action"] == "block":
        assert task.block_kind == "needs_input"
        assert _runs(conn, task_id)[-1]["summary"] == "needs token"
    if decision["action"] == "complete":
        run = _runs(conn, task_id)[-1]
        assert run["summary"] == "already shipped"
        assert json.loads(run["metadata"]) == {"evidence": {"pr": 42}}
        payload = _events(conn, task_id, "completed")[-1].payload
        assert payload["evidence"] == {"pr": 42}


def test_allow_receives_stable_snapshot_and_spawns(
    board, pre_claim_hooks, all_assignees_spawnable
):
    conn, root = board
    task_id = _task(conn, root)
    original = kb.get_task(conn, task_id)
    captured = {}
    pre_claim_hooks.append(
        lambda **kwargs: captured.update(kwargs) or {"action": "allow"}
    )

    result = kb.dispatch_once(conn, board="default", spawn_fn=lambda *_: None)

    assert result.spawned[0][0] == task_id
    assert result.pre_claim_dispositions == [(task_id, "allow")]
    assert captured.keys() == {
        "task_id",
        "board",
        "assignee",
        "source_status",
        "task",
        "dry_run",
    }
    assert captured["task_id"] == task_id
    assert captured["board"] == "default"
    assert captured["source_status"] == "ready"
    assert captured["dry_run"] is False
    assert captured["task"] == {
        field.name: getattr(original, field.name) for field in fields(kb.Task)
    }


def test_dry_run_allow_reports_decision_and_existing_would_spawn(
    board, pre_claim_hooks, all_assignees_spawnable
):
    conn, root = board
    task_id = _task(conn, root)
    pre_claim_hooks.append(lambda **_: {"action": "allow"})

    result = kb.dispatch_once(conn, dry_run=True)

    assert result.pre_claim_dispositions == [(task_id, "allow")]
    assert result.spawned == [(task_id, "worker", "")]
    assert kb.get_task(conn, task_id).status == "ready"


@pytest.mark.parametrize(
    "callbacks",
    [
        [lambda **_: None],
        [lambda **_: {"action": "wat"}],
        [lambda **_: {"action": "allow", "reason": "extra"}],
        [lambda **_: {"action": "defer", "reason": ""}],
        [lambda **_: {"action": "block", "reason": "x", "kind": "unknown"}],
        [lambda **_: {"action": "complete", "summary": "x", "evidence": {}}],
        [lambda **_: {"action": "complete", "summary": "x", "evidence": {"bad": {1}}}],
        [lambda **_: "allow"],
        [
            lambda **_: {"action": "defer"},
            lambda **_: {"action": "block", "reason": "x", "kind": "capability"},
        ],
    ],
)
def test_malformed_or_conflicting_results_fail_closed(
    board, pre_claim_hooks, all_assignees_spawnable, callbacks
):
    conn, root = board
    task_id = _task(conn, root)
    pre_claim_hooks.extend(callbacks)

    result = kb.dispatch_once(
        conn, spawn_fn=lambda *_: pytest.fail("must not spawn"), failure_limit=5
    )

    task = kb.get_task(conn, task_id)
    assert task.status == "ready"
    assert task.consecutive_failures == 1
    assert result.pre_claim_errors and result.pre_claim_errors[0][0] == task_id
    assert _runs(conn, task_id)[-1]["outcome"] == "kanban_pre_claim_error"
    assert _events(conn, task_id, "kanban_pre_claim_error")
    assert not _events(conn, task_id, "spawn_failed")


def test_callback_exception_uses_failure_breaker_and_is_bounded(
    board, pre_claim_hooks, all_assignees_spawnable
):
    conn, root = board
    task_id = _task(conn, root, max_retries=2)

    def raising(**_):
        raise RuntimeError("policy unavailable")

    pre_claim_hooks.append(raising)
    first = kb.dispatch_once(
        conn, spawn_fn=lambda *_: pytest.fail("must not spawn"), failure_limit=9
    )
    second = kb.dispatch_once(
        conn, spawn_fn=lambda *_: pytest.fail("must not spawn"), failure_limit=9
    )
    third = kb.dispatch_once(
        conn, spawn_fn=lambda *_: pytest.fail("must not spawn"), failure_limit=9
    )

    assert first.pre_claim_errors == [(task_id, "policy unavailable")]
    assert task_id in second.auto_blocked
    assert kb.get_task(conn, task_id).status == "blocked"
    assert third.pre_claim_errors == []
    assert len(_runs(conn, task_id)) == 2
    assert not _events(conn, task_id, "spawn_failed")


@pytest.mark.parametrize(
    "decision",
    [
        {"action": "defer", "reason": "later"},
        {"action": "block", "reason": "human", "kind": "needs_input"},
        {"action": "complete", "summary": "done", "evidence": {"check": True}},
    ],
)
def test_dry_run_reports_without_any_mutation_or_side_effect(
    board, pre_claim_hooks, all_assignees_spawnable, monkeypatch, decision
):
    conn, root = board
    task_id = _task(conn, root)
    before_events = len(kb.list_events(conn, task_id))
    pre_claim_hooks.append(
        lambda **kwargs: decision if kwargs["dry_run"] is True else pytest.fail()
    )
    monkeypatch.setattr(
        kb,
        "resolve_workspace",
        lambda *args, **kwargs: pytest.fail("resolver must not run"),
    )

    result = kb.dispatch_once(
        conn, dry_run=True, spawn_fn=lambda *_: pytest.fail("spawn must not run")
    )

    task = kb.get_task(conn, task_id)
    assert task.status == "ready"
    assert task.consecutive_failures == 0
    assert len(kb.list_events(conn, task_id)) == before_events
    assert _runs(conn, task_id) == []
    assert result.pre_claim_dispositions == [(task_id, decision["action"])]


@pytest.mark.parametrize("source_status", ["ready", "review"])
@pytest.mark.parametrize("action", ["defer", "block", "complete"])
def test_ready_and_review_have_identical_non_allow_contract(
    board, pre_claim_hooks, all_assignees_spawnable, source_status, action
):
    conn, root = board
    task_id = _task(conn, root, status=source_status)
    decisions = {
        "defer": {"action": "defer"},
        "block": {"action": "block", "reason": "policy", "kind": "capability"},
        "complete": {
            "action": "complete",
            "summary": "verified",
            "evidence": {"ok": 1},
        },
    }
    seen = []
    pre_claim_hooks.append(
        lambda **kwargs: seen.append(kwargs["source_status"]) or decisions[action]
    )

    first = kb.dispatch_once(conn, spawn_fn=lambda *_: pytest.fail("must not spawn"))
    second = kb.dispatch_once(conn, spawn_fn=lambda *_: pytest.fail("must not spawn"))

    assert seen[0] == source_status
    assert first.pre_claim_dispositions == [(task_id, action)]
    if action == "defer":
        assert second.pre_claim_dispositions == [(task_id, action)]
        repeated = kb.get_task(conn, task_id)
        assert repeated is not None and repeated.status == source_status
    else:
        assert second.pre_claim_dispositions == []


def test_non_allow_cas_does_not_overwrite_hook_status_change(
    board, pre_claim_hooks, all_assignees_spawnable
):
    conn, root = board
    task_id = _task(conn, root)

    def race(**_):
        with kb.connect(board="default") as other:
            other.execute("UPDATE tasks SET status='todo' WHERE id=?", (task_id,))
            other.commit()
        return {"action": "block", "reason": "stale", "kind": "capability"}

    pre_claim_hooks.append(race)
    result = kb.dispatch_once(conn, spawn_fn=lambda *_: pytest.fail("must not spawn"))

    assert kb.get_task(conn, task_id).status == "todo"
    assert result.pre_claim_dispositions == []
    assert not _events(conn, task_id, "blocked")


def test_hook_runs_while_board_dispatch_lock_is_held(
    board, pre_claim_hooks, all_assignees_spawnable
):
    conn, root = board
    task_id = _task(conn, root)
    nested = []

    def inspect_lock(**_):
        with kb.connect(board="default") as other:
            nested.append(kb.dispatch_once(other).skipped_locked)
        return {"action": "defer"}

    pre_claim_hooks.append(inspect_lock)
    kb.dispatch_once(conn)

    assert nested == [True]
    assert kb.get_task(conn, task_id).status == "ready"
