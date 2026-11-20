"""``kanban_complete(disposition='continue')``: a worker yields a finished batch (#138729).

Every board is an isolated temp ``HERMES_HOME``; calls go through the real tool registry.
"""
import json
from contextlib import nullcontext

import pytest

from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """A claimed task with the dispatcher's task and run ids pinned in the env."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


def invoke(**overrides):
    from model_tools import handle_function_call
    args = {"summary": "Batch verified; one batch remains.", "disposition": "continue"}
    args.update(overrides)
    return json.loads(handle_function_call("kanban_complete", args, enabled_toolsets=["kanban"]))


def state(tid):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    with connect() as conn:
        return kb.get_task(conn, tid), kb.latest_run(conn, tid), kb.list_events(conn, tid)


def test_continue_requeues_same_card_and_keeps_workspace(worker_env, tmp_path):
    from model_tools import get_tool_definitions
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    defs = get_tool_definitions(enabled_toolsets=["kanban"], quiet_mode=True)
    schema = next(d["function"] for d in defs if d["function"]["name"] == "kanban_complete")
    assert schema["parameters"]["properties"]["disposition"]["enum"] == ["done", "continue"]
    work = tmp_path / "retained-workspace"
    work.mkdir()
    sentinel = work / "evidence.txt"
    sentinel.write_text("retained")
    with connect() as conn:
        from hermes_cli.kanban_db_workspace import set_workspace_path
        set_workspace_path(conn, worker_env, str(work))
        child = kb.create_task(conn, title="dependent", assignee="test-worker", parents=[worker_env])
    _before, old_run, _ = state(worker_env)
    result = invoke(metadata={"remaining": 1})
    assert result["ok"] is True
    assert result["task_id"] == worker_env
    assert result["status"] == "ready"
    assert result["disposition"] == "continue"
    task, run, events = state(worker_env)
    assert task.current_run_id is None
    assert task.claim_lock is None
    assert task.completed_at is None
    assert task.status == "ready"
    assert run.id == old_run.id
    assert run.outcome == "continued"
    assert run.metadata == {"remaining": 1}
    assert run.ended_at is not None
    assert sentinel.read_text() == "retained"
    assert events[-1].kind == "continued"
    assert not any(e.kind in ("completed", "blocked", "block_loop_detected") for e in events)
    with connect() as conn:
        assert kb.get_task(conn, child).status == "todo"
        assert kb.claim_task(conn, worker_env, claimer="successor")
        assert kb.latest_run(conn, worker_env).id != old_run.id
        context = kb.build_worker_context(conn, worker_env)
        assert "Batch verified; one batch remains." in context


@pytest.mark.parametrize("args", [
    {"disposition": "bogus"}, {"disposition": None},
    {"summary": "   "}, {"metadata": []},
    {"artifacts": ["/nonexistent"]}, {"created_cards": ["t_phantom"]},
    {"dispostion": "continue"}, {"task_id": "t_someone_else"},
])
def test_rejections_leave_task_running(worker_env, args):
    _before, run, events = state(worker_env)
    assert invoke(**args).get("error")
    after, next_run, next_events = state(worker_env)
    assert after.status == "running"
    assert after.current_run_id == run.id
    assert next_run.ended_at is None
    assert len(next_events) == len(events)


def _unbind_run(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    return nullcontext()


def _stale_run(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "9999999")
    return nullcontext()


def _terminal_child(monkeypatch):
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    return nullcontext()


_NOT_THE_RUN_OWNER = {
    "unbound": _unbind_run,
    "stale": _stale_run,
    "delegate": lambda _mp: delegated_child_context(),
    "terminal_child": _terminal_child,
    "cron": lambda _mp: non_dispatcher_owned_context(),
}


@pytest.mark.parametrize("kind", sorted(_NOT_THE_RUN_OWNER))
def test_only_the_run_owner_can_continue(worker_env, monkeypatch, kind):
    with _NOT_THE_RUN_OWNER[kind](monkeypatch):
        assert invoke().get("error")
    assert state(worker_env)[0].status == "running"


def test_parent_dependency_guard(worker_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    with connect() as conn:
        parent = kb.create_task(conn, title="new dependency", assignee="test-worker")
        conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (parent, worker_env))
    assert invoke().get("error")
    assert state(worker_env)[0].status == "running"


def test_board_route_does_not_mutate_origin(worker_env):
    assert invoke(board="other-isolated-board").get("error")
    assert state(worker_env)[0].status == "running"


def test_kernel_requires_run_identity(worker_env):
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli.kanban_lifecycle import continue_task
    with connect() as conn:
        assert continue_task(conn, worker_env, summary="batch") is False
    assert state(worker_env)[0].status == "running"


def test_repeat_cannot_terminate_successor(worker_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    assert invoke()["ok"]
    with connect() as conn:
        assert kb.claim_task(conn, worker_env, claimer="successor")
        next_id = kb.latest_run(conn, worker_env).id
    assert invoke().get("error")
    assert state(worker_env)[0].current_run_id == next_id


def test_default_still_completes(worker_env):
    from model_tools import handle_function_call
    result = json.loads(handle_function_call("kanban_complete", {"summary": "All acceptance verified."}, enabled_toolsets=["kanban"]))
    assert result["ok"]
    assert state(worker_env)[0].status == "done"
    assert state(worker_env)[1].outcome == "completed"


def test_continued_pr_author_is_not_held_as_a_duplicate(worker_env):
    """A worker that linked its PR and yielded the batch resumes that PR, not a duplicate."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli.kanban_db_dispatch import check_respawn_guard
    with connect() as conn:
        kb.add_comment(conn, worker_env, "test-worker",
                       "Opened https://github.com/example/repo/pull/12")
        conn.execute("UPDATE task_comments SET created_at = created_at - 5 WHERE task_id = ?",
                     (worker_env,))
    assert invoke()["ok"]
    with connect() as conn:
        assert check_respawn_guard(conn, worker_env) is None
