from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def claimed(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_id = kb.create_task(
        conn, title="external work", assignee="codex-lane",
        workspace_kind="dir", workspace_path=str(workspace),
    )
    task = kb.claim_task(conn, task_id, claimer="claim-123", ttl_seconds=300)
    assert task is not None
    current = kb.get_task(conn, task_id)
    yield conn, task_id, current.current_run_id, workspace
    conn.close()


def _call(claimed, **overrides):
    conn, task_id, run_id, workspace = claimed
    args = dict(
        run_id=run_id, profile="codex-lane", workspace=str(workspace),
        claim_lock="claim-123", idempotency_key="attempt-1",
        transition="complete", comment="handoff: tests pass",
        summary="implemented atomic handoff", metadata={"tests": 6}, result="ok",
    )
    args.update(overrides)
    return kb.terminal_handoff(conn, task_id, **args)


def test_complete_commits_comment_result_run_and_events_atomically(claimed):
    response = _call(claimed)
    conn, task_id, run_id, _ = claimed
    task = kb.get_task(conn, task_id)
    run = kb.latest_run(conn, task_id)
    assert response == {"task_id": task_id, "run_id": run_id, "status": "done", "transition": "complete", "comment_id": 1, "replayed": False}
    assert task.status == "done" and task.result == "ok" and task.current_run_id is None
    assert run.outcome == "completed" and run.summary == "implemented atomic handoff"
    assert [c.body for c in kb.list_comments(conn, task_id)] == ["handoff: tests pass"]
    assert [e.kind for e in kb.list_events(conn, task_id)][-2:] == ["commented", "completed"]


def test_replay_returns_original_without_duplication(claimed):
    first = _call(claimed)
    replay = _call(claimed)
    conn, task_id, _, _ = claimed
    assert replay == {**first, "replayed": True}
    assert len(kb.list_comments(conn, task_id)) == 1
    assert len([e for e in kb.list_events(conn, task_id) if e.kind == "completed"]) == 1


@pytest.mark.parametrize("override, message", [
    ({"run_id": 999999}, "stale or superseded"),
    ({"workspace": "/wrong/workspace"}, "workspace does not match"),
    ({"profile": "wrong-lane"}, "profile does not match"),
    ({"claim_lock": "reclaimed"}, "claim lock does not match"),
])
def test_mismatched_claim_rejects_without_partial_mutation(claimed, override, message):
    conn, task_id, _, _ = claimed
    before_events = len(kb.list_events(conn, task_id))
    with pytest.raises(kb.TerminalHandoffError, match=message):
        _call(claimed, **override)
    assert kb.get_task(conn, task_id).status == "running"
    assert kb.list_comments(conn, task_id) == []
    assert len(kb.list_events(conn, task_id)) == before_events
    assert conn.execute("SELECT count(*) FROM terminal_handoffs").fetchone()[0] == 0


def test_block_commits_handoff_and_terminal_transition(claimed):
    response = _call(
        claimed, transition="block", summary=None, result=None,
        reason="review-required: inspect diff",
    )
    conn, task_id, _, _ = claimed
    assert response["status"] == "blocked"
    assert kb.get_task(conn, task_id).status == "blocked"
    run = kb.latest_run(conn, task_id)
    assert run.status == "blocked"
    assert run.outcome == "blocked"
    assert kb.list_comments(conn, task_id)[0].body == "handoff: tests pass"


def test_dependency_block_routes_to_todo_atomically(claimed):
    response = _call(
        claimed, transition="block", summary=None, result=None,
        reason="waiting for parent output", kind="dependency",
    )
    conn, task_id, _, _ = claimed
    task = kb.get_task(conn, task_id)
    assert response["status"] == "todo"
    assert task.status == "todo"
    assert task.block_kind == "dependency"
    assert task.block_recurrences == 0
    run = kb.latest_run(conn, task_id)
    assert run.status == "blocked"
    assert run.outcome == "blocked"
    assert [e.kind for e in kb.list_events(conn, task_id)][-2:] == [
        "commented", "dependency_wait",
    ]


def test_repeated_same_kind_block_escalates_to_triage_atomically(claimed):
    conn, task_id, _, _ = claimed
    conn.execute(
        "UPDATE tasks SET block_kind = ?, block_recurrences = ? WHERE id = ?",
        ("needs_input", kb.BLOCK_RECURRENCE_LIMIT - 1, task_id),
    )
    conn.commit()

    response = _call(
        claimed, transition="block", summary=None, result=None,
        reason="still awaiting operator choice", kind="needs_input",
    )
    task = kb.get_task(conn, task_id)
    assert response["status"] == "triage"
    assert task.status == "triage"
    assert task.block_kind == "needs_input"
    assert task.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT
    assert kb.latest_run(conn, task_id).status == "blocked"
    assert kb.list_events(conn, task_id)[-1].kind == "block_loop_detected"
