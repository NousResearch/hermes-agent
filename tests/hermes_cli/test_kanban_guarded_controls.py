"""Native control races: a stale activation must not undo a newer Stop."""
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


def token(conn, task_id):
    return kb.list_events(conn, task_id)[-1].id


@pytest.mark.parametrize("existing_kind", [None, "capability"])
def test_guarded_activation_preserves_newer_stop_and_history(board, existing_kind):
    task_id = kb.create_task(board, title="New assignment", assignee="worker", initial_status="blocked")
    if existing_kind:
        assert kb.block_task(board, task_id, reason="Existing block", kind=existing_kind)
    prepared_activation = token(board, task_id)
    before = kb.get_task(board, task_id)
    with kbc.connect_closing() as operator:
        assert kb.block_task(operator, task_id, reason="Stop", kind="needs_input",
                             expected_event_id=prepared_activation)
    stopped_at = token(board, task_id)
    stopped = kb.get_task(board, task_id)
    assert stopped_at > prepared_activation
    assert stopped.status == "blocked"
    if existing_kind:
        assert (stopped.block_kind, stopped.block_recurrences) == (before.block_kind, before.block_recurrences)
    events = kb.list_events(board, task_id)
    runs = kb.list_runs(board, task_id)
    assert not kb.unblock_task(board, task_id, expected_event_id=prepared_activation)
    assert kb.get_task(board, task_id).status == "blocked"
    assert kb.list_events(board, task_id) == events
    assert kb.list_runs(board, task_id) == runs
    # A new current authorized resume can use the newer token; an old retry cannot.
    assert kb.unblock_task(board, task_id, expected_event_id=stopped_at)
    assert kb.get_task(board, task_id).status == "ready"


@pytest.mark.parametrize("queued_status", ["ready", "todo", "scheduled", "review"])
def test_stop_parks_queued_work_until_an_explicit_resume(board, queued_status):
    parent = kb.create_task(board, title="Upstream work", assignee="worker")
    task_id = kb.create_task(board, title="Queued assignment", assignee="worker",
                             parents=(parent,) if queued_status == "todo" else ())
    if queued_status == "scheduled":
        assert kb.schedule_task(board, task_id, reason="Later")
    elif queued_status == "review":
        assert kb.request_review(board, task_id, summary="Ready for review", reviewer="reviewer")
    assert kb.get_task(board, task_id).status == queued_status
    observed = token(board, task_id)
    assert kb.block_task(board, task_id, reason="Stop queued work", kind="needs_input",
                         expected_event_id=observed)
    assert kb.get_task(board, task_id).status == "blocked"
    kb.recompute_ready(board)
    assert kb.get_task(board, task_id).status == "blocked"
    stopped_at = token(board, task_id)
    assert not kb.unblock_task(board, task_id, expected_event_id=observed)
    assert kb.unblock_task(board, task_id, expected_event_id=stopped_at)
    resumed_status = queued_status if queued_status in ("todo", "review") else "ready"
    assert kb.get_task(board, task_id).status == resumed_status


def test_guarded_stop_rejects_claim_and_replacement_run_races(board):
    task_id = kb.create_task(board, title="Worker claim race", assignee="worker")
    before_claim = token(board, task_id)
    with kbc.connect_closing() as dispatcher:
        run_one = kb.claim_task(dispatcher, task_id)
    claimed_at = token(board, task_id)
    assert not kb.block_task(board, task_id, reason="Stale queued Stop", kind="needs_input",
                             expected_event_id=before_claim)
    assert kb.get_task(board, task_id).status == "running"
    assert token(board, task_id) == claimed_at
    assert kb.block_task(board, task_id, reason="Current Stop", kind="needs_input",
                         expected_run_id=run_one.current_run_id, expected_event_id=claimed_at)
    assert kb.unblock_task(board, task_id, expected_event_id=token(board, task_id))
    run_two = kb.claim_task(board, task_id)
    new_run_at = token(board, task_id)
    assert run_two.current_run_id != run_one.current_run_id
    # Even a fresh event token cannot grant an old worker ownership of a replacement run.
    assert not kb.block_task(board, task_id, reason="Old worker", kind="needs_input",
                             expected_run_id=run_one.current_run_id, expected_event_id=new_run_at)
    assert kb.get_task(board, task_id).current_run_id == run_two.current_run_id
    assert token(board, task_id) == new_run_at


@pytest.mark.parametrize("invalid_token", [True, 0, -1, "1"])
def test_guarded_controls_refuse_invalid_tokens_without_mutation(board, invalid_token):
    task_id = kb.create_task(board, title="Token validation", assignee="worker", initial_status="blocked")
    before = kb.list_events(board, task_id)
    for control in (kb.block_task, kb.unblock_task):
        with pytest.raises(ValueError, match="expected_event_id"):
            control(board, task_id, expected_event_id=invalid_token)
    assert kb.list_events(board, task_id) == before
    assert kb.get_task(board, task_id).status == "blocked"


def test_cli_show_and_guarded_controls_preserve_a_newer_stop(board):
    from hermes_cli import kanban as cli

    task_id = kb.create_task(board, title="CLI assignment", assignee="worker", initial_status="blocked")
    shown = json.loads(cli.run_slash(f"show {task_id} --json"))
    activation = shown["latest_event_id"]
    assert activation == shown["events"][-1]["id"]
    stopped = cli.run_slash(f"block {task_id} Stop --kind needs_input --expected-event-id {activation}")
    assert "Blocked" in stopped
    comments = kb.list_comments(board, task_id)
    events = kb.list_events(board, task_id)
    refused = cli.run_slash(f"unblock {task_id} --reason Stale --expected-event-id {activation}")
    assert "cannot unblock" in refused
    assert kb.list_comments(board, task_id) == comments
    assert kb.list_events(board, task_id) == events
    current = json.loads(cli.run_slash(f"show {task_id} --json"))["latest_event_id"]
    resumed = cli.run_slash(f"unblock {task_id} --reason Resume --expected-event-id {current}")
    assert "Unblocked" in resumed
    assert kb.get_task(board, task_id).status == "ready"
    assert kb.list_comments(board, task_id)[-1].body == "UNBLOCK: Resume"


@pytest.mark.parametrize("verb", ["block", "unblock"])
def test_cli_rejects_guarded_bulk_controls_without_comments_or_events(board, verb):
    from hermes_cli import kanban as cli

    tasks = [kb.create_task(board, title=label, assignee="worker", initial_status="blocked")
             for label in ("First", "Second")]
    observed = token(board, tasks[0])
    before = [kb.list_events(board, task_id) for task_id in tasks]
    command = (f"block {tasks[0]} Stop --ids {tasks[1]} --kind needs_input" if verb == "block"
               else f"unblock {' '.join(tasks)} --reason Resume")
    refused = cli.run_slash(f"{command} --expected-event-id {observed}")
    assert "exactly one task" in refused
    assert [kb.list_events(board, task_id) for task_id in tasks] == before
    assert all(not kb.list_comments(board, task_id) for task_id in tasks)
