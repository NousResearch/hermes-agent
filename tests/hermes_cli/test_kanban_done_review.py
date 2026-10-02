"""Explicit review of completed cards keeps their evidence and dependency fences (#99282)."""

from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_review as kbr
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(key, raising=False)
    kb.init_db()
    return home


@pytest.mark.parametrize("completion", ["worker", "manual", "review"])
def test_cli_done_review_preserves_evidence_and_routes_changes(kanban_home, completion):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Completed implementation", assignee="builder")
        workspace = kbw.resolve_workspace(kb.get_task(conn, task_id))
        kbw.set_workspace_path(conn, task_id, workspace)
        artifact = workspace / "report.txt"
        artifact.write_text("verified output", encoding="utf-8")
        run_id = None
        if completion != "manual":
            run_id = kb.claim_task(conn, task_id).current_run_id
        if completion == "review":
            assert kbr.request_review(
                conn, task_id, summary="Ready", reviewer="reviewer", expected_run_id=run_id,
            )
            run_id = kb.claim_review_task(conn, task_id).current_run_id
        assert kb.complete_task(
            conn, task_id, result="Implementation delivered", summary="Original full handoff",
            metadata={"checks": 3, "artifacts": [str(artifact)]}, expected_run_id=run_id,
        )
        completed = kb.get_task(conn, task_id)
        original = kb.latest_run(conn, task_id)
        original_row = dict(conn.execute("SELECT * FROM task_runs WHERE id = ?", (original.id,)).fetchone())
        attachments = kb.list_attachments(conn, task_id)
        assert not workspace.exists()
        assert attachments and Path(attachments[0].stored_path).read_text() == "verified output"
        # The current assignment must not replace the durable implementer identity.
        assert kb.assign_task(conn, task_id, "reviewer")

    refused = kc.run_slash(f"request-review {task_id} --force --reviewer reviewer")
    assert "cannot request review" in refused
    output = kc.run_slash(f"request-review {task_id} --from-done --reviewer reviewer")
    assert "Requested review" in output, output

    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert (task.status, task.assignee, task.result) == ("review", "reviewer", completed.result)
        assert task.completed_at is None
        assert dict(conn.execute("SELECT * FROM task_runs WHERE id = ?", (original.id,)).fetchone()) == original_row
        assert kb.list_attachments(conn, task_id) == attachments
        handoff = kb.latest_run(conn, task_id)
        assert handoff.id != original.id
        assert (handoff.summary, handoff.metadata) == (original.summary, original.metadata)
        event = [e for e in kb.list_events(conn, task_id) if e.kind == "review_requested"][-1]
        assert event.payload["previous_status"] == "done"
        assert event.payload["previous_completed_at"] == completed.completed_at
        assert event.payload["completed_run_id"] == original.id
        assert event.payload["implementer"] == "builder"
        review = kb.claim_review_task(conn, task_id)
        assert review is not None
        assert kb.request_changes(
            conn, task_id, reason="Add a boundary case", expected_run_id=review.current_run_id,
        ) == (True, "builder")
        assert kb.get_task(conn, task_id).assignee == "builder"


@pytest.mark.parametrize("refusal", [None, "opt_in", "run_id", "parent", "artifact"])
def test_done_review_is_atomic_and_retracts_dependents_after_commit(kanban_home, monkeypatch, refusal):
    with kbc.connect() as conn:
        parent_id = kb.create_task(conn, title="Prerequisite", assignee="planner")
        assert kb.complete_task(conn, parent_id, result="Ready")
        task_id = kb.create_task(conn, title="Review me", assignee="builder", parents=[parent_id])
        workspace = kbw.resolve_workspace(kb.get_task(conn, task_id))
        kbw.set_workspace_path(conn, task_id, workspace)
        assert kb.complete_task(conn, task_id, result="Delivered")
        child_id = kb.create_task(conn, title="Using that result", assignee="consumer", parents=[task_id])
        child_run = kb.claim_task(conn, child_id)
        kbd._set_worker_pid(conn, child_id, 424242)
        done_child = kb.create_task(conn, title="Already used that result", parents=[task_id])
        assert kb.complete_task(conn, done_child, result="Consumed")
        if refusal == "parent":
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (parent_id,))
        tables = ("tasks", "task_runs", "task_events", "task_attachments")
        before = {table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id")] for table in tables}
        kills = []

        def terminate(pid, claim_lock, *, started_at=None):
            # A separate connection must see both changes before a worker is killed.
            with kbc.connect() as other:
                assert kb.get_task(other, task_id).status == "review"
                assert kb.get_task(other, child_id).status == "todo"
                assert any(e.kind == "descendant_invalidated" for e in kb.list_events(other, child_id))
            kills.append(pid)

        monkeypatch.setattr(kb, "_terminate_reclaimed_worker", terminate)
        kwargs = {"from_done": refusal != "opt_in", "reviewer": "reviewer", "with_reason": True}
        if refusal == "run_id":
            kwargs["expected_run_id"] = child_run.current_run_id
        if refusal == "artifact":
            kwargs["metadata"] = {"artifacts": [str(workspace / "missing.txt")]}
            with pytest.raises(kb.ArtifactPreservationError):
                kbr.request_review(conn, task_id, **kwargs)
        else:
            ok, reason = kbr.request_review(conn, task_id, **kwargs)
            assert ok is (refusal is None), reason
        if refusal is not None:
            assert {table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id")] for table in tables} == before
            assert kills == []
        else:
            assert kills == [424242]
            assert kb.get_task(conn, done_child).status == "todo"
            assert kb.get_task(conn, done_child).completed_at is None
            assert kb.latest_run(conn, child_id).outcome == "reclaimed"
            kb.recompute_ready(conn)
            assert kb.get_task(conn, child_id).status == "todo"
            assert kb.get_task(conn, done_child).status == "todo"
