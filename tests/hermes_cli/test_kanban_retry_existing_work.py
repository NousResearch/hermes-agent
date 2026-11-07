"""Retry partial work without inferring completion from comments (#147231, #67249)."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


@pytest.mark.parametrize("comment", [
    "Finished two outputs: https://github.com/example/repo/pull/1 and "
    "https://github.com/example/repo/pull/2; two outputs remain.",
    "Related work by another task: https://github.com/other/repo/pull/3",
])
def test_retry_reuses_partial_work_and_still_obeys_claim_and_failure_limits(
    tmp_path, monkeypatch, all_assignees_spawnable, comment,
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    kb.init_db()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = workspace / "first-output.txt"
    output.write_text("completed output\n", encoding="utf-8")

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title="Publish four independent outputs", assignee="worker",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        first = kb.claim_task(conn, task_id)
        assert first is not None
        kb.add_comment(conn, task_id, author="worker", body=comment)
        assert not kbd._record_task_failure(
            conn, task_id, "Iteration budget exhausted; unfinished outputs remain",
            outcome="timed_out", failure_limit=2, release_claim=True, end_run=True,
        )

        spawned = []

        def spawn(task, path):
            spawned.append((task.id, path, kb.build_worker_context(conn, task.id)))

        result = kbd.dispatch_once(conn, spawn_fn=spawn, failure_limit=2)
        assert [row[0] for row in result.spawned] == [task_id]
        assert result.respawn_guarded == []
        assert len(spawned) == 1
        assert spawned[0][1] == str(workspace)
        assert comment in spawned[0][2]
        assert "Iteration budget exhausted" in spawned[0][2]
        assert output.read_text(encoding="utf-8") == "completed output\n"
        retry = kb.get_task(conn, task_id)
        assert retry.current_run_id != first.current_run_id
        assert retry.consecutive_failures == 1

        # A second tick cannot start another worker while this run owns the task.
        assert kbd.dispatch_once(conn, spawn_fn=spawn).spawned == []
        assert kb.claim_task(conn, task_id) is None
        assert len(spawned) == 1

        # Another failed attempt exhausts the original budget, despite the PR links.
        assert kbd._record_task_failure(
            conn, task_id, "Worker crashed", outcome="crashed", failure_limit=2,
            release_claim=True, end_run=True,
        )
        blocked = kb.get_task(conn, task_id)
        assert blocked.status == "blocked"
        assert blocked.consecutive_failures == 2
        assert kbd.dispatch_once(conn, spawn_fn=spawn, failure_limit=2).spawned == []
        assert len(spawned) == 1


@pytest.mark.parametrize("max_retries", [None, 1])
def test_reclaimed_delivery_obeys_lifecycle_and_resumes_the_same_workspace(
    tmp_path, monkeypatch, all_assignees_spawnable, max_retries,
):
    """Exercise the missing-terminal-call path raised in the review of #70240."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kbd, "_recent_worker_exits", {})
    kb.init_db()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = workspace / "delivered.txt"
    output.write_text("existing work\n", encoding="utf-8")
    comment = "Delivered part of the task: https://github.com/example/repo/pull/42"

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title="Finish delivery and follow-up", assignee="worker",
            workspace_kind="dir", workspace_path=str(workspace), max_retries=max_retries,
        )
        first = kb.claim_task(conn, task_id)
        assert first is not None
        kb.add_comment(conn, task_id, author="worker", body=comment)
        kbd._set_worker_pid(conn, task_id, 987654)
        log = kb.worker_log_path(task_id)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"Published partial work\n{KANBAN_WORKER_EXIT_TRAILER}0\n", encoding="utf-8")

        # A clean exit without a lifecycle call is reclaimed and budgeted as a violation.
        assert kbd.detect_crashed_workers(conn) == [task_id]
        reclaimed = kb.get_task(conn, task_id)
        assert reclaimed.status == ("blocked" if max_retries == 1 else "ready")
        assert kbd._protocol_violation_streak(conn, task_id) == 1
        assert reclaimed.claim_lock is None
        assert reclaimed.worker_pid is None
        if max_retries is None:
            assert kb.block_task(conn, task_id, reason="Awaiting review of delivered work")

        spawned = []

        def spawn(task, path):
            spawned.append((task.id, path, kb.build_worker_context(conn, task.id)))

        assert kbd.dispatch_once(conn, spawn_fn=spawn).spawned == []
        assert kb.get_task(conn, task_id).status == "blocked"
        assert spawned == []

        assert kb.unblock_task(conn, task_id)
        resumed = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert [row[0] for row in resumed.spawned] == [task_id]
        assert resumed.respawn_guarded == []
        assert len(spawned) == 1
        assert spawned[0][:2] == (task_id, str(workspace))
        assert comment in spawned[0][2]
        assert "Published partial work" in spawned[0][2]
        assert output.read_text(encoding="utf-8") == "existing work\n"
        assert kb.get_task(conn, task_id).current_run_id != first.current_run_id
        assert kbd.dispatch_once(conn, spawn_fn=spawn).spawned == []
        assert kb.claim_task(conn, task_id) is None
        assert len(spawned) == 1
