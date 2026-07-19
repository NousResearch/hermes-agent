import subprocess
from pathlib import Path

from hermes_cli import kanban_db as kb


def _run(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def test_review_handoff_does_not_recur_unrelated_block():
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="candidate", assignee="worker")
        assert kb.claim_task(conn, task_id) is not None
        assert kb.block_task(conn, task_id, reason="provisioning", kind="capability")
        assert kb.unblock_task(conn, task_id)
        assert kb.claim_task(conn, task_id) is not None

        assert kb.block_task(
            conn,
            task_id,
            reason="review-required: https://github.com/acme/repo/pull/1 @ abc123",
            kind="review_required",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"
        assert task.block_kind == "review_required"
        assert task.block_recurrences == 1
        assert not any(e.kind == "block_loop_detected" for e in kb.list_events(conn, task_id))
    finally:
        conn.close()


def test_same_unresolved_block_still_escalates():
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="blocked", assignee="worker")
        assert kb.claim_task(conn, task_id) is not None
        assert kb.block_task(conn, task_id, reason="need token", kind="capability")
        assert kb.unblock_task(conn, task_id)
        assert kb.claim_task(conn, task_id) is not None
        assert kb.block_task(conn, task_id, reason="need token", kind="capability")
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "triage"
        assert any(e.kind == "block_loop_detected" for e in kb.list_events(conn, task_id))
    finally:
        conn.close()


def test_missing_forced_skill_rejected_once_before_run(
    all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kb, "_available_profile_skills", lambda _profile: {"present"})
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn, title="skill gate", assignee="worker", skills=["missing"],
        )
        first = kb.dispatch_once(conn, failure_limit=1)
        second = kb.dispatch_once(conn, failure_limit=1)

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"
        assert task.current_run_id is None
        assert kb.list_runs(conn, task_id) == []
        assert first.auto_blocked == [task_id]
        assert second.auto_blocked == []
        events = kb.list_events(conn, task_id)
        rejected = [e for e in events if e.kind == "dispatch_admission_rejected"]
        assert len(rejected) == 1
        assert rejected[0].payload is not None
        assert rejected[0].payload["code"] == "missing_forced_skills"
        assert rejected[0].run_id is None
    finally:
        conn.close()


def test_duplicate_mutable_workspace_rejected_but_scratch_is_safe(
    all_assignees_spawnable, tmp_path,
):
    conn = kb.connect()
    try:
        shared = tmp_path / "shared"
        owner = kb.create_task(
            conn, title="owner", assignee="worker",
            workspace_kind="dir", workspace_path=str(shared),
        )
        contender = kb.create_task(
            conn, title="contender", assignee="worker",
            workspace_kind="dir", workspace_path=str(shared),
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='running' WHERE id=?", (owner,))
        result = kb.dispatch_once(conn)
        task = kb.get_task(conn, contender)
        assert task is not None
        assert task.status == "blocked"
        assert contender in result.auto_blocked
        event = next(
            e for e in kb.list_events(conn, contender)
            if e.kind == "dispatch_admission_rejected"
        )
        assert event.payload is not None
        assert event.payload["code"] == "workspace_owned"

        scratch = kb.get_task(
            conn, kb.create_task(conn, title="scratch", assignee="worker")
        )
        assert scratch is not None
        scratch_path = kb.resolve_workspace(scratch)
        assert kb._validate_mutable_workspace(conn, scratch, scratch_path, None) is None
    finally:
        conn.close()


def test_respawn_guard_events_are_bounded_and_changes_are_visible(
    all_assignees_spawnable,
):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="guard", assignee="worker")
        kb.add_comment(
            conn, task_id, "worker",
            "PR https://github.com/acme/repo/pull/1 head aaaaaaa",
        )
        for _ in range(100):
            result = kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)
            assert result.respawn_guarded == [(task_id, "active_pr")]
        guarded = [e for e in kb.list_events(conn, task_id) if e.kind == "respawn_guarded"]
        assert len(guarded) == 1
        state = conn.execute(
            "SELECT skipped_count FROM respawn_guard_state WHERE task_id=?", (task_id,),
        ).fetchone()
        assert state is not None
        assert state["skipped_count"] == 100

        kb.add_comment(
            conn, task_id, "worker",
            "updated PR https://github.com/acme/repo/pull/1 head bbbbbbb",
        )
        kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)
        guarded = [e for e in kb.list_events(conn, task_id) if e.kind == "respawn_guarded"]
        assert len(guarded) == 2
        assert guarded[-1].payload is not None
        assert guarded[-1].payload["identity"].endswith("@bbbbbbb")

        with kb.write_txn(conn):
            conn.execute("DELETE FROM task_comments WHERE task_id=?", (task_id,))
        kb.dispatch_once(conn, spawn_fn=lambda *_args: 123)
        assert any(
            e.kind == "respawn_guard_cleared" for e in kb.list_events(conn, task_id)
        )
    finally:
        conn.close()


def test_failure_limit_one_is_recorded_and_gave_up_is_run_bound(
    all_assignees_spawnable,
):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="one attempt", assignee="worker")

        def fail_spawn(*_args):
            raise RuntimeError("deterministic spawn failure")

        result = kb.dispatch_once(conn, spawn_fn=fail_spawn, failure_limit=1)
        assert result.auto_blocked == [task_id]
        events = kb.list_events(conn, task_id)
        claimed = next(e for e in events if e.kind == "claimed")
        gave_up = next(e for e in events if e.kind == "gave_up")
        assert claimed.payload is not None
        assert gave_up.payload is not None
        assert claimed.payload["effective_failure_limit"] == 1
        assert claimed.payload["failure_limit_source"] == "dispatcher"
        assert claimed.run_id is not None
        assert gave_up.run_id == claimed.run_id
        assert gave_up.payload["effective_limit"] == 1
    finally:
        conn.close()


def test_timeout_and_gave_up_events_share_the_real_run(monkeypatch):
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn, title="timeout", assignee="worker", max_runtime_seconds=1,
        )
        assert kb.claim_task(conn, task_id) is not None
        task = kb.get_task(conn, task_id)
        assert task is not None
        run_id = task.current_run_id
        assert run_id is not None
        kb._set_worker_pid(conn, task_id, 424242)
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET started_at=0 WHERE id=?", (run_id,))
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

        assert kb.enforce_max_runtime(
            conn, signal_fn=lambda *_args: None, failure_limit=1,
        ) == [task_id]
        events = kb.list_events(conn, task_id)
        timed_out = next(e for e in events if e.kind == "timed_out")
        gave_up = next(e for e in events if e.kind == "gave_up")
        assert timed_out.run_id == run_id
        assert gave_up.run_id == run_id
    finally:
        conn.close()


def test_crash_and_gave_up_events_share_the_real_run(monkeypatch):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="crash", assignee="worker")
        assert kb.claim_task(conn, task_id) is not None
        task = kb.get_task(conn, task_id)
        assert task is not None
        run_id = task.current_run_id
        assert run_id is not None
        kb._set_worker_pid(conn, task_id, 434343)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET started_at=0 WHERE id=?", (task_id,))
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)
        monkeypatch.setattr(kb, "_classify_worker_exit", lambda _pid: ("nonzero_exit", 1))

        assert kb.detect_crashed_workers(conn, failure_limit=1) == [task_id]
        events = kb.list_events(conn, task_id)
        crashed = next(e for e in events if e.kind == "crashed")
        gave_up = next(e for e in events if e.kind == "gave_up")
        assert crashed.run_id == run_id
        assert gave_up.run_id == run_id
    finally:
        conn.close()


def test_project_worktree_starts_from_fetched_remote_default(
    all_assignees_spawnable, tmp_path,
):
    remote = tmp_path / "origin.git"
    primary = tmp_path / "primary"
    updater = tmp_path / "updater"
    _run("git", "init", "--bare", str(remote), cwd=tmp_path)
    _run("git", "clone", str(remote), str(primary), cwd=tmp_path)
    _run("git", "config", "user.email", "test@example.com", cwd=primary)
    _run("git", "config", "user.name", "Test", cwd=primary)
    _run("git", "switch", "-c", "main", cwd=primary)
    (primary / "base.txt").write_text("base\n", encoding="utf-8")
    _run("git", "add", "base.txt", cwd=primary)
    _run("git", "commit", "-m", "base", cwd=primary)
    _run("git", "push", "-u", "origin", "main", cwd=primary)
    _run("git", "symbolic-ref", "HEAD", "refs/heads/main", cwd=remote)

    _run("git", "clone", str(remote), str(updater), cwd=tmp_path)
    _run("git", "config", "user.email", "test@example.com", cwd=updater)
    _run("git", "config", "user.name", "Test", cwd=updater)
    (updater / "remote.txt").write_text("remote\n", encoding="utf-8")
    _run("git", "add", "remote.txt", cwd=updater)
    _run("git", "commit", "-m", "remote update", cwd=updater)
    _run("git", "push", "origin", "main", cwd=updater)
    remote_head = _run("git", "rev-parse", "HEAD", cwd=updater)

    _run("git", "switch", "-c", "dirty-local", cwd=primary)
    (primary / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn, title="project", assignee="worker", workspace_kind="worktree",
            workspace_path=str(primary), branch_name="project/task",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET project_id='project' WHERE id=?", (task_id,))
        task = kb.get_task(conn, task_id)
        assert task is not None
        workspace, branch = kb._resolve_worktree_workspace(task)
        assert branch == "project/task"
        assert _run("git", "rev-parse", "HEAD", cwd=workspace) == remote_head
        assert _run("git", "branch", "--show-current", cwd=workspace) == "project/task"
        assert (primary / "dirty.txt").exists()
    finally:
        conn.close()
