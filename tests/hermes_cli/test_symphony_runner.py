from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest
import subprocess
import sys
import time

from hermes_cli import symphony


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setattr(symphony, "get_hermes_home", lambda: tmp_path)


def test_legacy_agent_runs_unique_index_migrates_without_losing_history(monkeypatch, tmp_path):
    db = tmp_path / "symphony" / "symphony.db"
    db.parent.mkdir(parents=True)
    raw = __import__("sqlite3").connect(db)
    raw.executescript("""
        CREATE TABLE agent_runs (
          run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, kanban_task_id TEXT UNIQUE,
          workspace_path TEXT, branch_name TEXT, worker_pid INTEGER, status TEXT NOT NULL,
          started_at INTEGER NOT NULL, finished_at INTEGER, last_progress_at INTEGER NOT NULL,
          result_artifact TEXT, test_summary TEXT, error TEXT
        );
        INSERT INTO agent_runs VALUES ('legacy_old', 'task_legacy', 'kanban_legacy', NULL, NULL, NULL, 'paused', 1, NULL, 1, NULL, NULL, NULL);
    """)
    raw.commit()
    raw.close()

    monkeypatch.setattr(symphony, "get_hermes_home", lambda: tmp_path)
    with symphony.connect() as conn:
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, kanban_task_id, status, started_at, last_progress_at) VALUES ('legacy_new', 'task_legacy', 'kanban_legacy', 'running', 2, 2)"
        )
        rows = conn.execute("SELECT run_id, status FROM agent_runs WHERE task_id=? ORDER BY started_at", ("task_legacy",)).fetchall()
        indexes = [
            (row["unique"], [col["name"] for col in conn.execute(f"PRAGMA index_info('{row['name']}')").fetchall()])
            for row in conn.execute("PRAGMA index_list(agent_runs)").fetchall()
        ]

    assert [(row["run_id"], row["status"]) for row in rows] == [("legacy_old", "paused"), ("legacy_new", "running")]
    assert not any(unique and "kanban_task_id" in columns for unique, columns in indexes)


def test_concurrent_legacy_agent_runs_migration_is_consistent(monkeypatch, tmp_path):
    db = tmp_path / "symphony" / "symphony.db"
    db.parent.mkdir(parents=True)
    raw = __import__("sqlite3").connect(db)
    raw.executescript("""
        CREATE TABLE agent_runs (
          run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, kanban_task_id TEXT UNIQUE,
          workspace_path TEXT, branch_name TEXT, worker_pid INTEGER, status TEXT NOT NULL,
          started_at INTEGER NOT NULL, finished_at INTEGER, last_progress_at INTEGER NOT NULL,
          result_artifact TEXT, test_summary TEXT, error TEXT
        );
        INSERT INTO agent_runs VALUES ('legacy_concurrent', 'task_concurrent', 'kanban_concurrent', NULL, NULL, NULL, 'paused', 1, NULL, 1, NULL, NULL, NULL);
    """)
    raw.commit()
    raw.close()
    monkeypatch.setattr(symphony, "get_hermes_home", lambda: tmp_path)

    def open_and_read():
        with symphony.connect() as conn:
            return conn.execute("SELECT COUNT(*) AS count FROM agent_runs").fetchone()["count"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: open_and_read(), range(2)))
    with symphony.connect() as conn:
        rows = conn.execute("SELECT run_id FROM agent_runs").fetchall()
        indexes = [
            (row["unique"], [col["name"] for col in conn.execute(f"PRAGMA index_info('{row['name']}')").fetchall()])
            for row in conn.execute("PRAGMA index_list(agent_runs)").fetchall()
        ]
    assert results == [1, 1]
    assert [row["run_id"] for row in rows] == ["legacy_concurrent"]
    assert not any(unique and "kanban_task_id" in columns for unique, columns in indexes)


def test_runner_dispatches_ready_task_once_and_records_agent_run(monkeypatch):
    task = symphony.create_task(
        title="Тестовый runner", source_text="Проверить controlled dispatch",
        project_id="project-a",
    )
    symphony.mark_ready(task.task_id)

    monkeypatch.setattr(symphony.kb, "create_board", lambda *args, **kwargs: None)
    monkeypatch.setattr(symphony.kb, "create_task", lambda *args, **kwargs: "k_test")
    monkeypatch.setattr(
        symphony.kb,
        "get_task",
        lambda *args, **kwargs: type("Task", (), {"workspace_path": "C:/repo/.worktrees/k_test", "branch_name": "project-a/k_test", "status": "ready", "result": None})(),
    )

    queued = symphony.enqueue(task.task_id)
    assert queued.status == "queued"

    calls = []

    def fake_dispatch(conn, *, board, spawn_fn, max_spawn, **kwargs):
        calls.append((board, max_spawn))
        worker_pid = spawn_fn(
            type("Task", (), {"id": "k_test"})(),
            "C:/repo/.worktrees/k_test",
            board=board,
        )
        assert worker_pid == 4242
        return type("Result", (), {"spawned": ["k_test"]})()

    monkeypatch.setattr(symphony.kb, "dispatch_once", fake_dispatch)

    runner = symphony.SymphonyRunner(max_concurrency=1)
    dispatched = runner.run_once(spawn_fn=lambda *_: 4242)

    assert dispatched == [task.task_id]
    assert calls == [("development", 1)]
    run = symphony.latest_agent_run(task.task_id)
    assert run is not None
    assert run.worker_pid == 4242
    assert run.workspace_path == "C:/repo/.worktrees/k_test"


def test_cancelled_task_is_never_dispatched(monkeypatch):
    task = symphony.create_task(title="Отмена", source_text="Не запускать", project_id="project-a")
    symphony.mark_ready(task.task_id)
    symphony.cancel(task.task_id)

    runner = symphony.SymphonyRunner(max_concurrency=1)
    assert runner.run_once(spawn_fn=lambda *_: pytest.fail("must not spawn")) == []


def test_completion_requires_real_evidence_and_moves_to_review(monkeypatch, tmp_path):
    task = symphony.create_task(title="Evidence", source_text="Run tests", project_id="project-a")
    symphony.mark_ready(task.task_id)
    monkeypatch.setattr(symphony.kb, "create_board", lambda *args, **kwargs: None)
    monkeypatch.setattr(symphony.kb, "create_task", lambda *args, **kwargs: "k_evidence")
    monkeypatch.setattr(symphony.kb, "get_task", lambda *args, **kwargs: type("Task", (), {
        "workspace_path": "C:/repo/.worktrees/k_evidence", "branch_name": "project-a/k_evidence",
        "status": "ready", "result": None,
    })())
    symphony.enqueue(task.task_id)
    def fake_dispatch(conn, *, board, spawn_fn, max_spawn, **kwargs):
        spawn_fn(type("Task", (), {"id": "k_evidence"})(), "C:/repo/.worktrees/k_evidence", board=board)
        return type("Result", (), {"spawned": ["k_evidence"]})()
    monkeypatch.setattr(symphony.kb, "dispatch_once", fake_dispatch)
    symphony.SymphonyRunner().run_once(spawn_fn=lambda *_: 4242)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with symphony.connect() as conn:
        conn.execute("UPDATE development_tasks SET workspace_path=? WHERE task_id=?", (str(workspace), task.task_id))
    symphony.write_result_artifact(
        workspace, task_id=task.task_id, changed_files=[],
        tests=[{"command": "pytest", "exit_code": 0, "summary": "1 passed"}],
        summary="verified", limitations=[],
    )
    evidence_path = str(workspace / ".symphony" / f"{task.task_id}-result.json")

    with pytest.raises(ValueError, match="test_summary"):
        symphony.complete_agent_run(task.task_id, test_summary="", result_summary="x")
    completed = symphony.complete_agent_run(
        task.task_id, test_summary="pytest: 1 passed", result_summary="Изменение проверено",
        evidence_path=evidence_path,
    )
    assert completed.status == "awaiting_human_review"
    assert symphony.latest_agent_run(task.task_id).status == "completed"


def test_completion_persists_canonical_artifact_when_evidence_path_is_omitted(tmp_path):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    task = symphony.create_task(title="Canonical artifact", source_text="Verify", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', workspace_path=? WHERE task_id=?",
            (str(workspace), task.task_id),
        )
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, status, started_at, last_progress_at) VALUES ('run_canonical_artifact', ?, 'running', 1, 1)",
            (task.task_id,),
        )
    artifact = symphony.write_result_artifact(
        workspace, task_id=task.task_id, changed_files=[],
        tests=[{"command": "pytest", "exit_code": 0, "summary": "1 passed"}],
        summary="verified", limitations=[],
    )

    symphony.complete_agent_run(
        task.task_id, test_summary="pytest: 1 passed", result_summary="Verified",
    )

    assert symphony.latest_agent_run(task.task_id).result_artifact == str(artifact)


def test_completion_rejects_evidence_outside_the_task_workspace(tmp_path):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    task = symphony.create_task(title="Bound evidence", source_text="Verify", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', workspace_path=? WHERE task_id=?",
            (str(workspace), task.task_id),
        )
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, status, started_at, last_progress_at) VALUES ('run_bound', ?, 'running', 1, 1)",
            (task.task_id,),
        )
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="inside the task workspace|result artifact"):
        symphony.complete_agent_run(
            task.task_id, test_summary="pytest: 1 passed", result_summary="Verified", evidence_path=str(outside),
        )


def test_completion_rejects_missing_or_invalid_evidence_artifact(tmp_path):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    task = symphony.create_task(title="Strict evidence", source_text="Verify", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', workspace_path=? WHERE task_id=?",
            (str(workspace), task.task_id),
        )
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, status, started_at, last_progress_at) VALUES ('run_strict_evidence', ?, 'running', 1, 1)",
            (task.task_id,),
        )

    with pytest.raises(ValueError, match="artifact"):
        symphony.complete_agent_run(
            task.task_id,
            test_summary="pytest: 1 passed",
            result_summary="Verified",
            evidence_path=str(workspace / ".symphony" / f"{task.task_id}-result.json"),
        )

    root = workspace / ".symphony"
    root.mkdir()
    (root / f"{task.task_id}-result.json").write_text('{"task_id":"wrong"}', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact"):
        symphony.complete_agent_run(
            task.task_id,
            test_summary="pytest: 1 passed",
            result_summary="Verified",
            evidence_path=str(root / f"{task.task_id}-result.json"),
        )


def test_completion_rejects_result_artifact_symlink_outside_workspace(tmp_path):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    task = symphony.create_task(title="Symlink evidence", source_text="Verify", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', workspace_path=? WHERE task_id=?",
            (str(workspace), task.task_id),
        )
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, status, started_at, last_progress_at) VALUES ('run_symlink', ?, 'running', 1, 1)",
            (task.task_id,),
        )
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"task_id": task.task_id}), encoding="utf-8")
    artifact_dir = workspace / ".symphony"
    artifact_dir.mkdir()
    try:
        (artifact_dir / f"{task.task_id}-result.json").symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    with pytest.raises(ValueError, match="outside the workspace"):
        symphony.complete_agent_run(
            task.task_id, test_summary="pytest: 1 passed", result_summary="Verified",
            evidence_path=str(artifact_dir / f"{task.task_id}-result.json"),
        )


def test_completion_rejects_artifact_without_passing_test_exit_code(tmp_path):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    task = symphony.create_task(title="Failed evidence", source_text="Verify tests", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', workspace_path=? WHERE task_id=?",
            (str(workspace), task.task_id),
        )
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, status, started_at, last_progress_at) VALUES ('run_failed_evidence', ?, 'running', 1, 1)",
            (task.task_id,),
        )
    artifact = workspace / ".symphony" / f"{task.task_id}-result.json"
    artifact.parent.mkdir()
    artifact.write_text(
        '{"task_id": "' + task.task_id + '", "changed_files": [], '
        '"tests": [{"command": "pytest", "exit_code": 1}], '
        '"summary": "not verified", "limitations": [], "created_at": 1}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="test evidence"):
        symphony.complete_agent_run(
            task.task_id,
            test_summary="pytest: failed",
            result_summary="Not verified",
            evidence_path=str(workspace / ".symphony" / f"{task.task_id}-result.json"),
        )


def test_completion_rejects_empty_test_evidence(tmp_path):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    task = symphony.create_task(title="Empty evidence", source_text="Verify tests", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute("UPDATE development_tasks SET status='running', workspace_path=? WHERE task_id=?", (str(workspace), task.task_id))
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, status, started_at, last_progress_at) VALUES ('run_empty_evidence', ?, 'running', 1, 1)",
            (task.task_id,),
        )
    artifact = workspace / ".symphony" / f"{task.task_id}-result.json"
    artifact.parent.mkdir()
    artifact.write_text(
        '{"task_id": "' + task.task_id + '", "changed_files": [], '
        '"tests": [], "summary": "done", "limitations": [], "created_at": 1}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="test evidence"):
        symphony.complete_agent_run(
            task.task_id,
            test_summary="tests claimed passed",
            result_summary="Done",
            evidence_path=str(workspace / ".symphony" / f"{task.task_id}-result.json"),
        )


def test_stale_agent_run_is_recovered_without_duplicate_dispatch(monkeypatch):
    task = symphony.create_task(title="Recovery", source_text="Recover", project_id="project-a")
    symphony.mark_ready(task.task_id)
    monkeypatch.setattr(symphony.kb, "create_board", lambda *args, **kwargs: None)
    monkeypatch.setattr(symphony.kb, "create_task", lambda *args, **kwargs: "k_recovery")
    monkeypatch.setattr(symphony.kb, "get_task", lambda *args, **kwargs: type("Task", (), {
        "workspace_path": "C:/repo/.worktrees/k_recovery", "branch_name": "project-a/k_recovery",
        "status": "ready", "result": None,
    })())
    symphony.enqueue(task.task_id)
    def fake_dispatch(conn, *, board, spawn_fn, max_spawn, **kwargs):
        spawn_fn(type("Task", (), {"id": "k_recovery"})(), "C:/repo/.worktrees/k_recovery", board=board)
        return type("Result", (), {"spawned": ["k_recovery"]})()
    monkeypatch.setattr(symphony.kb, "dispatch_once", fake_dispatch)
    symphony.SymphonyRunner().run_once(spawn_fn=lambda *_: 4343)
    with symphony.connect() as conn:
        conn.execute("UPDATE agent_runs SET last_progress_at=?", (0,))
    recovered = symphony.reconcile_agent_runs(stale_after_seconds=1)
    assert recovered[0].status == "failed"
    assert symphony.latest_agent_run(task.task_id).status == "failed"


def test_stale_agent_run_terminates_only_recorded_worker(monkeypatch):
    task = symphony.create_task(title="Worker timeout", source_text="Recover worker", project_id="project-a")
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, worker_pid, status, started_at, last_progress_at) VALUES (?, ?, ?, 'running', ?, ?)",
            ("run_stale", task.task_id, 4567, 1, 0),
        )
        conn.execute("UPDATE development_tasks SET status='running' WHERE task_id=?", (task.task_id,))
    terminated = []
    monkeypatch.setattr(symphony, "_terminate_worker_process", lambda pid: terminated.append(pid))
    recovered = symphony.reconcile_agent_runs(stale_after_seconds=1)
    assert [item.task_id for item in recovered] == [task.task_id]
    assert terminated == [4567]
    assert symphony.latest_agent_run(task.task_id).status == "failed"


def test_concurrent_runner_claims_one_queued_task(monkeypatch):
    task = symphony.create_task(title="Concurrent claim", source_text="One worker only", project_id="project-a")
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute("UPDATE development_tasks SET status='queued', kanban_task_id='k_claim', kanban_board='development', workspace_path='C:/repo/.worktrees/k_claim' WHERE task_id=?", (task.task_id,))
    spawned = []

    def fake_spawn(*_args):
        spawned.append(1)
        return 3003

    def fake_dispatch(conn, *, board, spawn_fn, max_spawn, **kwargs):
        spawn_fn(type("Task", (), {"id": "k_claim"})(), "C:/repo/.worktrees/k_claim", board=board)
        return type("Result", (), {"spawned": ["k_claim"]})()

    monkeypatch.setattr(symphony.kb, "dispatch_once", fake_dispatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: symphony.SymphonyRunner().run_once(spawn_fn=fake_spawn), range(2)))
    with symphony.connect() as conn:
        runs = conn.execute("SELECT status FROM agent_runs WHERE task_id=?", (task.task_id,)).fetchall()
    assert sum(bool(result) for result in results) == 1
    assert len(spawned) == 1
    assert [row["status"] for row in runs] == ["running"]


def test_pause_resume_dispatch_fences_old_run(monkeypatch):
    task = symphony.create_task(title="Resume dispatch", source_text="Fence retries", project_id="project-a")
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='validating', kanban_task_id='k_retry', kanban_board='development', workspace_path='C:/repo/.worktrees/k_retry', branch_name='project-a/k_retry' WHERE task_id=?",
            (task.task_id,),
        )
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, kanban_task_id, worker_pid, status, started_at, last_progress_at) VALUES (?, ?, ?, ?, 'validating', ?, ?)",
            ("run_old_retry", task.task_id, "k_retry", 1001, 1, 1),
        )
    monkeypatch.setattr(symphony.kb, "stop_task_for_user", lambda *args, **kwargs: True)
    monkeypatch.setattr(symphony.kb, "unblock_task", lambda *args, **kwargs: True)
    paused = symphony.pause(task.task_id)
    assert paused.status == "paused"
    symphony.resume(task.task_id)

    def fake_dispatch(conn, *, board, spawn_fn, max_spawn, **kwargs):
        spawned = spawn_fn(type("Task", (), {"id": "k_retry"})(), "C:/repo/.worktrees/k_retry", board=board)
        assert spawned == 2002
        return type("Result", (), {"spawned": ["k_retry"]})()

    monkeypatch.setattr(symphony.kb, "dispatch_once", fake_dispatch)
    dispatched = symphony.SymphonyRunner(max_concurrency=1).run_once(spawn_fn=lambda *_: 2002)
    assert dispatched == [task.task_id]
    with symphony.connect() as conn:
        runs = conn.execute("SELECT run_id, status, worker_pid FROM agent_runs WHERE task_id=? ORDER BY started_at", (task.task_id,)).fetchall()
    assert [(row["run_id"], row["status"], row["worker_pid"]) for row in runs] == [
        ("run_old_retry", "paused", 1001),
        (runs[-1]["run_id"], "running", 2002),
    ]
    assert runs[-1]["run_id"] != "run_old_retry"
    with pytest.raises(ValueError, match="no longer current"):
        symphony.record_progress(task.task_id, action="stale", run_id="run_old_retry")


def test_pause_validating_checks_transition_before_kanban_and_resume_queues(monkeypatch):
    task = symphony.create_task(title="Pause validating", source_text="Pause safely", project_id="project-a")
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='validating', kanban_task_id='k_pause', kanban_board='development' WHERE task_id=?",
            (task.task_id,),
        )
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, kanban_task_id, worker_pid, status, started_at, last_progress_at) VALUES (?, ?, ?, ?, 'validating', ?, ?)",
            ("run_pause", task.task_id, "k_pause", 8899, 1, 1),
        )
    calls = []
    monkeypatch.setattr(symphony.kb, "stop_task_for_user", lambda *args, **kwargs: calls.append("stop"))
    monkeypatch.setattr(symphony, "_terminate_worker_process", lambda pid: calls.append(pid))

    paused = symphony.pause(task.task_id)
    assert paused.status == "paused"
    assert calls == ["stop"]
    resumed = symphony.resume(task.task_id)
    assert resumed.status == "queued"
    assert symphony.latest_agent_run(task.task_id).run_id == "run_pause"
    assert symphony.latest_agent_run(task.task_id).status == "paused"


def test_starting_run_without_pid_recovers_without_terminating_unknown_process(monkeypatch):
    task = symphony.create_task(title="Unknown worker", source_text="Do not kill unknown", project_id="project-a")
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        now = int(__import__("time").time()) - 100
        conn.execute("UPDATE development_tasks SET status='running', updated_at=? WHERE task_id=?", (now, task.task_id))
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, kanban_task_id, status, started_at, last_progress_at) VALUES (?, ?, ?, 'starting', ?, ?)",
            ("run_no_pid", task.task_id, "kanban-no-pid", now, now),
        )
    terminated = []
    monkeypatch.setattr(symphony, "_terminate_worker_process", lambda pid: terminated.append(pid))
    recovered = symphony.reconcile_agent_runs(stale_after_seconds=1)
    assert [item.task_id for item in recovered] == [task.task_id]
    assert terminated == []
    assert symphony.latest_agent_run(task.task_id).status == "failed"
    with symphony.connect() as conn:
        row = conn.execute("SELECT payload FROM development_events WHERE task_id=? AND kind='agent.recovered' ORDER BY id DESC LIMIT 1", (task.task_id,)).fetchone()
    assert __import__("json").loads(row["payload"])["reason"] == "heartbeat_expired"


def test_orphaned_workspace_claim_recovers_without_agent_run(monkeypatch):
    task = symphony.create_task(title="Orphan claim", source_text="Recover claim", project_id="project-a")
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='provisioning_workspace', updated_at=? WHERE task_id=?",
            (0, task.task_id),
        )
    recovered = symphony.reconcile_agent_runs(stale_after_seconds=1)
    assert [item.task_id for item in recovered] == [task.task_id]
    assert recovered[0].status == "failed"
    assert symphony.latest_agent_run(task.task_id) is None
    with symphony.connect() as conn:
        row = conn.execute("SELECT kind, payload FROM development_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (task.task_id,)).fetchone()
    assert row["kind"] == "agent.recovered"
    assert __import__("json").loads(row["payload"])["reason"] == "workspace_claim_expired"


def test_validating_stale_agent_run_is_recovered_without_duplicate(monkeypatch):
    task = symphony.create_task(title="Validating recovery", source_text="Recover validation", project_id="project-a")
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='validating', kanban_task_id='k_validating' WHERE task_id=?",
            (task.task_id,),
        )
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, kanban_task_id, worker_pid, status, started_at, last_progress_at) VALUES (?, ?, ?, ?, 'validating', ?, ?)",
            ("run_validating_stale", task.task_id, "k_validating", 7788, 1, 0),
        )
    terminated = []
    monkeypatch.setattr(symphony, "_terminate_worker_process", lambda pid: terminated.append(pid))

    recovered = symphony.reconcile_agent_runs(stale_after_seconds=1)

    assert [item.task_id for item in recovered] == [task.task_id]
    assert terminated == [7788]
    assert symphony.latest_agent_run(task.task_id).status == "failed"
    with symphony.connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS count FROM agent_runs WHERE task_id=?", (task.task_id,)).fetchone()["count"]
    assert count == 1


def test_progress_from_stale_run_cannot_refresh_newer_heartbeat():
    task = symphony.create_task(title="Heartbeat", source_text="Track", project_id="project-a")
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute("UPDATE development_tasks SET status='running' WHERE task_id=?", (task.task_id,))
        conn.execute("INSERT INTO agent_runs(run_id, task_id, status, started_at, last_progress_at) VALUES (?, ?, 'failed', ?, ?)", ("run_old", task.task_id, 1, 1))
        conn.execute("INSERT INTO agent_runs(run_id, task_id, status, started_at, last_progress_at) VALUES (?, ?, 'running', ?, ?)", ("run_new", task.task_id, 2, 2))
    with pytest.raises(ValueError, match="no longer current"):
        symphony.record_progress(task.task_id, action="old", run_id="run_old")
    assert symphony.latest_agent_run(task.task_id).run_id == "run_new"


def test_reconcile_done_emits_task_completed_once(monkeypatch, tmp_path):
    task = symphony.create_task(title="Done", source_text="Canonical completion", project_id="project-a")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    symphony.write_result_artifact(
        workspace,
        task_id=task.task_id,
        changed_files=[],
        tests=[{"command": "test", "exit_code": 0, "summary": "passed"}],
        summary="merged by operator",
        limitations=[],
    )
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='awaiting_human_review', kanban_task_id='k_done', kanban_board='development', workspace_path=? WHERE task_id=?",
            (str(workspace), task.task_id),
        )
    class FakeConnection:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
    monkeypatch.setattr(symphony.kb, "connect_closing", lambda board: FakeConnection())
    monkeypatch.setattr(symphony.kb, "get_task", lambda conn, task_id: type("KanbanTask", (), {
        "status": "done", "workspace_path": str(workspace), "branch_name": None, "result": "merged by operator",
    })())

    first = symphony.reconcile(task.task_id)
    second = symphony.reconcile(task.task_id)
    assert first[0].status == "completed"
    assert second[0].status == "completed"
    with symphony.connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS count FROM development_events WHERE task_id=? AND kind='task.completed'",
            (task.task_id,),
        ).fetchone()["count"]
    assert count == 1


def test_reconcile_done_projects_running_task_to_completed(monkeypatch):
    task = symphony.create_task(title="Running done", source_text="Canonical completion", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', kanban_task_id='k_running_done', kanban_board='development' WHERE task_id=?",
            (task.task_id,),
        )

    class FakeConnection:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(symphony.kb, "connect_closing", lambda board: FakeConnection())
    monkeypatch.setattr(symphony.kb, "get_task", lambda conn, task_id: type("KanbanTask", (), {
        "status": "done", "workspace_path": None, "branch_name": None,
        "result": "worker completed the task",
    })())

    result = symphony.reconcile(task.task_id)

    assert result[0].status == "running"
    with symphony.connect() as conn:
        blocked = conn.execute(
            "SELECT COUNT(*) AS count FROM development_events WHERE task_id=? AND kind='task.completion_blocked'",
            (task.task_id,),
        ).fetchone()["count"]
    assert blocked == 1


def test_publish_for_review_commits_opens_pr_and_never_completes(monkeypatch):
    task = symphony.create_task(title="PR", source_text="Publish", project_id="project-a")
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', kanban_task_id='k_pr', kanban_board='development', workspace_path='C:/repo/.worktrees/k_pr' WHERE task_id=?",
            (task.task_id,),
        )
    Path("C:/repo/.worktrees/k_pr/.symphony").mkdir(parents=True, exist_ok=True)
    symphony.write_result_artifact("C:/repo/.worktrees/k_pr", task_id=task.task_id, changed_files=[], tests=[{"command": "pytest", "exit_code": 0, "summary": "1 passed"}], summary="Verified", limitations=[])
    commit_calls = []
    monkeypatch.setattr(symphony.web_git, "review_commit", lambda cwd, message, push: commit_calls.append((cwd, message, push)) or {"ok": True})
    monkeypatch.setattr(symphony.web_git, "review_ship_info", lambda cwd: {"ghReady": True, "pr": None})
    monkeypatch.setattr(symphony.web_git, "review_create_pr", lambda cwd: {"url": "https://github.example/pr/7"})
    review_calls = []
    monkeypatch.setattr(symphony.kb, "request_review", lambda conn, task_id, **kwargs: review_calls.append((task_id, kwargs)) or True)
    monkeypatch.setattr(symphony.kb, "complete_task", lambda *args, **kwargs: pytest.fail("must not complete"))
    monkeypatch.setattr(symphony.kb, "get_task", lambda *args, **kwargs: type("Task", (), {"current_run_id": 3})())

    result = symphony.publish_for_review(
        task.task_id, commit_message="feat: publish", test_summary="pytest: 1 passed",
        result_summary="Key=sk_abcdefghijklmnop was checked",
    )

    assert result.status == "awaiting_human_review"
    assert result.pr_url == "https://github.example/pr/7"
    assert commit_calls == [("C:/repo/.worktrees/k_pr", "feat: publish", True)]
    assert review_calls and review_calls[0][1]["expected_run_id"] == 3
    assert "sk_abcdefghijklmnop" not in result.result_summary


def test_publish_for_review_requires_real_evidence_artifact(monkeypatch, tmp_path):
    task = symphony.create_task(title="Evidence PR", source_text="Publish", project_id="project-a")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', kanban_task_id='k_evidence_pr', kanban_board='development', workspace_path=? WHERE task_id=?",
            (str(workspace), task.task_id),
        )
    monkeypatch.setattr(symphony.web_git, "review_ship_info", lambda cwd: {"ghReady": True, "pr": {"url": "https://github.example/pr/9"}})

    with pytest.raises(ValueError, match="result artifact"):
        symphony.publish_for_review(
            task.task_id, commit_message="feat: evidence", test_summary="pytest: 1 passed", result_summary="Verified",
        )


def test_answer_clarification_reopens_task_and_prepares_it(tmp_path):
    task = symphony.create_task(title="Short task", source_text="fix", project_id="project-a")
    prepared = symphony.prepare_task(task.task_id)
    assert prepared.status == "needs_clarification"

    result = symphony.answer_clarification(task.task_id, "В проекте нужно исправить обработку timeout и добавить regression test")

    assert result.status == "ready_for_development"
    assert "Уточнение пользователя" in result.source_text
    assert result.technical_brief


def test_publish_for_review_requires_ci_checks_when_snapshot_exists(monkeypatch, tmp_path):
    task = symphony.create_task(title="CI gate", source_text="Publish", project_id="project-a")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".symphony").mkdir()
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', kanban_task_id='k_ci', kanban_board='development', workspace_path=? WHERE task_id=?",
            (str(workspace), task.task_id),
        )
    symphony.write_result_artifact(workspace, task_id=task.task_id, changed_files=[], tests=[{"command": "pytest", "exit_code": 0, "summary": "1 passed"}], summary="Verified", limitations=[])
    monkeypatch.setattr(symphony.web_git, "review_ship_info", lambda cwd: {"ghReady": True, "pr": None})
    monkeypatch.setattr(symphony.web_git, "review_ci_snapshot", lambda cwd: {"state": "pending", "checks": []})
    with pytest.raises(ValueError, match="CI"):
        symphony.publish_for_review(
            task.task_id, commit_message="feat: ci", test_summary="pytest: 1 passed", result_summary="Verified",
        )


def test_publish_for_review_persists_ci_snapshot_and_pr_event(monkeypatch, tmp_path):
    task = symphony.create_task(title="CI pass", source_text="Publish", project_id="project-a")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".symphony").mkdir()
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', kanban_task_id='k_ci_pass', kanban_board='development', workspace_path=? WHERE task_id=?",
            (str(workspace), task.task_id),
        )
    symphony.write_result_artifact(workspace, task_id=task.task_id, changed_files=[], tests=[{"command": "pytest", "exit_code": 0, "summary": "1 passed"}], summary="Verified", limitations=[])
    monkeypatch.setattr(symphony.web_git, "review_ship_info", lambda cwd: {"ghReady": True, "pr": {"url": "https://github.example/pr/10"}})
    monkeypatch.setattr(symphony.web_git, "review_ci_snapshot", lambda cwd: {"state": "pass", "checks": [{"name": "tests", "state": "SUCCESS", "link": "https://ci.example/1"}]})
    monkeypatch.setattr(symphony.kb, "request_review", lambda *args, **kwargs: True)
    monkeypatch.setattr(symphony.kb, "get_task", lambda *args, **kwargs: type("Task", (), {"current_run_id": 1})())

    result = symphony.publish_for_review(task.task_id, commit_message="feat: ci", test_summary="pytest: 1 passed", result_summary="Verified")

    assert result.pr_url == "https://github.example/pr/10"
    with symphony.connect() as conn:
        row = conn.execute("SELECT ci_state, ci_checks FROM development_tasks WHERE task_id=?", (task.task_id,)).fetchone()
        kinds = [item["kind"] for item in conn.execute("SELECT kind FROM development_events WHERE task_id=?", (task.task_id,))]
    assert row["ci_state"] == "pass"
    assert "tests" in row["ci_checks"]
    assert "pr.created" in kinds


def test_reconcile_ci_snapshot_persists_changes_and_emits_delivery_event(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = symphony.create_task(title="CI polling", source_text="Track checks", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='awaiting_human_review', workspace_path=?, pr_url=? WHERE task_id=?",
            (str(workspace), "https://github.example/pr/11", task.task_id),
        )
    monkeypatch.setattr(symphony.web_git, "review_ci_snapshot", lambda cwd: {
        "state": "pass", "checks": [{"name": "tests", "state": "SUCCESS", "link": "https://ci.example/11"}],
    })

    changed = symphony.reconcile_ci(task.task_id)

    assert changed is True
    with symphony.connect() as conn:
        row = conn.execute("SELECT ci_state, ci_checks FROM development_tasks WHERE task_id=?", (task.task_id,)).fetchone()
        kinds = [item["kind"] for item in conn.execute("SELECT kind FROM development_events WHERE task_id=?", (task.task_id,))]
    assert row["ci_state"] == "pass"
    assert "tests" in row["ci_checks"]
    assert "ci.updated" in kinds
    assert symphony.claim_delivery_events(limit=100)[-1]["kind"] == "ci.updated"


def test_reconcile_ci_does_not_duplicate_unchanged_snapshot(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = symphony.create_task(title="CI stable", source_text="Track checks", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='awaiting_human_review', workspace_path=?, pr_url=? WHERE task_id=?",
            (str(workspace), "https://github.example/pr/12", task.task_id),
        )
    snapshot = {"state": "pending", "checks": []}
    monkeypatch.setattr(symphony.web_git, "review_ci_snapshot", lambda cwd: snapshot)

    assert symphony.reconcile_ci(task.task_id) is True
    assert symphony.reconcile_ci(task.task_id) is False


def test_publish_for_review_reuses_existing_pr_without_new_commit(monkeypatch):
    task = symphony.create_task(title="Existing PR", source_text="Reuse", project_id="project-a")
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', kanban_task_id='k_existing', kanban_board='development', workspace_path='C:/repo/.worktrees/k_existing' WHERE task_id=?",
            (task.task_id,),
        )
    Path("C:/repo/.worktrees/k_existing/.symphony").mkdir(parents=True, exist_ok=True)
    symphony.write_result_artifact("C:/repo/.worktrees/k_existing", task_id=task.task_id, changed_files=[], tests=[{"command": "pytest", "exit_code": 0, "summary": "1 passed"}], summary="Verified", limitations=[])
    monkeypatch.setattr(symphony.web_git, "review_commit", lambda *args, **kwargs: pytest.fail("must not commit"))
    monkeypatch.setattr(symphony.web_git, "review_create_pr", lambda *args, **kwargs: pytest.fail("must not create"))
    monkeypatch.setattr(symphony.web_git, "review_ship_info", lambda cwd: {"ghReady": True, "pr": {"url": "https://github.example/pr/8"}})
    monkeypatch.setattr(symphony.kb, "request_review", lambda *args, **kwargs: True)
    monkeypatch.setattr(symphony.kb, "get_task", lambda *args, **kwargs: type("Task", (), {"current_run_id": 4})())

    result = symphony.publish_for_review(
        task.task_id, commit_message="feat: reuse", test_summary="pytest: 1 passed", result_summary="Verified",
    )

    assert result.pr_url == "https://github.example/pr/8"
    assert result.status == "awaiting_human_review"


def test_workflow_policy_is_created_once_in_workspace(tmp_path):
    workspace = tmp_path / "worktree"
    workspace.mkdir()

    created = symphony.ensure_workflow_policy(workspace)
    content = created.read_text(encoding="utf-8")

    assert created == workspace / "WORKFLOW.md"
    assert "---" in content
    assert "не выполняй production deploy" in content
    assert "machine-readable" in content
    created.write_text("# Operator policy\n", encoding="utf-8")
    assert symphony.ensure_workflow_policy(workspace).read_text(encoding="utf-8") == "# Operator policy\n"


def test_result_artifact_is_redacted_and_bounded_to_workspace(tmp_path):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    artifact = symphony.write_result_artifact(
        workspace,
        task_id="dev_test",
        changed_files=["src/app.py"],
        tests=[{"command": "pytest -q", "exit_code": 0, "summary": "token=sk_abcdefghijklmnop"}],
        summary="Implemented safely",
        limitations=["No deploy"],
    )

    payload = __import__("json").loads(artifact.read_text(encoding="utf-8"))
    assert artifact.parent == workspace / ".symphony"
    assert payload["task_id"] == "dev_test"
    assert "sk_abcdefghijklmnop" not in artifact.read_text(encoding="utf-8")
    assert payload["tests"][0]["exit_code"] == 0


def test_publish_rejects_malformed_result_artifact(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".symphony").mkdir()
    task = symphony.create_task(title="Malformed", source_text="Verify artifact", project_id="project-a")
    symphony.mark_ready(task.task_id)
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', kanban_task_id='k_malformed', kanban_board='development', workspace_path=? WHERE task_id=?",
            (str(workspace), task.task_id),
        )
    (workspace / ".symphony" / f"{task.task_id}-result.json").write_text(
        '{"task_id":"wrong","summary":"x"}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="artifact"):
        symphony.validate_result_artifact(task.task_id, workspace)


def test_delivery_events_are_claimed_once_and_marked_delivered():
    task = symphony.create_task(title="Delivery", source_text="Notify")

    pending = symphony.claim_delivery_events(limit=10)

    assert len(pending) == 1
    assert pending[0]["task_id"] == task.task_id
    assert symphony.claim_delivery_events(limit=10) == []
    symphony.mark_delivery_delivered(pending[0]["id"])
    with symphony.connect() as conn:
        row = conn.execute("SELECT delivered_at FROM delivery_events WHERE id=?", (pending[0]["id"],)).fetchone()
    assert row["delivered_at"] > 0


def test_failed_delivery_claim_is_released_for_safe_retry():
    symphony.create_task(title="Retry delivery", source_text="Notify")
    event = symphony.claim_delivery_events(limit=1)[0]

    symphony.release_delivery_event(event["id"])

    assert symphony.claim_delivery_events(limit=1)[0]["id"] == event["id"]


def test_stale_delivery_claim_is_recovered_after_worker_crash():
    symphony.create_task(title="Recover delivery", source_text="Notify")
    event = symphony.claim_delivery_events(limit=1)[0]
    with symphony.connect() as conn:
        conn.execute("UPDATE delivery_events SET claimed_at=? WHERE id=?", (1, event["id"]))

    reclaimed = symphony.claim_delivery_events(limit=1, claim_ttl_seconds=1)

    assert reclaimed[0]["id"] == event["id"]


def test_delivery_claim_lease_is_not_reclaimed_while_fresh():
    symphony.create_task(title="Fresh delivery", source_text="Notify")
    event = symphony.claim_delivery_events(limit=1, claim_ttl_seconds=120)[0]

    assert symphony.claim_delivery_events(limit=1, claim_ttl_seconds=120) == []


def test_status_report_scopes_queue_and_exposes_safe_progress():
    task = symphony.create_task(title="Status", source_text="Inspect")

    report = symphony.status_report()

    assert report["counts"]["draft"] == 1
    assert report["active"] == []
    assert report["queue_size"] == 0
    assert report["recent"][0]["task_id"] == task.task_id


def test_progress_event_updates_last_action_and_status_report():
    task = symphony.create_task(title="Progress", source_text="Track worker", project_id="project-a")
    symphony.record_progress(task.task_id, action="pytest -q", detail="1 passed")

    report = symphony.status_report()

    assert report["recent"][0]["last_action"] == "pytest -q"
    assert report["recent"][0]["progress_detail"] == "1 passed"


def test_transient_retry_is_bounded_and_scheduled():
    task = symphony.create_task(title="Retry", source_text="Temporary network error", project_id="project-a")
    task = symphony.mark_ready(task.task_id)
    failed = symphony.schedule_retry(task.task_id, reason="network timeout", transient=True, max_retries=2)

    assert failed.status == "queued"
    assert failed.retry_count == 1
    assert symphony.schedule_retry(task.task_id, reason="permission denied", transient=False).status == "failed"


def test_runner_does_not_dispatch_before_next_retry_at(monkeypatch):
    task = symphony.create_task(title="Delayed retry", source_text="Retry later", project_id="project-a")
    symphony.mark_ready(task.task_id)
    symphony.schedule_retry(task.task_id, reason="temporary", transient=True, max_retries=2)

    monkeypatch.setattr(symphony.kb, "dispatch_once", lambda *args, **kwargs: pytest.fail("must wait"))

    assert symphony.SymphonyRunner().run_once() == []


def test_runner_skips_eligible_kanban_task_until_retry_deadline(monkeypatch):
    task = symphony.create_task(title="Delayed Kanban retry", source_text="Retry later", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='queued', kanban_board='development', kanban_task_id='k_delayed', next_retry_at=? WHERE task_id=?",
            (2_000_000_000, task.task_id),
        )
    monkeypatch.setattr(symphony.kb, "dispatch_once", lambda *args, **kwargs: pytest.fail("must wait for retry deadline"))

    assert symphony.SymphonyRunner().run_once() == []


def test_development_runner_contract_exposes_lifecycle_operations():
    runner = symphony.SymphonyRunner()

    assert runner.create_task(title="Contract", source_text="Lifecycle", project_id="project-a").status == "draft"
    assert callable(runner.enqueue)
    assert callable(runner.get_status)
    assert callable(runner.cancel)
    assert callable(runner.resume)
    assert callable(runner.reconcile)


def test_cancel_terminates_real_worker_process_and_keeps_single_run(tmp_path):
    task = symphony.create_task(title="Live worker", source_text="Cancel process", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', workspace_path=? WHERE task_id=?",
            (str(tmp_path), task.task_id),
        )
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, status, worker_pid, started_at, last_progress_at) VALUES ('run_live', ?, 'running', ?, ?, ?)",
            (task.task_id, 0, int(time.time()), int(time.time())),
        )
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with symphony.connect() as conn:
        conn.execute("UPDATE agent_runs SET worker_pid=? WHERE run_id='run_live'", (worker.pid,))

    cancelled = symphony.cancel(task.task_id)
    worker.poll()

    assert cancelled.status == "cancelled"
    assert worker.returncode is not None
    with symphony.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM agent_runs WHERE task_id=?", (task.task_id,)).fetchone()[0] == 1


def test_restart_recovery_keeps_live_worker_without_duplicate_dispatch(monkeypatch, tmp_path):
    task = symphony.create_task(title="Recover live worker", source_text="Restart safely", project_id="project-a")
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    now = int(time.time())
    with symphony.connect() as conn:
        conn.execute(
            "UPDATE development_tasks SET status='running', workspace_path=?, kanban_board='development', kanban_task_id='k_live_recovery' WHERE task_id=?",
            (str(tmp_path), task.task_id),
        )
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, kanban_task_id, status, worker_pid, started_at, last_progress_at) VALUES ('run_live_recovery', ?, 'k_live_recovery', 'running', ?, ?, ?)",
            (task.task_id, worker.pid, now, now),
        )

    monkeypatch.setattr(symphony.kb, "dispatch_once", lambda *args, **kwargs: pytest.fail("must not duplicate a live worker"))
    try:
        assert symphony.SymphonyRunner().run_once() == []
        run = symphony.latest_agent_run(task.task_id)
        assert run is not None and run.worker_pid == worker.pid and run.status == "running"
    finally:
        symphony.cancel(task.task_id)
        worker.wait(timeout=10)


def test_run_worker_checks_workspace_and_writes_passing_evidence(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = symphony.create_task(title="Worker wrapper", source_text="Run check", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute("UPDATE development_tasks SET status='running', workspace_path=? WHERE task_id=?", (str(workspace), task.task_id))
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, status, started_at, last_progress_at) VALUES ('run_wrapper', ?, 'running', 1, 1)",
            (task.task_id,),
        )

    result = symphony.run_worker_checks(
        task.task_id,
        commands=[[sys.executable, "-c", "print('ok')"]],
    )

    assert result["tests"][0]["exit_code"] == 0
    assert Path(result["artifact"]).parent == workspace / ".symphony"
    assert symphony.latest_agent_run(task.task_id).status == "completed"


def test_run_worker_checks_timeout_is_failed_and_audited(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = symphony.create_task(title="Timed out worker", source_text="Run timeout check", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute("UPDATE development_tasks SET status='running', workspace_path=? WHERE task_id=?", (str(workspace), task.task_id))
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, status, started_at, last_progress_at) VALUES ('run_timeout_wrapper', ?, 'running', 1, 1)",
            (task.task_id,),
        )

    result = symphony.run_worker_checks(
        task.task_id,
        commands=[[sys.executable, "-c", "import time; time.sleep(2)"]],
        timeout_seconds=1,
    )

    assert result["tests"][0]["exit_code"] == 124
    assert symphony.latest_agent_run(task.task_id).status == "failed"
    with symphony.connect() as conn:
        rows = conn.execute("SELECT kind, payload FROM development_events WHERE task_id=? ORDER BY id", (task.task_id,)).fetchall()
    timeout_event = next(__import__("json").loads(row["payload"]) for row in rows if row["kind"] == "validation.check")
    assert timeout_event["timed_out"] is True
    assert timeout_event["exit_code"] == 124
    assert Path(result["artifact"]).is_file()
    with pytest.raises(ValueError, match="only running or validating"):
        symphony.publish_for_review(task.task_id, commit_message="timeout", test_summary="timeout", result_summary="timeout")


def test_run_worker_checks_persists_failure_without_completion(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = symphony.create_task(title="Failing worker", source_text="Run failing check", project_id="project-a")
    with symphony.connect() as conn:
        conn.execute("UPDATE development_tasks SET status='running', workspace_path=? WHERE task_id=?", (str(workspace), task.task_id))
        conn.execute(
            "INSERT INTO agent_runs(run_id, task_id, status, started_at, last_progress_at) VALUES ('run_failed_wrapper', ?, 'running', 1, 1)",
            (task.task_id,),
        )

    result = symphony.run_worker_checks(
        task.task_id,
        commands=[[sys.executable, "-c", "raise SystemExit(3)"]],
    )

    assert result["tests"][0]["exit_code"] == 3
    assert symphony.latest_agent_run(task.task_id).status == "failed"
    with symphony.connect() as conn:
        rows = conn.execute("SELECT kind, payload FROM development_events WHERE task_id=? ORDER BY id", (task.task_id,)).fetchall()
    kinds = [row["kind"] for row in rows]
    assert "validation.finished" in kinds
    assert "agent.failed" in kinds
    assert "task.failed" in kinds
    validation = next(__import__("json").loads(row["payload"]) for row in rows if row["kind"] == "validation.finished")
    check = next(__import__("json").loads(row["payload"]) for row in rows if row["kind"] == "validation.check")
    assert validation["passed"] is False
    assert check["cwd"] == "task workspace"
    assert check["exit_code"] == 3
    assert "C:\\" not in check["command"]
    assert Path(result["artifact"]).is_file()
    with pytest.raises(ValueError, match="only running or validating"):
        symphony.publish_for_review(task.task_id, commit_message="no", test_summary="no", result_summary="no")
