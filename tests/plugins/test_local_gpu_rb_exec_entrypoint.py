import json
import os
import sqlite3

import pytest

from plugins.local_gpu_admission.rb_exec_entrypoint import (
    _bind_context, _scope_contains_unit, _task_is_still_claimed, _worker_exec_env,
)


def test_worker_exec_env_uses_served_profile_scope(monkeypatch, tmp_path):
    from tools.environments import local

    home = tmp_path / "served-profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    calls = {}

    def build_env(*, base, target_home, inherit_credentials):
        calls.update(target_home=target_home, inherit_credentials=inherit_credentials)
        return {"HERMES_HOME": str(target_home), "HERMES_KANBAN_TASK": base["HERMES_KANBAN_TASK"]}

    monkeypatch.setattr(local, "served_profile_child_env", build_env)
    env = _worker_exec_env("rb-job-1")

    assert calls == {"target_home": str(home), "inherit_credentials": True}
    assert env == {
        "HERMES_HOME": str(home), "HERMES_KANBAN_TASK": "task-1",
        "RESOURCE_BALANCER_ADMISSION_JOB_ID": "rb-job-1",
    }


def test_bind_context_persists_only_to_an_active_gpu_reservation(tmp_path):
    state_dir = tmp_path / "rb"
    state_dir.mkdir()
    with sqlite3.connect(state_dir / "queue.sqlite3") as conn:
        conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, class TEXT, state TEXT, context TEXT)")
        conn.execute("INSERT INTO jobs VALUES ('job-1','gpu','running',NULL)")
        conn.execute("INSERT INTO jobs VALUES ('job-2','gpu','queued',NULL)")

    payload = {
        "task_id": "task-1", "run_id": "7", "claim_lock": "claim-a",
        "capability_id": "capability-1", "task_contract_sha256": "a" * 64,
    }
    _bind_context(str(state_dir), "job-1", payload)
    with sqlite3.connect(state_dir / "queue.sqlite3") as conn:
        saved = conn.execute(
            "SELECT context FROM hermes_local_gpu_reservations WHERE job_id='job-1'",
        ).fetchone()[0]
        rb_context = conn.execute("SELECT context FROM jobs WHERE id='job-1'").fetchone()[0]
    assert json.loads(saved) == payload
    assert rb_context is None

    with pytest.raises(RuntimeError, match="disappeared"):
        _bind_context(str(state_dir), "job-2", payload)


def test_bind_context_fails_closed_if_rb_job_is_not_active(tmp_path):
    state_dir = tmp_path / "rb"
    state_dir.mkdir()
    with sqlite3.connect(state_dir / "queue.sqlite3") as conn:
        conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, class TEXT, state TEXT, context TEXT)")
        conn.execute("INSERT INTO jobs VALUES ('job-1','gpu','queued',NULL)")
    with pytest.raises(RuntimeError, match="disappeared"):
        _bind_context(str(state_dir), "job-1", {})


def test_scope_unit_match_is_exact_enough_to_reject_neighbor_units():
    assert _scope_contains_unit("hermes-worker-kanban-t1-run-7.scope",
                                "0::/user.slice/hermes-worker-kanban-t1-run-7.scope")
    assert not _scope_contains_unit("hermes-worker-kanban-t1-run-7.scope",
                                    "0::/user.slice/hermes-worker-kanban-t1-run-70.scope")
    assert not _scope_contains_unit("hermes-worker-kanban-t1-run-7.scope",
                                    "0::/user.slice/prefix-hermes-worker-kanban-t1-run-7.scope")
    assert not _scope_contains_unit("hermes-worker-kanban-t1-run-7.scope",
                                    "0::/user.slice/hermes-worker-kanban-t1-run-7.scope.slice/child.scope")
    assert not _scope_contains_unit("hermes-worker-kanban-t1-run-7.scope",
                                    "0::/user.slice/hermes-worker-kanban-t1-run-7.scope-suffix")
    assert not _scope_contains_unit("hermes-worker-kanban-t1-run-7.service",
                                    "0::/user.slice/hermes-worker-kanban-t1-run-7.service")


def test_scoped_worker_replaces_systemd_run_launcher_pid(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_local_model_scope import task_run_contract_sha256

    db_path = kb.init_db()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title="scoped local task", body="Local-GPU-Capability: cap-1",
            assignee="profile", workspace_kind="scratch", workspace_path=str(workspace),
            model_override="model:tag", provider_override="ollama",
            completion_contract="local-only", skills=["local-gpu-model-task-gate"],
            max_runtime_seconds=300, max_retries=1,
        )
        task = kb.claim_task(conn, task_id, claimer="claim-1")
        assert task
        launcher_pid = os.getpid() + 100000
        conn.execute(
            "UPDATE tasks SET worker_pid=?,worker_started_at='launcher' WHERE id=?",
            (launcher_pid, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid=?,worker_started_at='launcher' WHERE id=?",
            (launcher_pid, task.current_run_id),
        )
        conn.commit()

    context = {
        "task_id": task_id, "run_id": str(task.current_run_id),
        "claim_lock": task.claim_lock, "capability_id": "cap-1",
        "task_contract_sha256": task_run_contract_sha256(task),
    }
    scope_unit = f"hermes-worker-kanban-{task_id}-run-{task.current_run_id}.scope"
    assert _task_is_still_claimed(
        str(db_path), context, os.getpid(), scope_unit,
        cgroup_text=f"0::/user.slice/{scope_unit}",
    )
    with sqlite3.connect(db_path) as conn:
        task_pid = conn.execute("SELECT worker_pid FROM tasks WHERE id=?", (task_id,)).fetchone()[0]
        run_pid = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (task.current_run_id,)).fetchone()[0]
    assert task_pid == run_pid == os.getpid()
