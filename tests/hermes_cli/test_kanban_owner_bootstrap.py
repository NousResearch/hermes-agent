"""Authenticated, one-use owner bootstrap for detached Kanban workers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.execution_context import ExecutionRole, execution_role_for_new_agent
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claim(conn, *, review: bool = False):
    task_id = kb.create_task(conn, title="bootstrap", assignee="worker")
    if review:
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        claimed = kb.claim_review_task(conn, task_id, claimer="test:review")
    else:
        claimed = kb.claim_task(conn, task_id, claimer="test:claim")
    assert claimed is not None
    assert claimed.current_run_id is not None
    assert claimed.owner_bootstrap_nonce
    return claimed


@pytest.mark.parametrize("review", [False, True])
def test_claim_persists_and_returns_per_run_bootstrap_nonce(kanban_home, review):
    with kb.connect() as conn:
        claimed = _claim(conn, review=review)
        run = kb.get_run(conn, claimed.current_run_id)

    assert run is not None
    assert run.owner_bootstrap_nonce == claimed.owner_bootstrap_nonce


def test_bootstrap_is_bound_to_exact_active_run_and_consumed_once(kanban_home):
    with kb.connect() as conn:
        claimed = _claim(conn)
        args = {
            "task_id": claimed.id,
            "run_id": claimed.current_run_id,
            "profile": "worker",
            "claim_lock": claimed.claim_lock,
            "nonce": claimed.owner_bootstrap_nonce,
        }
        for changed in (
            {"task_id": "t_wrong"},
            {"run_id": claimed.current_run_id + 1},
            {"profile": "other"},
            {"claim_lock": "wrong-lock"},
            {"nonce": "wrong-nonce"},
        ):
            attempt = {**args, **changed}
            assert kb.consume_task_owner_bootstrap(conn, **attempt) is False
        assert kb.consume_task_owner_bootstrap(conn, **args) is True
        assert kb.consume_task_owner_bootstrap(conn, **args) is False
        assert kb.get_run(conn, claimed.current_run_id).owner_bootstrap_nonce is None


def test_real_claim_spawn_env_authenticates_one_agent_once(
    kanban_home,
    tmp_path,
    monkeypatch,
):
    with kb.connect() as conn:
        claimed = _claim(conn)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured: dict[str, dict] = {}

    class FakeProc:
        pid = 4242

    def fake_popen(_cmd, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: None)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    assert kb._default_spawn(claimed, str(workspace)) == 4242
    worker_env = captured["env"]
    replay_env = dict(worker_env)
    assert worker_env["HERMES_KANBAN_SESSION"] == "1"
    assert worker_env["HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE"] == (
        claimed.owner_bootstrap_nonce
    )

    assert execution_role_for_new_agent(worker_env) is ExecutionRole.KANBAN_OWNER
    assert "HERMES_KANBAN_SESSION" not in worker_env
    assert "HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE" not in worker_env
    assert worker_env["HERMES_KANBAN_DELEGATE_SESSION"] == "1"
    assert execution_role_for_new_agent(replay_env) is ExecutionRole.KANBAN_DELEGATE

    with kb.connect() as conn:
        run = kb.get_run(conn, claimed.current_run_id)
    assert run.owner_bootstrap_nonce is None


def test_forged_mapping_and_fresh_process_never_become_owner(
    kanban_home,
):
    with kb.connect() as conn:
        claimed = _claim(conn)

    forged = {
        "HERMES_KANBAN_SESSION": "1",
        "HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE": "attacker-chosen",
        "HERMES_KANBAN_TASK": claimed.id,
        "HERMES_KANBAN_RUN_ID": str(claimed.current_run_id),
        "HERMES_KANBAN_CLAIM_LOCK": claimed.claim_lock,
        "HERMES_PROFILE": "worker",
        "HERMES_KANBAN_DB": os.environ["HERMES_KANBAN_DB"],
    }
    mapping_attempt = dict(forged)
    assert execution_role_for_new_agent(mapping_attempt) is ExecutionRole.KANBAN_DELEGATE
    assert "HERMES_KANBAN_SESSION" not in mapping_attempt
    assert "HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE" not in mapping_attempt
    assert mapping_attempt["HERMES_KANBAN_DELEGATE_SESSION"] == "1"

    child_env = os.environ.copy()
    child_env.update(forged)
    child_env.pop("HERMES_KANBAN_DELEGATE_SESSION", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from agent.execution_context import execution_role_for_new_agent; "
                "print(execution_role_for_new_agent().value)"
            ),
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=child_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "kanban_delegate"

    with kb.connect() as conn:
        run = kb.get_run(conn, claimed.current_run_id)
    assert run.owner_bootstrap_nonce == claimed.owner_bootstrap_nonce
