"""Regression tests for dispatcher-owned Kanban process lineage.

The worker identity is an execution boundary: the dispatcher-launched process
may use the task lifecycle, while ordinary descendants may keep workspace
context but must not inherit board authority.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def worker_env(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "17")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock-parent")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_KANBAN_DB", "/tmp/parent-kanban.db")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", "/tmp/parent-workspace")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", "/tmp/workspaces")
    monkeypatch.setenv("HERMES_KANBAN_OWNER_PID", str(os.getpid()))
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_TOKEN", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)


def test_dispatcher_identity_requires_matching_owner_pid(monkeypatch):
    from agent.delegation_context import is_dispatcher_owned_worker_context

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.delenv("HERMES_KANBAN_OWNER_PID", raising=False)
    assert is_dispatcher_owned_worker_context() is False

    monkeypatch.setenv("HERMES_KANBAN_OWNER_PID", str(os.getpid() + 1))
    assert is_dispatcher_owned_worker_context() is False

    monkeypatch.setenv("HERMES_KANBAN_OWNER_PID", str(os.getpid()))
    assert is_dispatcher_owned_worker_context() is True

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert is_dispatcher_owned_worker_context() is True


def test_dispatcher_bootstrap_binds_owner_pid_once(monkeypatch):
    from agent.delegation_context import (
        KANBAN_OWNER_PID_PENDING,
        bind_kanban_worker_identity,
    )

    env = {
        "HERMES_KANBAN_TASK": "t_parent",
        "HERMES_KANBAN_OWNER_PID": KANBAN_OWNER_PID_PENDING,
    }
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    assert bind_kanban_worker_identity(env) is True
    assert env["HERMES_KANBAN_OWNER_PID"] == "4242"

    child_env = dict(env)
    monkeypatch.setattr(os, "getpid", lambda: 4243)
    assert bind_kanban_worker_identity(child_env) is False
    assert child_env["HERMES_KANBAN_OWNER_PID"] == "4242"


def test_inherited_identity_is_dropped_but_workspace_context_remains():
    from agent.delegation_context import (
        KANBAN_WORKSPACE_ENV_KEYS,
        initialize_kanban_worker_process,
    )

    env = {
        "HERMES_KANBAN_TASK": "t_parent",
        "HERMES_KANBAN_RUN_ID": "17",
        "HERMES_KANBAN_CLAIM_LOCK": "lock-parent",
        "HERMES_KANBAN_BOARD": "default",
        "HERMES_KANBAN_DB": "/tmp/parent-kanban.db",
        "HERMES_KANBAN_OWNER_PID": str(os.getpid() + 1),
        "HERMES_KANBAN_WORKSPACE": "/tmp/parent-workspace",
        "HERMES_KANBAN_WORKSPACES_ROOT": "/tmp/workspaces",
        "HERMES_KANBAN_BRANCH": "feature/parent",
        "HERMES_KANBAN_FUTURE_CAPABILITY": "must-not-leak",
    }

    assert initialize_kanban_worker_process(env) is False
    assert not any(
        key.startswith("HERMES_KANBAN_") and key not in KANBAN_WORKSPACE_ENV_KEYS
        for key in env
    )
    assert env["HERMES_KANBAN_WORKSPACE"] == "/tmp/parent-workspace"
    assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == "/tmp/workspaces"


def test_real_child_cannot_reuse_parent_worker_identity(worker_env):
    code = (
        "import json, os; "
        "from agent.delegation_context import initialize_kanban_worker_process; "
        "initialize_kanban_worker_process(); "
        "from agent.delegation_context import is_dispatcher_owned_worker_context; "
        "from agent.kanban_stop import kanban_stop_nudge_enabled; "
        "from tools import kanban_tools; "
        "print(json.dumps({"
        "'owned': is_dispatcher_owned_worker_context(),"
        "'nudge': kanban_stop_nudge_enabled(),"
        "'task': kanban_tools._default_task_id(None),"
        "'run': kanban_tools._worker_run_id('t_parent'),"
        "'kanban_env': sorted(k for k in os.environ if k.startswith('HERMES_KANBAN_'))"
        "}))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    observed = json.loads(child.stdout)
    assert observed == {
        "owned": False,
        "nudge": False,
        "task": None,
        "run": None,
        "kanban_env": [
            "HERMES_KANBAN_WORKSPACE",
            "HERMES_KANBAN_WORKSPACES_ROOT",
        ],
    }


def test_terminal_and_execute_code_children_drop_identity(worker_env):
    from tools.code_execution_env import _scrub_child_env
    from tools.environments.local import _make_run_env, build_subprocess_env, hermes_subprocess_env

    terminal_env = _make_run_env(dict(os.environ))
    built_env = build_subprocess_env()
    sandbox_env = _scrub_child_env(
        dict(os.environ),
        is_passthrough=lambda key: key.startswith("HERMES_KANBAN_"),
        is_windows=False,
    )
    non_terminal_env = hermes_subprocess_env()

    for child_env in (terminal_env, built_env, sandbox_env, non_terminal_env):
        assert "HERMES_KANBAN_TASK" not in child_env
        assert "HERMES_KANBAN_RUN_ID" not in child_env
        assert "HERMES_KANBAN_CLAIM_LOCK" not in child_env
        assert "HERMES_KANBAN_BOARD" not in child_env
        assert "HERMES_KANBAN_DB" not in child_env
        assert child_env.get("HERMES_KANBAN_WORKSPACE") == "/tmp/parent-workspace" or "HERMES_KANBAN_WORKSPACE" not in child_env
        assert child_env.get("HERMES_DELEGATED_CHILD_CONTEXT") is None


def test_authorized_codex_runtime_can_keep_worker_identity(worker_env):
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_kanban=True)
    assert env["HERMES_KANBAN_TASK"] == "t_parent"
    assert env["HERMES_KANBAN_OWNER_PID"] == str(os.getpid())


def test_delegated_child_scrub_still_removes_all_kanban_keys(worker_env):
    from agent.delegation_context import delegated_child_context, scrub_kanban_env

    with delegated_child_context():
        env = scrub_kanban_env(dict(os.environ))

    assert not any(key.startswith("HERMES_KANBAN_") for key in env)
    assert env["HERMES_DELEGATED_CHILD_CONTEXT"] == "1"


def test_tool_identity_gates_fail_closed_in_non_dispatcher_context(worker_env):
    from agent.delegation_context import non_dispatcher_owned_context
    from tools import kanban_tools

    with non_dispatcher_owned_context():
        assert kanban_tools._default_task_id(None) is None
        assert kanban_tools._worker_run_id("t_parent") is None
        with pytest.raises(kanban_tools._Reject):
            kanban_tools._enforce_worker_task_ownership("t_parent")
        # An explicitly configured orchestrator may still target another card.
        kanban_tools._enforce_worker_task_ownership("t_other")


def test_cli_run_id_and_foreign_task_mutation_are_fenced(monkeypatch, tmp_path, capsys):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban as cli_kanban

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        foreign = kb.create_task(conn, title="foreign", assignee="other")
        parent_claim = kb.claim_task(conn, parent, claimer="gateway:worker")
        foreign_claim = kb.claim_task(conn, foreign, claimer="gateway:other")
        assert parent_claim is not None and foreign_claim is not None
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", parent)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(parent_claim.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_OWNER_PID", str(os.getpid()))

    assert cli_kanban._worker_run_id_for(parent) == parent_claim.current_run_id
    assert cli_kanban._worker_run_id_for(foreign) is None

    args = argparse.Namespace(
        kanban_action="complete",
        task_ids=[foreign],
        result="foreign completion",
        summary=None,
        metadata=None,
        board=None,
    )
    rc = cli_kanban.kanban_command(args)
    assert rc == 1
    assert "scoped to task" in capsys.readouterr().err

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, foreign).status == "running"
    finally:
        conn.close()


def test_dispatcher_spawn_stamps_pending_owner(monkeypatch, tmp_path):
    import subprocess as subprocess_module

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    root = tmp_path / ".hermes"
    (root / "profiles" / "worker").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(subprocess_module, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = kb.Task(
        id="t_spawn_owner",
        title="spawn owner",
        body=None,
        assignee="worker",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )

    assert kbd._default_spawn(task, str(workspace)) == 4242
    assert captured["env"]["HERMES_KANBAN_OWNER_PID"] == "pending"
