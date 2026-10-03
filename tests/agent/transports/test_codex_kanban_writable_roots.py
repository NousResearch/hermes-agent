"""Regression for #27941: task artifacts may live outside the Codex thread cwd."""

from contextlib import nullcontext
from unittest.mock import MagicMock
import tomllib

import pytest

from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
from agent.transports import codex_app_server


@pytest.fixture
def spawn(monkeypatch, tmp_path):
    proc = MagicMock()
    proc.stdin = proc.stdout = proc.stderr = None
    popen = MagicMock(return_value=proc)
    monkeypatch.setattr(codex_app_server.subprocess, "Popen", popen)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board" / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path / "artifacts"))

    def start():
        client = codex_app_server.CodexAppServerClient(codex_bin="codex")
        client._closed = True
        cmd = popen.call_args.args[0]
        overrides = "\n".join(cmd[i + 1] for i, arg in enumerate(cmd[:-1]) if arg == "-c")
        return tomllib.loads(overrides)

    return start


@pytest.mark.parametrize("workspace", [None, "artifacts", "board", 'artifacts "quoted" \U0001f4c1'])
def test_worker_grants_only_task_and_board_roots(spawn, monkeypatch, tmp_path, workspace):
    if workspace is None:
        monkeypatch.delenv("HERMES_KANBAN_WORKSPACE")
    else:
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", workspace)

    config = spawn()
    expected = {str((tmp_path / "board").resolve())}
    if workspace is not None:
        expected.add(str((tmp_path / workspace).resolve()))

    sandbox = config["sandbox_workspace_write"]
    assert set(sandbox["writable_roots"]) == expected
    assert len(sandbox["writable_roots"]) == len(expected)
    assert config["sandbox_mode"] == "workspace-write"
    assert sandbox["network_access"] is False


@pytest.mark.parametrize("context", ["interactive", "delegate", "cron", "descendant"])
def test_non_workers_do_not_inherit_kanban_sandbox_grants(spawn, monkeypatch, context):
    if context == "interactive":
        monkeypatch.delenv("HERMES_KANBAN_TASK")
    elif context == "descendant":
        monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")

    scopes = {"delegate": delegated_child_context, "cron": non_dispatcher_owned_context}
    with scopes.get(context, nullcontext)():
        config = spawn()

    assert "sandbox_mode" not in config
    assert "sandbox_workspace_write" not in config
