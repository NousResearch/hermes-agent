"""``hermes kanban create`` did not run through ``_dispatch_pre_tool_call_hooks`` (#116455):
an agent-invoked ``kanban_create`` tool call is covered by ``pre_tool_call`` plugins (e.g. a
board-inference plugin), but the human CLI command bypassed that dispatch entirely, so the
same plugin covered one path and not the other.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_ns(**overrides) -> argparse.Namespace:
    ns = argparse.Namespace(
        title="x", body=None, assignee="worker", created_by="user", workspace=None,
        branch=None, tenant=None, priority=0, parent=None, triage=False,
        idempotency_key=None, max_runtime=None, max_retries=None, skills=None,
        project=None, model_override=None, provider_override=None, goal_mode=False,
        goal_max_turns=None, completion_contract=None, initial_status="running", json=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _only_task(conn) -> dict:
    row = conn.execute("SELECT * FROM tasks").fetchone()
    assert row is not None
    return dict(row)


@patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks")
def test_block_directive_prevents_task_creation(mock_dispatch, kanban_home, capsys):
    """A plugin blocking the ``kanban_create`` tool call must also block the CLI path."""
    mock_dispatch.return_value = ("no boards available right now", None)

    rc = kc._cmd_create(_create_ns())

    assert rc != 0
    assert "no boards available right now" in capsys.readouterr().err
    with kbc.connect_closing() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


@patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks")
def test_modify_directive_overrides_cli_args(mock_dispatch, kanban_home):
    """A plugin rewriting args (board-inference: pin priority/assignee) must land on the task
    actually created by the CLI command, not just an agent tool call."""
    mock_dispatch.return_value = (None, {"priority": 9, "assignee": "inferred-worker"})

    rc = kc._cmd_create(_create_ns(assignee="worker", priority=0))

    assert rc == 0
    with kbc.connect_closing() as conn:
        task = _only_task(conn)
    assert task["priority"] == 9
    assert task["assignee"] == "inferred-worker"


@patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks")
def test_hook_sees_tool_shaped_args(mock_dispatch, kanban_home):
    """The dict handed to the hook must use the ``kanban_create`` tool's field names
    (``project``, ``model``, ``provider``, ...), not CLI-internal ones, so a plugin written
    against the agent tool schema works unmodified against the CLI."""
    mock_dispatch.return_value = (None, None)

    kc._cmd_create(_create_ns(
        title="ship it", assignee="worker", model_override="gpt-5", provider_override="openai",
    ))

    assert mock_dispatch.call_args is not None
    tool_name, tool_args = mock_dispatch.call_args.args
    assert tool_name == "kanban_create"
    assert tool_args["title"] == "ship it"
    assert tool_args["model"] == "gpt-5"
    assert tool_args["provider"] == "openai"


def test_no_plugins_registered_creates_task_normally(kanban_home):
    """Sanity check against the real (un-mocked) dispatch path: with no plugins loaded,
    creation behaves exactly as before this change."""
    rc = kc._cmd_create(_create_ns(title="plain task", assignee="worker"))

    assert rc == 0
    with kbc.connect_closing() as conn:
        task = _only_task(conn)
    assert task["title"] == "plain task"
    assert task["assignee"] == "worker"
