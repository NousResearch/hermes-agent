"""Tests for Slack direct Kanban task intake."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import gateway.platforms.slack as _slack_mod
_slack_mod.SLACK_AVAILABLE = True

from gateway.platforms.slack import SlackAdapter  # noqa: E402
from hermes_cli import kanban_db  # noqa: E402
from hermes_cli.kanban_slack_intake import parse_slack_kanban_intake  # noqa: E402


def test_parse_slack_kanban_intake_defaults_to_triage():
    request = parse_slack_kanban_intake('title="Fix login" body="Steps here"')

    assert request.title == "Fix login"
    assert request.body == "Steps here"
    assert request.column == "triage"


def test_parse_slack_kanban_intake_accepts_todo_shorthand():
    request = parse_slack_kanban_intake('--todo "Fix login" assignee=default')

    assert request.title == "Fix login"
    assert request.column == "todo"
    assert request.assignee == "default"


def test_parse_slack_kanban_intake_rejects_empty_title():
    with pytest.raises(ValueError, match="title is required"):
        parse_slack_kanban_intake("")


@pytest.mark.asyncio
async def test_kanban_add_slash_creates_triage_task_and_replies(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._reply_to_slack_kanban_intake = AsyncMock()

    await adapter._handle_slash_command(
        {
            "command": "/kanban-add",
            "text": 'title="Fix login" body="Steps here"',
            "user_id": "U123",
            "channel_id": "C123",
            "team_id": "T123",
            "response_url": "https://example.invalid/response",
        }
    )

    conn = kanban_db.connect(board="default")
    rows = conn.execute("SELECT id, title, body, status, created_by FROM tasks").fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0]["title"] == "Fix login"
    assert rows[0]["body"] == "Steps here"
    assert rows[0]["status"] == "triage"
    assert rows[0]["created_by"] == "slack:U123"

    adapter._reply_to_slack_kanban_intake.assert_awaited_once()
    await_args = adapter._reply_to_slack_kanban_intake.await_args
    assert await_args is not None
    reply = await_args.args[1]
    assert "Kanban task created" in reply
    assert f"task_id: {rows[0]['id']}" in reply
    assert "column: triage" in reply


@pytest.mark.asyncio
async def test_kanban_add_slash_can_force_todo(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._reply_to_slack_kanban_intake = AsyncMock()

    await adapter._handle_slash_command(
        {
            "command": "/kanban-add",
            "text": 'column=todo title="Prepare invoice"',
            "user_id": "U123",
            "channel_id": "C123",
            "team_id": "T123",
        }
    )

    conn = kanban_db.connect(board="default")
    row = conn.execute("SELECT title, status FROM tasks").fetchone()
    conn.close()

    assert row["title"] == "Prepare invoice"
    assert row["status"] == "todo"
    await_args = adapter._reply_to_slack_kanban_intake.await_args
    assert await_args is not None
    reply = await_args.args[1]
    assert "column: todo" in reply
    assert "status: todo" in reply


@pytest.mark.asyncio
async def test_hermes_legacy_kanban_add_routes_directly(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._reply_to_slack_kanban_intake = AsyncMock()
    adapter.handle_message = AsyncMock()

    await adapter._handle_slash_command(
        {
            "command": "/hermes",
            "text": 'kanban-add title="Legacy route"',
            "user_id": "U123",
            "channel_id": "C123",
            "team_id": "T123",
        }
    )

    adapter.handle_message.assert_not_awaited()
    conn = kanban_db.connect(board="default")
    row = conn.execute("SELECT title, status FROM tasks").fetchone()
    conn.close()
    assert row["title"] == "Legacy route"
    assert row["status"] == "triage"


@pytest.mark.asyncio
async def test_kanban_add_slash_reports_invalid_input(monkeypatch):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._reply_to_slack_kanban_intake = AsyncMock()

    await adapter._handle_slash_command(
        {
            "command": "/kanban-add",
            "text": "column=doing title=Bad",
            "user_id": "U123",
            "channel_id": "C123",
            "team_id": "T123",
        }
    )

    await_args = adapter._reply_to_slack_kanban_intake.await_args
    assert await_args is not None
    reply = await_args.args[1]
    assert "Kanban task was not created" in reply
    assert "column must be one of" in reply
