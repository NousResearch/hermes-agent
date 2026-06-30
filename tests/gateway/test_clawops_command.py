from __future__ import annotations

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from hermes_cli import kanban_db as kb


def _make_event(text: str = "/clawops inspect runtime queue") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="msg-1",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="kj",
            user_name="KJ",
            chat_type="dm",
        ),
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._kanban_notifier_profile = "main"
    return runner


def test_clawops_command_is_registered_with_alias():
    from hermes_cli.commands import resolve_command

    assert resolve_command("clawops").name == "clawops"
    assert resolve_command("claw").name == "clawops"


@pytest.mark.asyncio
async def test_clawops_command_creates_task_and_subscription(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_CLAWOPS_ASSIGNEE", "clawops-test")

    result = await _make_runner()._handle_clawops_command(
        _make_event("/clawops verify Codex runtime health")
    )

    with kb.connect_closing(db_path) as conn:
        rows = conn.execute("SELECT * FROM tasks ORDER BY created_at ASC").fetchall()
        task_id = rows[0]["id"]
        subs = kb.list_notify_subs(conn, task_id)

    assert "ClawOps task queued" in result
    assert "Assigned Agent: `clawops-test`" in result
    assert "Hermes -> kanban queue -> ClawOps worker -> Hermes summary" in result
    assert len(rows) == 1
    assert rows[0]["status"] == "ready"
    assert rows[0]["assignee"] == "clawops-test"
    assert rows[0]["created_by"] == "hermes-clawops-intake"
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat-1"
    assert subs[0]["user_id"] == "kj"
    assert subs[0]["notifier_profile"] == "main"


@pytest.mark.asyncio
async def test_clawops_command_requires_objective():
    result = await _make_runner()._handle_clawops_command(_make_event("/clawops"))

    assert "Usage:" in result
    assert "/clawops <objective>" in result
