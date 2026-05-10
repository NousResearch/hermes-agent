"""Tests for the gateway /kanban command handler."""

from __future__ import annotations

import re
import os
from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def _isolate_kanban_board_env():
    prev = os.environ.get("HERMES_KANBAN_BOARD")
    os.environ.pop("HERMES_KANBAN_BOARD", None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("HERMES_KANBAN_BOARD", None)
        else:
            os.environ["HERMES_KANBAN_BOARD"] = prev


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="user-1",
            user_name="tester",
            thread_id="thread-9",
        ),
        message_id="msg-1",
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    return runner


@pytest.mark.asyncio
async def test_gateway_kanban_create_with_explicit_board_auto_subscribes(kanban_home):
    """`/kanban --board <slug> create ...` should still auto-subscribe."""
    runner = _make_runner()
    kb.create_board("projx")

    output = await runner._handle_kanban_command(
        _make_event("/kanban --board projx create 'board scoped task'")
    )

    task_id = re.search(r"(t_[0-9a-f]+)\b", output)
    assert task_id, output
    assert "subscribed" in output.lower(), output

    with kb.connect(board="projx") as conn:
        subs = kb.list_notify_subs(conn, task_id.group(1))
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat-1"
    assert subs[0]["thread_id"] == "thread-9"

    with kb.connect(board="default") as conn:
        assert kb.list_notify_subs(conn, task_id.group(1)) == []
