"""Regression tests for Matrix command batching and command routing."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageType
from gateway.session import SessionSource


ROOM_ID = "!room:example"
USER_ID = "@user:example"


def _make_adapter(*, synthetic_thread: bool = False):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = object.__new__(MatrixAdapter)
    adapter.config = SimpleNamespace(
        extra={"group_sessions_per_user": True, "thread_sessions_per_user": False}
    )
    adapter._text_batch_delay_seconds = 0.02
    adapter._text_batch_split_delay_seconds = 0.02
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}

    async def resolve_context(room_id, sender, event_id, body, source_content, relates_to):
        return body, not synthetic_thread, "group" if synthetic_thread else "dm", (
            event_id if synthetic_thread else None
        ), "User", SessionSource(
            platform=Platform.MATRIX,
            chat_id=ROOM_ID,
            chat_type="group" if synthetic_thread else "dm",
            user_id=USER_ID,
            thread_id=event_id if synthetic_thread else None,
            parent_chat_id=ROOM_ID if synthetic_thread else None,
            message_id=event_id,
        )

    adapter._resolve_message_context = resolve_context
    adapter.handle_message = AsyncMock()
    return adapter


async def _send(adapter, event_id: str, body: str) -> None:
    await adapter._handle_text_message(
        ROOM_ID, USER_ID, event_id, 0, {"body": body}, {}
    )


@pytest.mark.asyncio
async def test_unknown_command_like_continuation_joins_long_command_batch():
    adapter = _make_adapter()

    await _send(adapter, "$long", "!queue " + ("x" * 3900))
    await _send(adapter, "$tail", "/not-a-command client split")
    await asyncio.sleep(0.08)

    adapter.handle_message.assert_awaited_once()
    dispatched = adapter.handle_message.await_args.args[0]
    assert dispatched.message_type == MessageType.COMMAND
    assert dispatched.get_command() == "queue"
    assert "/not-a-command client split" in dispatched.text


@pytest.mark.asyncio
async def test_auto_threaded_continuation_joins_long_command_batch():
    adapter = _make_adapter(synthetic_thread=True)

    await _send(adapter, "$long", "!queue " + ("x" * 3900))
    await _send(adapter, "$tail", "client split continuation")
    await asyncio.sleep(0.08)

    adapter.handle_message.assert_awaited_once()
    dispatched = adapter.handle_message.await_args.args[0]
    assert dispatched.message_type == MessageType.COMMAND
    assert dispatched.get_command() == "queue"
    assert "client split continuation" in dispatched.text


@pytest.mark.parametrize("command", ["/stop", "/new"])
@pytest.mark.asyncio
async def test_recognized_control_bypasses_pending_long_command_batch(command):
    adapter = _make_adapter()

    await _send(adapter, "$long", "!queue " + ("x" * 3900))
    await _send(adapter, "$control", command)

    assert adapter.handle_message.await_count == 1
    assert adapter.handle_message.await_args.args[0].text == command

    await asyncio.sleep(0.08)
    assert adapter.handle_message.await_count == 2
    assert adapter.handle_message.await_args_list[1].args[0].get_command() == "queue"


@pytest.mark.asyncio
async def test_recognized_skill_bypasses_pending_long_command_batch(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/arxiv": {"name": "arxiv"}},
    )

    await _send(adapter, "$long", "!queue " + ("x" * 3900))
    await _send(adapter, "$skill", "/arxiv matrix routing")

    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].get_command() == "arxiv"
    await asyncio.sleep(0.08)


@pytest.mark.asyncio
async def test_recognized_plugin_bypasses_pending_long_command_batch(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(
        "hermes_cli.commands._iter_plugin_command_entries",
        lambda: [("plugin-command", "test plugin", "")],
    )

    await _send(adapter, "$long", "!queue " + ("x" * 3900))
    await _send(adapter, "$plugin", "/plugin-command now")

    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].get_command() == "plugin-command"
    await asyncio.sleep(0.08)


@pytest.mark.asyncio
async def test_long_command_flushes_pending_text_before_its_own_batch():
    adapter = _make_adapter()

    await _send(adapter, "$text", "pending text")
    await _send(adapter, "$long", "!queue " + ("x" * 3900))
    await _send(adapter, "$tail", "client split continuation")
    await asyncio.sleep(0.08)

    assert adapter.handle_message.await_count == 2
    text_event, command_event = [call.args[0] for call in adapter.handle_message.await_args_list]
    assert text_event.message_type == MessageType.TEXT
    assert text_event.text == "pending text"
    assert command_event.message_type == MessageType.COMMAND
    assert command_event.get_command() == "queue"
    assert "client split continuation" in command_event.text
