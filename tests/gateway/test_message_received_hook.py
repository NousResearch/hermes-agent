import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    return runner


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-123",
        chat_id="chat-456",
        thread_id="789",
        chat_type="forum",
    )


def test_message_received_hook_receives_full_message_context():
    runner = _make_runner()
    source = _make_source()
    message = "x" * 800

    result = asyncio.run(
        runner._emit_message_received_hook(
            source=source,
            session_id="session-1",
            message_text=message,
        )
    )

    assert result == message
    runner.hooks.emit.assert_awaited_once()
    event_type, context = runner.hooks.emit.await_args.args
    assert event_type == "message:received"
    assert context == {
        "platform": "telegram",
        "user_id": "user-123",
        "chat_id": "chat-456",
        "thread_id": "789",
        "chat_type": "forum",
        "session_id": "session-1",
        "message": message,
    }


def test_message_received_hook_can_mutate_message_text():
    runner = _make_runner()

    async def _mutate(_event_type, context):
        context["message"] = "rewritten by hook"

    runner.hooks.emit.side_effect = _mutate

    result = asyncio.run(
        runner._emit_message_received_hook(
            source=_make_source(),
            session_id="session-1",
            message_text="original",
        )
    )

    assert result == "rewritten by hook"


def test_message_received_hook_ignores_non_string_message_mutation():
    runner = _make_runner()

    async def _mutate(_event_type, context):
        context["message"] = {"unexpected": "value"}

    runner.hooks.emit.side_effect = _mutate

    result = asyncio.run(
        runner._emit_message_received_hook(
            source=_make_source(),
            session_id="session-1",
            message_text="original",
        )
    )

    assert result == "original"
