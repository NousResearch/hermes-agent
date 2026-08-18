"""Regression: _process_message_background must notify the user when a turn
raises a BaseException such as SystemExit/KeyboardInterrupt (#86651).

The handler is fire-and-forget (create_task, never awaited). Before the fix
its except chain caught only asyncio.CancelledError and Exception, so a
SystemExit escaping from a turn (e.g. a plugin/library calling sys.exit()
inside a tool call or the summary-LLM path) skipped the user-facing failure
notification and surfaced only as a "Task exception was never retrieved" log
line — radio silence for the user.

The fix catches BaseException: the failure notification is sent first, then
SystemExit/KeyboardInterrupt are re-raised so the loop's own shutdown
semantics are preserved, while other BaseExceptions stay contained.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _ProbeAdapter(BasePlatformAdapter):
    """Minimal concrete adapter that records deliveries."""

    def __init__(self) -> None:
        super().__init__(PlatformConfig(enabled=True, token="x"), Platform.SLACK)
        self.sent: list[str] = []

    async def start(self):  # pragma: no cover - unused
        pass

    async def stop(self):  # pragma: no cover - unused
        pass

    async def connect(self):  # pragma: no cover - unused
        pass

    async def disconnect(self):  # pragma: no cover - unused
        pass

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)

        class _R:
            success = True
            message_id = "m1"

        return _R()

    async def send_typing(self, chat_id, metadata=None):  # pragma: no cover - unused
        pass


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        user_id="U1",
        chat_id="C1",
        user_name="tester",
        chat_type="channel",
    )


def _make_adapter(handler) -> _ProbeAdapter:
    adapter = _ProbeAdapter()
    adapter.set_message_handler(handler)
    return adapter


def _raising_handler(exc: BaseException):
    async def handler(event):
        raise exc

    return handler


def _event() -> MessageEvent:
    return MessageEvent(text="hello", message_type=MessageType.TEXT, source=_source())


@pytest.mark.asyncio
async def test_system_exit_from_turn_notifies_user_and_is_reraised():
    """A SystemExit escaping a turn must still deliver the user-facing
    failure notification AND propagate (shutdown semantics preserved)."""
    adapter = _make_adapter(_raising_handler(SystemExit(1)))
    event = _event()

    with pytest.raises(SystemExit):
        await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent, "platform send must be called on SystemExit"
    assert "error" in adapter.sent[0].lower(), (
        f"notification should mention the error, got: {adapter.sent[0]!r}"
    )


@pytest.mark.asyncio
async def test_keyboard_interrupt_from_turn_notifies_user_and_is_reraised():
    """KeyboardInterrupt is handled like SystemExit: notify, then re-raise."""
    adapter = _make_adapter(_raising_handler(KeyboardInterrupt()))
    event = _event()

    with pytest.raises(KeyboardInterrupt):
        await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent, "platform send must be called on KeyboardInterrupt"


@pytest.mark.asyncio
async def test_plain_exception_still_notifies_without_propagating():
    """Ordinary Exception behavior is unchanged: notify, do NOT propagate."""
    adapter = _make_adapter(_raising_handler(RuntimeError("boom")))
    event = _event()

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent, "platform send must be called on a plain Exception"
    assert "error" in adapter.sent[0].lower()
