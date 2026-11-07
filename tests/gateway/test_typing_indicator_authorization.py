"""A sender the authorization gate will drop never sees a typing indicator (#149024).

``BasePlatformAdapter._process_message_background`` starts the typing refresh *before*
``_handle_message``, and the runner authorizes inside that handler. With Signal linked as
a secondary device on the owner's own number, every contact who messages the owner reaches
the adapter, so ``sendTyping`` went out to that contact — as the owner — and only then was
the message dropped. These tests drive the real turn path with a stubbed refresh loop, so
they assert "was the indicator started", not timing.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key

_OWNER = "signal-owner-uuid"
_STRANGER = "signal-contact-uuid"


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


class _StubRunner:
    """The one seam this fix reads: the runner's authorization verdict."""

    def __init__(self, *, authorized=True, raises=None):
        self.authorized = authorized
        self.raises = raises
        self.sources = []

    def _is_user_authorized_for_source(self, source):
        self.sources.append(source)
        if self.raises is not None:
            raise self.raises
        return self.authorized


class _MessageHandlerRunner(_StubRunner):
    """A runner wired as the adapter's message handler, with no ``gateway_runner``."""

    async def _handle_message(self, event):
        pass


def _event(*, internal=False, user_id=_STRANGER):
    source = SessionSource(
        platform=Platform.SIGNAL, chat_id=user_id, chat_type="dm", user_id=user_id,
    )
    return MessageEvent(text="hello", message_type=MessageType.TEXT, source=source, internal=internal)


async def _turn_body(_event):
    """Yield once so a typing task spawned by the turn runs one step before its cancellation."""
    await asyncio.sleep(0)


def _adapter(*, runner=None, handler=None):
    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="t", typing_indicator=True), Platform.SIGNAL,
    )
    adapter._message_handler = handler or AsyncMock(side_effect=_turn_body)
    adapter._keep_typing = AsyncMock(return_value=None)  # the refresh loop itself: not under test
    if runner is not None:
        adapter.gateway_runner = runner
    return adapter


async def _drive_turn(adapter, event):
    key = build_session_key(event.source)
    adapter._active_sessions[key] = asyncio.Event()
    await adapter._process_message_background(event, key)


@pytest.mark.asyncio
async def test_unauthorized_sender_never_sees_typing():
    """The reported leak: the contact got ``sendTyping`` as the owner before the drop."""
    runner = _StubRunner(authorized=False)
    adapter = _adapter(runner=runner)

    await _drive_turn(adapter, _event())

    # The reported defect: the refresh loop ran for a sender the handler then dropped.
    assert adapter._keep_typing.await_count == 0
    assert runner.sources
    # Only the indicator is withheld: the turn still runs so the handler can log/report it.
    assert adapter._message_handler.await_count == 1


@pytest.mark.asyncio
async def test_authorized_sender_still_gets_typing():
    runner = _StubRunner(authorized=True)
    adapter = _adapter(runner=runner)

    await _drive_turn(adapter, _event(user_id=_OWNER))

    assert runner.sources
    assert adapter._keep_typing.await_count == 1


@pytest.mark.asyncio
async def test_internal_event_skips_the_authorization_probe():
    """Background-process notices are admitted without an allowlist: no probe, indicator kept."""
    runner = _StubRunner(authorized=False)
    adapter = _adapter(runner=runner)

    await _drive_turn(adapter, _event(internal=True))

    assert runner.sources == []
    assert adapter._keep_typing.await_count == 1


@pytest.mark.asyncio
async def test_runner_is_resolved_through_the_message_handler():
    """An adapter built outside ``_create_adapter`` has no ``gateway_runner``; its handler owns one."""
    runner = _MessageHandlerRunner(authorized=False)
    adapter = _adapter(handler=runner._handle_message)

    await _drive_turn(adapter, _event())

    assert runner.sources
    assert adapter._keep_typing.await_count == 0


@pytest.mark.asyncio
async def test_adapter_without_a_runner_keeps_typing():
    adapter = _adapter()

    await _drive_turn(adapter, _event())

    assert adapter._keep_typing.await_count == 1


@pytest.mark.asyncio
async def test_authorization_probe_error_keeps_typing():
    """Fail open: a broken verdict must not be the reason a conversation loses its indicator."""
    adapter = _adapter(runner=_StubRunner(raises=OSError("config backend unavailable")))

    await _drive_turn(adapter, _event())

    assert adapter._keep_typing.await_count == 1
