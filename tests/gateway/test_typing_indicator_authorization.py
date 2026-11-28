"""A sender the authorization gate will drop never sees a typing indicator (#149024).

``BasePlatformAdapter._process_message_background`` starts the typing refresh *before*
``_handle_message``, and the runner authorizes inside that handler. With Signal linked as
a secondary device on the owner's own number, every contact who messages the owner reaches
the adapter, so ``sendTyping`` went out to that contact — as the owner — and only then was
the message dropped. These tests drive the real turn path with a stubbed refresh loop, so
they assert "was the indicator started", not timing.
"""

import asyncio
import contextlib
import contextvars
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
@pytest.mark.parametrize("follow_up", [False, True])
async def test_unauthorized_sender_gets_no_typing_stop_either(follow_up):
    """The turn's cleanup stops typing too (Signal: ``sendTyping`` with ``stop``): none for a dropped sender,
    including the queued follow-up hand-off."""
    adapter = _adapter(runner=_StubRunner(authorized=False))
    adapter.stop_typing = AsyncMock()
    event, queued = _event(), _event()
    key = build_session_key(event.source)

    async def _turn(current):
        if follow_up and current is event:  # a second message from the same sender arrives mid-turn
            adapter._pending_messages[key] = queued
        await asyncio.sleep(0)

    adapter._message_handler = AsyncMock(side_effect=_turn)
    await _drive_turn(adapter, event)
    await asyncio.gather(*adapter._background_tasks)  # the follow-up's drain turn, cleanup included

    assert adapter._message_handler.await_count == (2 if follow_up else 1)
    assert adapter.stop_typing.await_count == 0


@pytest.mark.asyncio
async def test_authorized_sender_still_gets_typing_stop():
    adapter = _adapter(runner=_StubRunner(authorized=True))
    adapter.stop_typing = AsyncMock()

    await _drive_turn(adapter, _event(user_id=_OWNER))

    assert adapter.stop_typing.await_count >= 1


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


class _ReactingAdapter(_StubAdapter):
    """Records the processing hooks: Signal/Telegram/Discord send 👀/✅ reactions from them."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_calls = []

    async def on_processing_start(self, event):
        self.hook_calls.append("start")

    async def on_processing_complete(self, event, outcome):
        self.hook_calls.append("complete")


def _reacting_adapter(runner):
    adapter = _ReactingAdapter(
        PlatformConfig(enabled=True, token="t", typing_indicator=True), Platform.SIGNAL,
    )
    adapter._message_handler = AsyncMock(side_effect=_turn_body)
    adapter._keep_typing = AsyncMock(return_value=None)
    adapter.gateway_runner = runner
    return adapter


@pytest.mark.asyncio
async def test_unauthorized_sender_gets_no_processing_reactions():
    """The same pre-admission leak through the lifecycle hooks: 👀 before the drop, ✅ after."""
    adapter = _reacting_adapter(_StubRunner(authorized=False))

    await _drive_turn(adapter, _event())

    assert adapter.hook_calls == []
    assert adapter._message_handler.await_count == 1


@pytest.mark.asyncio
async def test_authorized_sender_keeps_processing_reactions():
    runner = _StubRunner(authorized=True)
    adapter = _reacting_adapter(runner)

    await _drive_turn(adapter, _event(user_id=_OWNER))

    assert adapter.hook_calls == ["start", "complete"]
    assert len(runner.sources) == 1  # one verdict per event, shared by typing and both hooks


class _LaunchScopedRunner(_StubRunner):
    """Admits the sender only inside the launch-profile scope the standalone handler runs under
    (``_standalone_scoped``): ambient ``os.environ`` is stale or foreign once multi-profile hosting is on."""

    _in_launch_scope = contextvars.ContextVar("in_launch_scope", default=False)

    @classmethod
    @contextlib.contextmanager
    def _standalone_launch_scope(cls):
        token = cls._in_launch_scope.set(True)
        try:
            yield
        finally:
            cls._in_launch_scope.reset(token)

    def _is_user_authorized_for_source(self, source):
        self.sources.append(source)
        return self._in_launch_scope.get()


@pytest.mark.asyncio
async def test_probe_reads_the_scope_the_handler_authorizes_under():
    """An admitted sender keeps typing and reactions when the launch scope and ambient env disagree."""
    adapter = _reacting_adapter(_LaunchScopedRunner())

    await _drive_turn(adapter, _event(user_id=_OWNER))

    assert adapter._keep_typing.await_count == 1
    assert adapter.hook_calls == ["start", "complete"]
