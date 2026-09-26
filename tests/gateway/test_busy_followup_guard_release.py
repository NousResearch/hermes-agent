"""Regression for #120999: a busy follow-up must not be parked when the turn it
queued behind finishes while the runner's busy handler is still deciding.

The base adapter routes a message to ``_handle_message_while_active`` because
the previous turn's task still holds the session guard, and awaits the
runner's busy handler.  The handler awaits before it decides: in the stock
``interrupt`` mode ``_resolve_busy_steer_or_redirect`` reads the compression
lock (two ``asyncio.to_thread`` hops), and with multiplexed profiles the
wrapper first loads the profile's secret scope in a worker thread.  If the
previous turn reaches ``_finish_session_task`` meanwhile, it finds the slot
empty, releases the guard and exits.  The follow-up is then queued (by the
runner's FIFO, or by the base adapter when the handler returns False) where no
task owns it.  Nothing drains it: stale-lock healing needs a guard, the FIFO
orphan rescue (#99882) only covers overflow, and the next inbound message
starts a fresh turn that runs BEFORE the parked one.  #28649 closed the same
hole for one caller (the /goal continuation); this pins it for the busy path.
The handler can also raise after it stored the follow-up; the base adapter
must then treat it as accepted, not start it and queue it again (two turns).

Real: ``BasePlatformAdapter`` dispatch/turn/cleanup, and the ``GatewayRunner``
busy path wired by ``_wire_adapter_handlers`` with a real ``SessionStore``.
Fake: the agent turn (records which message started it) and one worker-thread
read on the busy path, which blocks on an event so the test decides when the
busy handler resumes.
"""

import asyncio
import threading

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


class _Adapter(BasePlatformAdapter):
    """Concrete adapter whose final-reply send can be held open."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.SIGNAL)
        self.sent: list[str] = []
        self.hold_reply = None
        self.reply_held = asyncio.Event()
        self.release_reply = asyncio.Event()

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if content == self.hold_reply:
            self.reply_held.set()
            await self.release_reply.wait()
        self.sent.append(content)
        return SendResult(success=True, message_id=f"out-{len(self.sent)}")


class _Turns:
    """Stands in for the agent turn: records the order messages started turns."""

    def __init__(self):
        self.order: list[str] = []
        self._ran: dict[str, asyncio.Event] = {}

    def ran(self, text: str) -> asyncio.Event:
        return self._ran.setdefault(text, asyncio.Event())

    async def __call__(self, event: MessageEvent):
        self.order.append(event.text)
        self.ran(event.text).set()
        return f"reply to {event.text}"


class _HeldRead:
    """A read the busy path makes in a worker thread: it blocks until released
    and says when it has been entered."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self.entered = asyncio.Event()
        self._released = threading.Event()

    def block(self):
        self._loop.call_soon_threadsafe(self.entered.set)
        self._released.wait(timeout=30)

    def get_compression_lock_holder(self, session_id):
        self.block()
        return None  # no compression in flight: the ordinary interrupt-mode path

    def release(self):
        self._released.set()


def _stock_interrupt(runner, monkeypatch, held):
    """Stock default: the handler queues the follow-up itself (returns True)
    after the compression-lock read."""
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._session_db = held


def _multiplexed_queue_text(runner, monkeypatch, held):
    """Multiplexed profiles with busy_text_mode queue: the handler returns
    False after the profile-scope load and the base adapter queues it."""
    runner.config.multiplex_profiles = True
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "queue"
    runner._session_db = None
    load_scope = gateway_run._load_profile_secret_scope

    def _held_load(profile_home):
        held.block()
        return load_scope(profile_home)

    monkeypatch.setattr(gateway_run, "_load_profile_secret_scope", _held_load)


def _stock_interrupt_ack_raises(runner, monkeypatch, held):
    """Stock default, but the handler raises after it queued the follow-up (composing the busy
    ack fails), so the base adapter sees an exception for an event that is already stored."""
    _stock_interrupt(runner, monkeypatch, held)

    def _raise(self, *_args, **_kwargs):
        raise RuntimeError("busy ack composition failed")

    monkeypatch.setattr(GatewayRunner, "_compose_busy_ack_message", _raise)


ROUTES = pytest.mark.parametrize(
    "route", [_stock_interrupt, _multiplexed_queue_text], ids=lambda r: r.__name__[1:]
)


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15550001",
            chat_type="dm",
            user_id="+15550001",
            user_name="owner",
        ),
        message_id=f"in-{text}",
    )


def _gateway(tmp_path, monkeypatch, route, held):
    adapter = _Adapter()
    turns = _Turns()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.SIGNAL: adapter}
    runner._draining = False
    runner._restart_requested = False
    runner._is_user_authorized = lambda _source: True
    runner.session_store = SessionStore(
        sessions_dir=tmp_path / "sessions", config=runner.config
    )
    runner.session_store._db = SessionDB(db_path=tmp_path / "state.db")
    route(runner, monkeypatch, held)
    runner._wire_adapter_handlers(adapter, message_handler=turns)
    key = adapter._event_session_key(_event("probe"))
    # A returning user: the session exists, so the compression check reads its lock.
    runner.session_store.get_or_create_session(_event("probe").source)
    assert runner.session_store._entries.get(key) is not None
    return adapter, turns, key


async def _followup_accepted_while_turn_unwinds(adapter, key, held):
    """M1's turn is delivering its reply (guard held) when M2 arrives; M1's task
    then finishes while M2's busy handler is blocked in its worker-thread read."""
    adapter.hold_reply = "reply to M1"
    await adapter.handle_message(_event("M1"))
    await asyncio.wait_for(adapter.reply_held.wait(), timeout=5)
    first_turn = adapter._session_tasks[key]
    assert key in adapter._active_sessions

    followup = asyncio.create_task(adapter.handle_message(_event("M2")))
    await asyncio.wait_for(held.entered.wait(), timeout=5)

    adapter.release_reply.set()
    await asyncio.wait_for(asyncio.shield(first_turn), timeout=5)
    held.release()
    await asyncio.wait_for(followup, timeout=5)


async def _run_until_idle(adapter, key):
    """Await the session's owner tasks until none is left (a drain hands off to a fresh task)."""
    while (owner := adapter._session_tasks.get(key)) is not None and not owner.done():
        await asyncio.wait_for(asyncio.shield(owner), timeout=5)


def _orphaned(adapter, key) -> bool:
    """A follow-up is queued (slot or queue-text buffer) but no live task owns the session."""
    owner = adapter._session_tasks.get(key)
    queued = key in adapter._pending_messages or key in adapter._text_debounce_store()
    return queued and (
        key not in adapter._active_sessions or owner is None or owner.done()
    )


@ROUTES
@pytest.mark.asyncio
async def test_followup_accepted_as_turn_ends_is_owned_and_runs_before_a_later_message(
    tmp_path, monkeypatch, route
):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    held = _HeldRead(asyncio.get_running_loop())
    adapter, turns, key = _gateway(tmp_path, monkeypatch, route, held)
    try:
        await _followup_accepted_while_turn_unwinds(adapter, key, held)

        assert not _orphaned(adapter, key), (
            "busy follow-up parked with no session guard and no live owner task "
            f"(guard={key in adapter._active_sessions}, "
            f"owner={adapter._session_tasks.get(key)!r}); nothing will run it"
        )
        await adapter.handle_message(_event("M3"))
        await asyncio.wait_for(turns.ran("M2").wait(), timeout=5)
        await asyncio.wait_for(turns.ran("M3").wait(), timeout=5)
        assert turns.order == ["M1", "M2", "M3"], (
            "the follow-up accepted during M1's turn ran after a message sent later"
        )
    finally:
        adapter.release_reply.set()
        held.release()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_followup_stored_before_the_busy_handler_raised_runs_once(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    held = _HeldRead(asyncio.get_running_loop())
    adapter, turns, key = _gateway(
        tmp_path, monkeypatch, _stock_interrupt_ack_raises, held
    )
    try:
        await _followup_accepted_while_turn_unwinds(adapter, key, held)

        await _run_until_idle(adapter, key)
        assert turns.order == ["M1", "M2"], (
            "the follow-up the busy handler stored before raising must run exactly "
            "once: it was left without an owner, or started and then queued again"
        )
        assert adapter.sent == ["reply to M1", "reply to M2"]
        assert not _orphaned(adapter, key)
    finally:
        adapter.release_reply.set()
        held.release()
        await adapter.cancel_background_tasks()
