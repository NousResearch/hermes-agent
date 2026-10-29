"""Regression: an inline (authority-drain) turn must drain the follow-up its hook queued.

A turn started by ``admit_automation`` runs inline in the session authority's drain task, so it owns
no adapter session task.  Its post-turn hook can still queue the next turn there -- ``/goal``'s
continuation prompt goes through ``_enqueue_fifo`` into ``adapter._pending_messages`` -- and the only
drains of that slot live in an adapter task's tail or in the next inbound message.  The continuation
therefore sat in the slot until the user sent something unrelated (typically ``/context``).

``deliver_settled`` is the lane that sends such a turn's reply, and the one point that runs after the
whole turn; it now hands the slot to the adapter's own session lane, exactly as a finishing adapter
task does at its tail.  Only a live owner task in ``_session_tasks`` suppresses that handoff:
``_active_sessions`` holds guards, and a command scope installs one without draining anything.
"""

from types import SimpleNamespace

import pytest

from gateway import session_ingress
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


class _Task:
    """Stand-in for the owner task recorded in ``_session_tasks``."""

    def __init__(self, done):
        self._done = done

    def done(self):
        return self._done


class _Adapter:
    """Minimal adapter surface used by the drain helper."""

    def __init__(self, pending=None, live=(), finished=()):
        self._pending_messages = dict(pending or {})
        self._session_tasks = {key: _Task(done=False) for key in live}
        self._session_tasks.update({key: _Task(done=True) for key in finished})
        # A command scope installs a guard without owning a task; it must not block the drain.
        self._active_sessions = {}
        self.spawned = []
        self.fail_spawn = False

    def _spawn_drain_task(self, pending_event, session_key, delay=0.0):
        if self.fail_spawn:
            raise RuntimeError("spawn refused")
        self.spawned.append((pending_event, session_key, delay))


def _event(text="[Continuing toward your standing goal] finish the task"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1"),
    )


def _authority(adapter, event, session_key, response="the reply"):
    return SimpleNamespace(
        profile_id="test-profile",
        pending_deliveries={"a1": (adapter, event, session_key, response, None)},
        native_waiters=set(),
        waiters={},
        held_turn_markers={},
        marker_adopters={},
        runner=SimpleNamespace(
            config=SimpleNamespace(multiplex_profiles=False),
            _clear_durable_active_turn=None,
        ),
    )


@pytest.mark.asyncio
async def test_inline_turn_resumes_the_queued_continuation(monkeypatch):
    """The regression: the continuation queued by an inline turn's hook must run without a nudge."""
    queued = _event()
    adapter = _Adapter(pending={"s": queued})
    sent = []

    async def fake_deliver(a, ev, key, response):
        sent.append((key, response))

    monkeypatch.setattr(session_ingress, "deliver_response", fake_deliver)
    authority = _authority(adapter, _event("the turn's own reply"), "s")

    await session_ingress.deliver_settled(authority, "a1")

    assert sent == [("s", "the reply")]          # the reply still goes out first
    assert adapter._pending_messages == {}       # the slot is drained
    assert adapter.spawned == [(queued, "s", 0.0)]


@pytest.mark.asyncio
async def test_a_live_adapter_task_keeps_ownership_of_the_slot(monkeypatch):
    """A user-initiated turn already owns the slot: its own tail drains it, so do not race it."""
    queued = _event()
    adapter = _Adapter(pending={"s": queued}, live=("s",))

    async def fake_deliver(a, ev, key, response):
        return None

    monkeypatch.setattr(session_ingress, "deliver_response", fake_deliver)
    authority = _authority(adapter, _event(), "s")

    await session_ingress.deliver_settled(authority, "a1")

    assert adapter.spawned == []
    assert adapter._pending_messages == {"s": queued}


@pytest.mark.asyncio
async def test_no_queued_followup_is_a_noop(monkeypatch):
    adapter = _Adapter()

    async def fake_deliver(a, ev, key, response):
        return None

    monkeypatch.setattr(session_ingress, "deliver_response", fake_deliver)
    authority = _authority(adapter, _event(), "s")

    await session_ingress.deliver_settled(authority, "a1")

    assert adapter.spawned == []


@pytest.mark.asyncio
async def test_a_finished_owner_task_does_not_block_the_drain(monkeypatch):
    """``_session_tasks`` keeps the exited task: only a live one means a tail will drain."""
    queued = _event()
    adapter = _Adapter(pending={"s": queued}, finished=("s",))

    async def fake_deliver(a, ev, key, response):
        return None

    monkeypatch.setattr(session_ingress, "deliver_response", fake_deliver)
    authority = _authority(adapter, _event(), "s")

    await session_ingress.deliver_settled(authority, "a1")

    assert adapter.spawned == [(queued, "s", 0.0)]


@pytest.mark.asyncio
async def test_a_guard_without_an_owner_task_does_not_block_the_drain(monkeypatch):
    """A command scope installs a guard but drains nothing on exit."""
    queued = _event()
    adapter = _Adapter(pending={"s": queued})
    adapter._active_sessions["s"] = object()

    async def fake_deliver(a, ev, key, response):
        return None

    monkeypatch.setattr(session_ingress, "deliver_response", fake_deliver)
    authority = _authority(adapter, _event(), "s")

    await session_ingress.deliver_settled(authority, "a1")

    assert adapter.spawned == [(queued, "s", 0.0)]


@pytest.mark.asyncio
async def test_a_refused_spawn_puts_the_event_back(monkeypatch):
    """Never drop a queued turn: a later inbound message still drains the slot."""
    queued = _event()
    adapter = _Adapter(pending={"s": queued})
    adapter.fail_spawn = True

    async def fake_deliver(a, ev, key, response):
        return None

    monkeypatch.setattr(session_ingress, "deliver_response", fake_deliver)
    authority = _authority(adapter, _event(), "s")

    await session_ingress.deliver_settled(authority, "a1")

    assert adapter._pending_messages == {"s": queued}


@pytest.mark.asyncio
async def test_a_waiter_owned_delivery_does_not_touch_the_slot(monkeypatch):
    """A native waiter's own adapter task sends the reply and drains the slot at its tail."""
    queued = _event()
    adapter = _Adapter(pending={"s": queued})

    async def fake_deliver(a, ev, key, response):
        raise AssertionError("the drain must not send a waiter-owned reply")

    monkeypatch.setattr(session_ingress, "deliver_response", fake_deliver)
    authority = _authority(adapter, _event(), "s")
    authority.native_waiters.add("a1")

    await session_ingress.deliver_settled(authority, "a1")

    assert adapter.spawned == []
    assert adapter._pending_messages == {"s": queued}
