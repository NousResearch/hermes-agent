"""A person's steer, /stop or /new reports itself to a plugin turn's background wait from the gateway
loop (``person_input_joined_turn``): with no plugin turn waiting it never reads SessionDB, and with
one waiting the owner lookup runs in a worker, never on the loop, and still ends the wait. The
/stop path (``_interrupt_and_clear_session``) is one of those reports."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until


def _runner(*, waiting):
    readers = []
    wait = threading.Event()
    if not waiting:
        wait.set()  # a plugin turn whose wait already ended

    def logical_owner(session_id):
        readers.append(threading.current_thread())
        return session_id

    authority = SimpleNamespace(profile_id="default", sessions={"sid": SimpleNamespace(background_wait=wait, task=None)},
                                logical_owner=logical_owner)
    store = SimpleNamespace(peek_session_id=lambda key: "sid")
    return SimpleNamespace(session_authority=authority, session_store=store), wait, readers


@pytest.mark.asyncio
async def test_no_waiting_plugin_turn_means_no_sessiondb_read():
    from gateway.run_plugin_injection import person_input_joined_turn

    runner, _wait, readers = _runner(waiting=False)
    person_input_joined_turn(runner, "key")

    assert readers == []


@pytest.mark.asyncio
async def test_a_waiting_plugin_turn_is_ended_with_the_owner_lookup_off_the_loop():
    from gateway.run_plugin_injection import person_input_joined_turn

    runner, wait, readers = _runner(waiting=True)
    person_input_joined_turn(runner, "key")
    await async_wait_until(wait.is_set, timeout=5, what="the wait ended")

    assert readers and threading.current_thread() not in readers


@pytest.mark.asyncio
async def test_stop_through_interrupt_and_clear_session_ends_the_wait(monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.config import Platform

    gateway = GatewayRunner(GatewayConfig())
    fake, wait, _readers = _runner(waiting=True)
    gateway.session_authority = fake.session_authority
    monkeypatch.setattr(gateway, "session_authorities", None, raising=False)
    monkeypatch.setattr(gateway.session_store, "peek_session_id", fake.session_store.peek_session_id)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id="1")
    await gateway._interrupt_and_clear_session("key", source, interrupt_reason="stop",
                                               invalidation_reason="stop_command")
    await async_wait_until(wait.is_set, timeout=5, what="the wait ended")
