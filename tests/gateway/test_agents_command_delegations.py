"""Gateway /agents surfaces background delegations with live activity (#51690).

Drives the REAL GatewayRunner._handle_agents_command against a REAL
async-delegation registry dispatch (no mocked list function), so the test
covers the whole projection: registry record → list_async_delegations()
live sampling → /agents rendering.
"""

import threading
import time

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._background_tasks = set()
    runner._session_key_for_source = lambda source: "agent:main:test:dm:1"
    return runner


class _Event:
    source = None


@pytest.mark.asyncio
async def test_agents_command_marks_stalling_delegation(monkeypatch):
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 0.1)
    # Force-finalization is a separate contract. Disable it so the stalling
    # projection stays observable until this test releases the worker.
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", float("inf"))
    gate = threading.Event()
    stalling = threading.Event()

    def blocked_runner():
        gate.wait()
        return {}

    res = ad.dispatch_async_delegation(
        goal="wedged child", context=None, toolsets=None, role="leaf",
        model="m", session_key="agent:main:test:dm:1", max_async_children=1,
        runner=blocked_runner,
        interrupt_fn=stalling.set,
        progress_fn=lambda: ((0, None), False),
    )
    assert res["status"] == "dispatched"

    try:
        assert stalling.wait(30.0), "delegation never reached stalling state"

        runner = _make_runner()
        out = await runner._handle_agents_command(_Event())
    finally:
        gate.set()

    assert res["delegation_id"] in out
    assert "stalling" in out
    assert "no progress" in out


@pytest.mark.asyncio
async def test_agents_lists_other_chats_work_only_for_a_configured_admin():
    """/agents counts every chat's work but names another chat's keys, commands and goals only to
    an explicitly configured admin (a session key is a routing handle, not authority)."""
    from types import SimpleNamespace

    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key

    owner = SessionSource(platform=Platform.TELEGRAM, chat_id="111", chat_type="dm", user_id="111")
    peer = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", user_id="222")
    same_room = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", user_id="333")
    owner_key = build_session_key(owner)
    runner = _make_runner()
    runner._session_key_for_source = build_session_key
    runner._running_agents = {owner_key: SimpleNamespace(session_id="OWNER-SID", model="m"),
                              build_session_key(same_room): SimpleNamespace(session_id="ROOM-SID", model="m")}
    proc = process_registry.spawn_local("sleep 30 # OWNER-CMD", task_id="t", session_key=owner_key)
    gate = threading.Event()
    ad.dispatch_async_delegation(goal="OWNER-GOAL", context=None, toolsets=None, role="leaf", model="m",
                                 session_key=owner_key, max_async_children=1,
                                 runner=lambda: gate.wait() and {})

    class _Ev:
        source = peer

    try:
        out = await runner._handle_agents_command(_Ev())
        for private in (owner_key, "OWNER-SID", "OWNER-CMD", "OWNER-GOAL"):
            assert private not in out
        assert "ROOM-SID" in out and "**Active agents:** 2" in out
        runner.config = SimpleNamespace(platforms={Platform.TELEGRAM: SimpleNamespace(
            extra={"group_allow_admin_from": ["222"]})})
        admin_out = await runner._handle_agents_command(_Ev())
        for private in (owner_key, "OWNER-SID", "OWNER-CMD", "OWNER-GOAL"):
            assert private in admin_out
    finally:
        gate.set()
        process_registry.kill_process(proc.id)


