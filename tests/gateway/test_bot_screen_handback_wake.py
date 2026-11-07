"""Bot Screen handback uses the real Desktop authority FIFO and conversation pin."""

import asyncio

import pytest

from gateway.config import Platform
from gateway.wake import wake_route
from tests.gateway.test_local_producer_admission import _Gateway
from tests.waiting import async_wait_until

NOTICE = "[Hermes gateway notice: Bot Screen handed back]"


async def _desktop(tmp_path, monkeypatch):
    gateway = _Gateway(tmp_path, monkeypatch)
    await gateway.start()
    authority = gateway.primary
    desktop = gateway.connect(authority)
    ref = gateway.create(authority, desktop, "desktop", "gui")
    desktop = await gateway.watch(authority, ref.session_id)
    live = authority.sessions[ref.session_id]
    assert live.source.platform == Platform.LOCAL
    route = {"session_key": live.route, "parent_session_id": ref.session_id}
    return gateway, authority, desktop, ref, route


@pytest.mark.asyncio
async def test_local_handback_queues_behind_running_turn_without_interrupt(tmp_path, monkeypatch):
    gateway, authority, desktop, ref, route = await _desktop(tmp_path, monkeypatch)
    gateway.gate = asyncio.Event()
    interrupted = []
    try:
        reply = await desktop.dispatch({"id": 1, "method": "prompt.submit", "params": {
            "session_id": ref.session_id, "input_id": "human", "text": "finish the task"}})
        assert "result" in reply, reply
        await async_wait_until(lambda: gateway.turns, timeout=15, what="Desktop turn running")
        agent = gateway.runner._cached_agent_for(route["session_key"])
        if agent is not None:
            monkeypatch.setattr(agent, "interrupt", lambda *a, **k: interrupted.append(True))
        assert await wake_route(gateway.runner, route, NOTICE, lambda: True, producer="bot-screen-handback")
        assert [row["status"] for row in gateway.rows(authority, ref.session_id)] == ["started", "queued"]
        assert [turn.text for turn in gateway.turns] == ["finish the task"]
        assert not interrupted
        gateway.gate.set()
        await gateway.settled(authority, ref.session_id, 2)
        assert [turn.text for turn in gateway.turns] == ["finish the task", NOTICE]
        assert not interrupted
    finally:
        gateway.gate.set()
        await gateway.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["new", "compression"])
async def test_local_handback_pin_refuses_new_and_follows_compression(tmp_path, monkeypatch, transition):
    gateway, authority, desktop, ref, route = await _desktop(tmp_path, monkeypatch)
    try:
        parent = ref.session_id
        if transition == "new":
            row = authority.db.get_session(parent)
            reply = await desktop.dispatch({"id": 1, "method": "session.mutate", "params": {
                "session_id": parent, "request_id": "new", "operation": "reset", "payload": {},
                "expected_revision": row["runtime_revision"], "expected_generation": row["runtime_generation"]}})
            assert "result" in reply, reply
            assert reply["result"]["target_session_id"] != parent
        else:
            authority.db.append_message(parent, "user", "earlier task")
            child = parent + "-tip"
            authority.db.publish_compression_child(
                parent_session_id=parent, child_session_id=child, source="local",
                messages=[{"role": "user", "content": "summary"}], require_compression_lease=False)
            assert gateway.runner.session_store.advance_compression_session(route["session_key"], parent, child)
        accepted = await wake_route(gateway.runner, route, NOTICE, lambda: True, producer="bot-screen-handback")
        assert accepted is (transition == "compression")
        if accepted:
            await gateway.settled(authority, parent, 1)
            assert [turn.text for turn in gateway.turns] == [NOTICE]
        else:
            assert not gateway.rows(authority, parent)
            assert not gateway.turns
    finally:
        await gateway.stop()
