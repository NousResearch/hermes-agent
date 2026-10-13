"""Async-delegation delivery holds at the agent turn boundary (#137581).

A background fan-out completing while the target session still runs a turn used
to be injected as a concurrent turn: ``relay.begin_turn`` found the session
occupied and degraded its instrumentation, and the mid-turn synthetic message
read as corrupted tool output — verification loops, false pollution alarms.
The delivery lane now reads the same active-turn registry the agent registers
in and queues (durably, replayable) until the boundary; a probe failure must
fail open so the lane can never wedge.
"""

import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from tools import async_delegation as ad


class _HostlessRegistry:
    """Turn registration lives on the coordinator; the relay host is not needed."""

    def for_profile(self, profile_key, create=True):
        return None


def _pending_delegation(evt: dict) -> dict:
    ad._persist_dispatch(evt)
    ad._persist_completion(evt, {"status": "completed", "summary": evt["summary"]})
    return evt


@pytest.mark.asyncio
async def test_delegation_delivery_waits_for_relay_turn_boundary(
    tmp_path, monkeypatch, private_db_probe_cleanup,
):
    """A live turn on the target session holds the delivery at the boundary.

    Mid-turn the completion is neither admitted nor claimed (the durable row
    stays pending/replayable); after ``end_turn`` the same queued event
    delivers exactly once, and a replay dedupes.
    """
    with _profile_runtime_scope(tmp_path):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        store = runner.session_store
        entry = store.get_or_create_session(SessionSource(
            platform=Platform.TELEGRAM, chat_id="turn-boundary", chat_type="dm",
        ))
        owner = entry.session_id
        accepted = []

        async def accept(event):
            accepted.append(event.text)
            event._gateway_accepted = True

        runner.adapters[Platform.TELEGRAM] = cast(Any, SimpleNamespace(handle_message=accept))
        evt = _pending_delegation({
            "type": "async_delegation", "delegation_id": "boundary-hold",
            "session_key": entry.session_key, "parent_session_id": owner,
            "dispatched_at": 1.0, "status": "completed", "summary": "fan-out result",
        })

        from agent import relay_runtime
        coordinator = relay_runtime.SESSION_COORDINATOR
        monkeypatch.setattr(coordinator, "registry", _HostlessRegistry())
        lease = coordinator.acquire_conversation(
            profile_key=relay_runtime.current_profile_key(), session_id=owner, platform="telegram",
        )
        turn = coordinator.begin_turn(lease, turn_id="boundary-probe", task_id="boundary-probe")
        try:
            for _ in range(3):
                assert await runner._deliver_async_delegation_group([evt]) is False
            row = ad.get_durable_delegation(evt["delegation_id"])
            assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
            assert accepted == []
        finally:
            coordinator.end_turn(turn, outcome="completed")

        assert await runner._deliver_async_delegation_group([evt]) is True
        assert len(accepted) == 1 and "fan-out result" in accepted[0]
        assert ad.get_durable_delegation(evt["delegation_id"])["delivery_state"] == "delivered"
        assert await runner._deliver_async_delegation_group([evt]) is None


@pytest.mark.asyncio
async def test_relay_probe_failure_fails_open(
    tmp_path, monkeypatch, private_db_probe_cleanup,
):
    """An unreadable turn registry must not wedge the delivery lane.

    ``agent.relay_runtime`` import failure degrades the boundary gate to
    "no opinion": the completion delivers now, exactly as before the gate.
    """
    with _profile_runtime_scope(tmp_path):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        store = runner.session_store
        entry = store.get_or_create_session(SessionSource(
            platform=Platform.TELEGRAM, chat_id="turn-boundary-failopen", chat_type="dm",
        ))
        owner = entry.session_id
        accepted = []

        async def accept(event):
            accepted.append(event.text)
            event._gateway_accepted = True

        runner.adapters[Platform.TELEGRAM] = cast(Any, SimpleNamespace(handle_message=accept))
        evt = _pending_delegation({
            "type": "async_delegation", "delegation_id": "boundary-failopen",
            "session_key": entry.session_key, "parent_session_id": owner,
            "dispatched_at": 1.0, "status": "completed", "summary": "fan-out result",
        })

        from agent import relay_runtime
        import agent as agent_pkg
        coordinator = relay_runtime.SESSION_COORDINATOR
        monkeypatch.setattr(coordinator, "registry", _HostlessRegistry())
        lease = coordinator.acquire_conversation(
            profile_key=relay_runtime.current_profile_key(), session_id=owner, platform="telegram",
        )
        turn = coordinator.begin_turn(lease, turn_id="boundary-probe", task_id="boundary-probe")
        try:
            # A real import failure (missing relay wheel): hide both the module and the
            # already-bound package attribute so the gate's lazy import raises.
            monkeypatch.setitem(sys.modules, "agent.relay_runtime", None)
            monkeypatch.delattr(agent_pkg, "relay_runtime", raising=False)
            assert await runner._deliver_async_delegation_group([evt]) is True
            assert len(accepted) == 1
            assert ad.get_durable_delegation(evt["delegation_id"])["delivery_state"] == "delivered"
        finally:
            coordinator.end_turn(turn, outcome="completed")
