"""Load-bearing tests for the platform-neutral Gateway lifecycle seam."""

import asyncio
from dataclasses import fields

import pytest

from gateway.turn_observer import GatewayTurnObserver, TurnLifecycleEvent


def test_event_contract_is_closed_and_platform_neutral():
    names = {field.name for field in fields(TurnLifecycleEvent)}
    assert names == {
        "phase",
        "platform",
        "profile",
        "channel_id",
        "session_id",
        "turn_id",
        "started_at",
        "triggering_event_id",
        "is_new_session",
        "tool_call_id",
        "tool_name",
        "tool_status",
        "outcome",
    }
    assert not any("buzz" in name.lower() for name in names)
    assert "metadata" not in names


@pytest.mark.asyncio
async def test_observer_uses_adapter_scoped_seam_and_fails_open():
    received = []

    class Route:
        def on_turn_lifecycle(self, event):
            received.append(event)
            raise RuntimeError("observer failure must not break chat")

    route = Route()
    observer = GatewayTurnObserver(
        platform="buzz",
        profile="default",
        channel_id="channel-1",
        session_id="session-1",
        triggering_event_id=None,
        is_new_session=False,
        route=route,
        loop=asyncio.get_running_loop(),
        is_current=lambda: True,
    )

    assert observer.start(liveness_interval=0) is False
    assert len(received) == 1
    assert received[0].phase == "turn_started"


@pytest.mark.asyncio
async def test_tool_identifiers_are_bounded_opaque_and_privacy_safe():
    received = []

    class Route:
        def on_turn_lifecycle(self, event):
            received.append(event)
            return True

    observer = GatewayTurnObserver(
        platform="buzz",
        profile="default",
        channel_id="channel-1",
        session_id="session-1",
        triggering_event_id=None,
        is_new_session=False,
        route=Route(),
        loop=asyncio.get_running_loop(),
        is_current=lambda: True,
    )
    observer.start(liveness_interval=0)
    secret_id = "private-token/" + "x" * 5000
    secret_name = "terminal\n/private/path/" + "y" * 5000

    observer.tool_started(secret_id, secret_name, {"password": "do-not-copy"})
    observer.tool_finished(secret_id, secret_name, {}, "private result")

    tool_events = [event for event in received if event.phase.startswith("tool_")]
    assert len(tool_events) == 2
    assert tool_events[0].tool_call_id == tool_events[1].tool_call_id == "tool-1"
    assert tool_events[0].tool_name == tool_events[1].tool_name == "tool"
    assert len(tool_events[0].tool_call_id) <= 32
    assert len(tool_events[0].tool_name) <= 64
    serialized = repr(tool_events)
    assert "private-token" not in serialized
    assert "/private/path" not in serialized
    assert "do-not-copy" not in serialized
