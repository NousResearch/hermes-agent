"""Behaviour of the gateway readers that moved off the legacy dict views
(_running_agents_ts, _busy_ack_ts, _queued_events, _pending_approvals) onto SessionState fields."""

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.i18n import t
from gateway.platforms.event import MessageEvent
from gateway.run import _AGENT_PENDING_SENTINEL
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=make_restart_source(), message_id="1")


@pytest.mark.asyncio
@pytest.mark.parametrize("approvals", [{"command": "rm -rf /tmp/x"}, {}], ids=["prompt", "empty-prompt"])
@pytest.mark.parametrize(("command", "stale_key"), [
    ("/approve", "gateway.approval_expired"),
    ("/deny", "gateway.deny.stale"),
])
async def test_stale_approval_prompt_is_reported_once_and_dropped(command, stale_key, approvals):
    runner, _adapter = make_restart_runner()
    event = _event(command)
    key = runner._session_key_for_source(event.source)
    runner._session_state(key).persistent.approvals = approvals
    handler = runner._handle_approve_command if command == "/approve" else runner._handle_deny_command

    assert await handler(event) == t(stale_key)
    assert runner._session_state(key).persistent.approvals is None
    assert await handler(event) != t(stale_key)


@pytest.mark.asyncio
async def test_approve_without_session_state_creates_no_row():
    runner, _adapter = make_restart_runner()
    event = _event("/approve")

    assert await runner._handle_approve_command(event) == t("gateway.approve.no_pending")
    assert runner._session_key_for_source(event.source) not in (runner.__dict__.get("_sessions") or {})


@pytest.mark.asyncio
async def test_agents_elapsed_comes_from_turn_started_ts():
    runner, _adapter = make_restart_runner()
    running = MagicMock(session_id="sid-1", model="m")
    runner._running_agents = {"agent:main:x:dm:1": running, "agent:main:x:dm:2": _AGENT_PENDING_SENTINEL}
    runner._session_state("agent:main:x:dm:1").turn.started_ts = time.time() - 3725

    lines = (await runner._handle_agents_command(_event("/agents"))).splitlines()

    assert any("agent:main:x:dm:1" in line and "1h 2m" in line for line in lines)
    assert any("agent:main:x:dm:2" in line and " 0s" in line for line in lines)


@pytest.mark.asyncio
async def test_stop_flushes_only_nonempty_overflow_and_resets_turn_timestamps():
    runner, _adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.0
    first, second = _event("first"), _event("second")
    runner._session_state("a").turn.started_ts = 5.0
    runner._session_state("a").turn.busy_ack_ts = 6.0
    runner._session_state("b").conversation.queued_events = [first, second]
    runner._session_state("c")
    flushed = []

    with (
        patch("gateway.status.remove_pid_file"),
        patch("gateway.status.publish_runtime_status"),
        patch("agent.auxiliary_client.shutdown_cached_clients"),
        patch("gateway.shutdown_flush.flush_overflow_to_file",
              side_effect=lambda mapping, **_kw: flushed.append(mapping)),
    ):
        await runner.stop()

    assert flushed == [{"b": [first, second]}]
    state = runner._sessions_map()["a"]
    assert (state.turn.started_ts, state.turn.busy_ack_ts) == (0.0, 0.0)
