"""``_hermes-agent/state_update``: opt-in run boundaries and background roster for editors that
ask (ACP v2's ``state_update.background`` carried over the v1 connection as an extension
notification, modeled on ``_bb.dev/state_update``).

An editor opts in through ``initialize``'s ``clientCapabilities._meta`` — either the
``hermes-agent`` namespace or the ``bb.dev`` alias — with ``stateUpdate >= 1``. Everything rides
the existing claim stream: ``message.start`` (every admission, today dropped by the projection)
becomes ``running``, ``message.complete`` becomes ``idle``, and the roster is recomputed from the
owner's ``process.list`` ground truth on every send.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent


def _agent(state_update=True, processes=None, rpc_error=None):
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    agent._state_update = state_update
    gateway = MagicMock()
    if rpc_error is not None:
        gateway.rpc = AsyncMock(side_effect=rpc_error)
    else:
        gateway.rpc = AsyncMock(return_value={"processes": processes or []})
    agent._gateway = gateway
    return agent


def _observer_event(params):
    event = MagicMock()
    event.direction = "incoming"
    event.message = {"method": "initialize", "params": params}
    return event


def _meta(namespace="hermes-agent", version=1):
    return {"clientCapabilities": {"_meta": {namespace: {"stateUpdate": version}}}}


def _event(kind, sid="s", aid="a1", payload=None):
    return {"type": kind, "session_id": sid, "admission_id": aid,
            "replay_epoch": 1, "seq": 1, "payload": payload or {}}


def _state_calls(agent):
    return [call for call in agent._conn.ext_notification.await_args_list
            if call.args[0] == "hermes-agent/state_update"]


# --- opt-in --------------------------------------------------------------------

@pytest.mark.parametrize("params,expected", [
    (_meta(), True),
    (_meta(namespace="bb.dev"), True),
    (_meta(version=2), True),
    (_meta(version=0), False),
    ({"clientCapabilities": {}}, False),
    ({}, False),
    ({"clientCapabilities": {"_meta": {"hermes-agent": {}}}}, False),
])
def test_initialize_meta_opt_in(params, expected):
    agent = GatewayACPAgent()
    agent._observe_client_message(_observer_event(params))
    assert agent._state_update is expected


def test_non_initialize_messages_are_ignored():
    agent = GatewayACPAgent()
    event = MagicMock(); event.direction = "incoming"
    event.message = {"method": "session/new", "params": _meta()}
    agent._observe_client_message(event)
    assert agent._state_update is False


# --- boundaries ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_message_start_sends_running_with_roster():
    agent = _agent(processes=[{"session_id": "p1", "command": "make test", "status": "running",
                               "exit_code": None}])
    await agent._project(_event("message.start"))
    (call,) = _state_calls(agent)
    update = call.args[1]
    assert update["sessionId"] == "s" and update["state"] == "running"
    assert update["_meta"] == {"runId": "a1"}
    assert update["background"] == [{"type": "terminal", "id": "p1", "status": "running"}]


@pytest.mark.asyncio
async def test_message_complete_sends_idle_with_stop_reason():
    agent = _agent()
    await agent._project(_event("message.complete", payload={"outcome": "completed"}))
    (call,) = _state_calls(agent)
    assert call.args[1]["state"] == "idle" and call.args[1]["stopReason"] == "end_turn"


@pytest.mark.asyncio
async def test_cancelled_outcome_maps_to_cancelled_stop_reason():
    agent = _agent()
    await agent._project(_event("message.complete", payload={"outcome": "cancelled"}))
    (call,) = _state_calls(agent)
    assert call.args[1]["stopReason"] == "cancelled"


@pytest.mark.asyncio
async def test_no_opt_in_sends_nothing():
    agent = _agent(state_update=False)
    await agent._project(_event("message.start"))
    await agent._project(_event("message.complete", payload={"outcome": "completed"}))
    agent._conn.ext_notification.assert_not_awaited()


# --- roster semantics ----------------------------------------------------------

@pytest.mark.asyncio
async def test_exit_code_zero_maps_to_completed_nonzero_to_failed():
    agent = _agent(processes=[
        {"session_id": "ok", "command": "true", "status": "exited", "exit_code": 0},
        {"session_id": "bad", "command": "false", "status": "exited", "exit_code": 2}])
    await agent._project(_event("message.start"))
    (call,) = _state_calls(agent)
    assert {item["id"]: item["status"] for item in call.args[1]["background"]} == \
        {"ok": "completed", "bad": "failed"}
    assert next(i for i in call.args[1]["background"] if i["id"] == "bad")["exitCode"] == 2


@pytest.mark.asyncio
async def test_exited_process_held_until_its_wake_settles():
    exited = [{"session_id": "p1", "command": "sleep 3", "status": "exited", "exit_code": 0}]
    agent = _agent(processes=exited)
    await agent._project(_event("message.complete", payload={"outcome": "completed"}))
    roster = _state_calls(agent)[-1].args[1]["background"]
    assert [i["status"] for i in roster] == ["completed"]  # the idle still reports it

    await agent._project(_event("message.start", aid="wake-1",
                                payload={"display_kind": "process_complete"}))
    await agent._project(_event("message.complete", aid="wake-1", payload={"outcome": "completed"}))
    # The wake turn settled: the exited process leaves the roster for good, even once the
    # owner's finished registry evicts it.
    agent._gateway.rpc = AsyncMock(return_value={"processes": []})
    await agent._project(_event("message.start", aid="a2"))
    roster = _state_calls(agent)[-1].args[1]["background"]
    assert roster == []


@pytest.mark.asyncio
async def test_roster_read_failure_still_sends_the_boundary():
    agent = _agent(processes=[{"session_id": "p1", "command": "x", "status": "exited",
                               "exit_code": 0}])
    await agent._project(_event("message.start"))
    agent._gateway.rpc = AsyncMock(side_effect=RuntimeError("owner busy"))
    await agent._project(_event("message.complete", payload={"outcome": "completed"}))
    (call,) = _state_calls(agent)[-1:]
    assert call.args[1]["state"] == "idle"
    assert call.args[1]["background"] == [{"type": "terminal", "id": "p1",
                                           "status": "completed", "exitCode": 0}]


@pytest.mark.asyncio
async def test_notification_failure_never_breaks_the_pump():
    agent = _agent()
    agent._conn.ext_notification = AsyncMock(side_effect=RuntimeError("editor gone"))
    await agent._project(_event("message.start"))  # must not raise
    await agent._project(_event("message.complete", payload={"outcome": "completed"}))


# --- adopt ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adopt_sends_running_when_a_turn_is_in_flight():
    agent = _agent()
    agent._snapshots = {}
    snapshot = {"session_id": "s", "messages": [], "prompts": [],
                "pending": [{"admission_id": "live-1", "status": "started"}],
                "replay_epoch": 1, "last_sequence": 0}
    await agent._adopt("s", snapshot, client=agent._gateway)
    (call,) = _state_calls(agent)
    assert call.args[1]["state"] == "running" and call.args[1]["_meta"] == {"runId": "live-1"}


@pytest.mark.asyncio
async def test_adopt_sends_idle_without_a_gateway_omits_the_roster():
    agent = _agent()
    agent._snapshots = {}
    agent._gateway = None
    snapshot = {"session_id": "s", "messages": [], "prompts": [], "pending": [],
                "replay_epoch": 1, "last_sequence": 0}
    await agent._adopt("s", snapshot)
    (call,) = _state_calls(agent)
    assert call.args[1]["state"] == "idle"
    assert "background" not in call.args[1]  # unknown, never "quiet"
