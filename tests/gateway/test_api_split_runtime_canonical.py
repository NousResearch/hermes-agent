"""Split-runtime client tools on a canonical (session-authority) /v1/runs admission.

The canonical turn builds and caches its agent inside the gateway's turn runner, not in
``_execute_run``: a run's client tools reach that agent through its admission observer, are
relayed back to the caller, and leave the cached agent again on a turn that supplies none."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.platforms import api_server as _api_server
from gateway.platforms.api_server_runs import _RunLaunch, _execute_run
from gateway.session_api_turn import admit_api_turn, wire_api_observers
from tests.waiting import async_wait_until

SET_TIMER = {"type": "function", "function": {
    "name": "set_timer", "description": "Set a timer on the device.",
    "parameters": {"type": "object", "properties": {"seconds": {"type": "integer"}}}}}
HOST_TOOL = {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}


def _cached_agent():
    """The attributes the turn runner's per-turn wiring and the relay touch on a real agent."""
    agent = SimpleNamespace(
        tools=[HOST_TOOL], valid_tool_names={"web_search"}, status_callback=None,
        interim_assistant_callback=None, reasoning_callback=None, invalidations=0)
    agent._invalidate_system_prompt = lambda: setattr(agent, "invalidations", agent.invalidations + 1)
    return agent


def _launch(api, admitted, run_id, client_tools):
    queue = asyncio.Queue()
    api._run_streams[run_id] = queue
    api._run_statuses[run_id] = {}
    return _RunLaunch(api, run_id, queue, admitted[1].session_id, None, False, "hello", [], False,
                      agent_kwargs={}, request_profile=None, browser_control_principal=None,
                      browser_control_transport_family=None, admission=admitted,
                      client_tools=client_tools), queue


def _turn_owner(owner, ref):
    return (owner, ref.session_id, owner.db.get_session(ref.session_id)["runtime_generation"])


def _names(agent):
    return [t["function"]["name"] for t in agent.tools]


@pytest.mark.asyncio
async def test_client_tool_is_relayed_on_a_canonical_turn_and_dropped_after(api, owner):
    from agent.inline_tool_executors import relay_client_tool
    from tools import client_tool_gateway

    agent = _cached_agent()
    first = admit_api_turn(api, session_id="device", user_message="set a timer", conversation_history=[])
    launch, queue = _launch(api, first, "run_one", [SET_TIMER])
    relayed = {}

    async def first_turn(event):
        from gateway.session_results import execution_result
        wire_api_observers(agent, _turn_owner(owner, first[1]), False)
        assert _names(agent) == ["web_search", "set_timer"]
        call = asyncio.ensure_future(asyncio.to_thread(
            relay_client_tool, agent, "set_timer", {"seconds": 3}, "call-1"))
        await async_wait_until(lambda: client_tool_gateway.has_pending("run_one"))
        assert client_tool_gateway.resolve_client_tool("run_one", "call-1", "timer set")
        relayed["result"] = await call
        execution_result.get()["result"] = {"final_response": "Done."}
        return "Done."

    owner.runner._handle_message = first_turn
    await _execute_run(api, launch, _api_server=_api_server)
    await asyncio.sleep(0)
    events = []
    while not queue.empty():
        item = queue.get_nowait()
        if item is not None:
            events.append(item)
    request = next(e for e in events if e["event"] == "tool_call.request")
    assert (request["call_id"], request["name"], request["arguments"]) == ("call-1", "set_timer", {"seconds": 3})
    assert relayed["result"] == "timer set"
    assert events[-1]["event"] == "run.completed"
    assert client_tool_gateway.get_notify("run_one") is None  # released with the run

    # The next turn of the same conversation reuses the cached agent but supplies no tools.
    second = admit_api_turn(api, session_id="device", user_message="thanks", conversation_history=[])
    launch, _ = _launch(api, second, "run_two", None)

    async def second_turn(event):
        from gateway.session_results import execution_result
        wire_api_observers(agent, _turn_owner(owner, second[1]), False)
        execution_result.get()["result"] = {"final_response": "ok"}
        return "ok"

    owner.runner._handle_message = second_turn
    await _execute_run(api, launch, _api_server=_api_server)
    assert _names(agent) == ["web_search"]
    assert agent.valid_tool_names == {"web_search"}


@pytest.mark.asyncio
async def test_memory_tool_started_on_a_canonical_turn_is_tagged_memory(api, owner):
    from gateway.run_turn_runner import TurnRunner

    agent = _cached_agent()
    agent._memory_manager = SimpleNamespace(get_all_tool_names=lambda: {"honcho_search"})
    admitted = admit_api_turn(api, session_id="recall", user_message="what do you know", conversation_history=[])
    launch, queue = _launch(api, admitted, "run_mem", None)

    async def turn(event):
        from gateway.session_results import execution_result
        runner = object.__new__(TurnRunner)
        runner._ctx = SimpleNamespace(_native_slack_task_cards=False, _voice_ack_guild=[None], progress_mode="off")
        runner._approval_owner = _turn_owner(owner, admitted[1])
        wire_api_observers(agent, runner._approval_owner, False)
        runner.combined_tool_start_callback("call-m", "honcho_search", {"query": "x"})
        runner.combined_tool_start_callback("call-w", "web_search", {"query": "x"})
        execution_result.get()["result"] = {"final_response": "ok"}
        return "ok"

    owner.runner._handle_message = turn
    await _execute_run(api, launch, _api_server=_api_server)
    await asyncio.sleep(0)
    started = []
    while not queue.empty():
        item = queue.get_nowait()
        if item is not None and item["event"] == "tool.started":
            started.append((item["tool"], item.get("kind")))
    assert started == [("honcho_search", "memory"), ("web_search", None)]
    assert "run_mem" not in api._run_memory_tool_names  # released with the run


def test_unchanged_client_tools_keep_the_cached_system_prompt():
    from gateway.platforms.api_server_client_tools import apply_client_tools

    agent = _cached_agent()
    apply_client_tools(agent, [SET_TIMER], "run_one")
    apply_client_tools(agent, [SET_TIMER], "run_two")
    assert agent.invalidations == 1  # the second turn's identical set is not a prompt change
    assert agent._client_tool_session_key == "run_two"  # but its relays belong to the new run


def test_shadowed_host_tool_returns_when_the_client_stops_supplying_it():
    from gateway.platforms.api_server_client_tools import apply_client_tools

    agent = _cached_agent()
    client_search = {"type": "function", "function": {"name": "web_search", "description": "on device"}}
    apply_client_tools(agent, [client_search], "run_one")
    assert agent.tools == [client_search] and agent._client_tool_names == {"web_search"}
    apply_client_tools(agent, None, "run_two")
    assert agent.tools == [HOST_TOOL] and agent._client_tool_names == set()
