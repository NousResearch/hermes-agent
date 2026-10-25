"""delegation.busy_result_mode=steer on the gateway (the runtime classic CLI now runs on): a subagent
result finishing while the session's turn runs rides that turn's next tool result instead of
becoming a new turn, and a result no tool batch carried still arrives as the next turn."""
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace

import pytest

from agent.background_results import BACKGROUND_RESULT_MARKER, attach_to_tool_tail, confirm_attached
from tests.gateway.test_completion_delivery import AdmittingHandler, _async_event, _runner


@pytest.fixture
def steering_runner(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module
    registry = pr_module.ProcessRegistry()
    registry.completion_queue = queue.Queue()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    monkeypatch.setattr("gateway.run._load_gateway_config",
                        lambda *a, **k: {"delegation": {"busy_result_mode": "steer"}})
    settled = []
    from tools import async_delegation
    for op in ("complete", "defer", "release"):
        monkeypatch.setattr(async_delegation, f"{op}_completion_delivery",
                            lambda delegation_id, _claim, op=op: settled.append((op, delegation_id)) or True)
    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    agent = SimpleNamespace(steer=lambda text: True)
    runner._peek_session_state = lambda key: SimpleNamespace(turn=SimpleNamespace(agent=agent))
    return runner, adapter, agent, registry, settled


def test_busy_result_rides_the_running_turns_tool_result_not_a_new_turn(steering_runner):
    runner, adapter, agent, _registry, settled = steering_runner

    delivered = asyncio.run(runner._deliver_completion_notification("subagent report", _async_event("deleg_a")))

    assert delivered is True
    adapter.handle_message.assert_not_awaited()  # no queued next turn
    messages = [{"role": "tool", "tool_call_id": "c1", "content": "tool output"}]
    attached = attach_to_tool_tail(agent, messages)
    assert len(attached) == 1
    assert BACKGROUND_RESULT_MARKER in messages[-1]["content"] and "subagent report" in messages[-1]["content"]
    confirm_attached(agent, attached, True)  # the tool-progress flush persisted this row
    asyncio.run(runner._settle_carried_completions())
    assert settled == [("complete", "deleg_a")]


def test_a_carry_that_lands_during_the_final_settle_is_still_settled(steering_runner):
    """run_turn settles while the agent slot is still held, so a delivery landing mid-settle sees
    that agent and registers a new carry; the final drain has to pick it up, not park its waiter."""
    from agent.background_results import attach_to_tool_tail, confirm_attached, offer
    from gateway.run_notifications_busy_result import CarriedCompletion

    runner, adapter, agent, _registry, settled = steering_runner

    assert asyncio.run(runner._deliver_completion_notification("subagent report", _async_event("deleg_a"))) is True
    confirm_attached(agent, attach_to_tool_tail(agent, [{"role": "tool", "tool_call_id": "c1", "content": "out"}]), True)

    original = runner._settle_carried_completion
    landed = []

    async def settle_and_land_another(carried, operation):
        await original(carried, operation)
        if landed:
            return
        landed.append(True)
        late = CarriedCompletion(_async_event("deleg_b"), "deleg_b", "claim_b", None, ())
        offer(agent, late, "late report")
        attached = attach_to_tool_tail(agent, [{"role": "tool", "tool_call_id": "c2", "content": "out"}])
        confirm_attached(agent, attached, True)

    runner._settle_carried_completion = settle_and_land_another
    asyncio.run(runner._settle_carried_completions(agent, final=True))

    assert settled == [("complete", "deleg_a"), ("complete", "deleg_b")]
    assert adapter.handle_message.await_count == 0  # both rode the turn, neither queued a new one


def test_result_the_turn_never_carried_returns_as_the_next_turn(steering_runner):
    runner, adapter, agent, registry, settled = steering_runner

    asyncio.run(runner._deliver_completion_notification("subagent report", _async_event("deleg_b")))
    asyncio.run(runner._settle_carried_completions(agent, final=True))  # turn ended, no tool call

    assert settled == [("defer", "deleg_b")]  # no delivery attempt spent
    requeued = registry.completion_queue.get_nowait()
    assert requeued["delegation_id"] == "deleg_b" and "completion_ids" not in requeued
    adapter.handle_message.assert_not_awaited()
