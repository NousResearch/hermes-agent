"""delegation.busy_result_mode=steer on the gateway (the runtime classic CLI now runs on): a subagent
result finishing while the session's turn runs rides that turn's next tool result instead of
becoming a new turn, and a result no tool batch carried still arrives as the next turn."""
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace

import pytest

from agent.background_results import BACKGROUND_RESULT_MARKER, attach_to_tool_tail
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
    assert attach_to_tool_tail(agent, messages) == 1
    assert BACKGROUND_RESULT_MARKER in messages[-1]["content"] and "subagent report" in messages[-1]["content"]
    asyncio.run(runner._settle_carried_completions())
    assert settled == [("complete", "deleg_a")]


def test_result_the_turn_never_carried_returns_as_the_next_turn(steering_runner):
    runner, adapter, agent, registry, settled = steering_runner

    asyncio.run(runner._deliver_completion_notification("subagent report", _async_event("deleg_b")))
    asyncio.run(runner._settle_carried_completions(agent, final=True))  # turn ended, no tool call

    assert settled == [("defer", "deleg_b")]  # no delivery attempt spent
    requeued = registry.completion_queue.get_nowait()
    assert requeued["delegation_id"] == "deleg_b" and "completion_ids" not in requeued
    adapter.handle_message.assert_not_awaited()
