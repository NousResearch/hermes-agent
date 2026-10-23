"""display.background_process_busy_result_mode=steer: a notify_on_complete process completion that
finishes while the session's turn runs rides that turn's next tool result instead of becoming a new
turn; one no tool batch carried still arrives as the next turn, and one the agent already read with
process(wait/log) is not shown again."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent.background_results import PROCESS_RESULT_MARKER, attach_to_tool_tail
from tests.gateway.test_completion_delivery import AdmittingHandler, _completion_event, _runner


@pytest.fixture
def steering_runner(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    # Through the loader the gateway reads at delivery time, not a patched dict.
    (tmp_path / "config.yaml").write_text("display:\n  background_process_busy_result_mode: steer\n")
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    adapter = SimpleNamespace(handle_message=AdmittingHandler())
    runner = _runner(adapter)
    agent = SimpleNamespace(steer=lambda text: True)
    runner._peek_session_state = lambda key: SimpleNamespace(turn=SimpleNamespace(agent=agent))
    return runner, adapter, agent, registry


def _tool_tail():
    return [{"role": "tool", "tool_call_id": "c1", "content": "tool output"}]


async def _deliver_while_busy(runner, agent, during_turn):
    """Deliver one completion while ``agent``'s turn runs; ``during_turn`` plays the rest of that turn."""
    delivery = asyncio.create_task(
        runner._deliver_completion_notification("process report", _completion_event(started_at=1.0)))
    for _ in range(5):
        await asyncio.sleep(0)  # let the delivery reach the running turn
    assert not delivery.done(), "a carried completion waits for the turn's outcome"
    await during_turn()
    return await asyncio.wait_for(delivery, 5)


def test_busy_completion_rides_the_running_turns_tool_result_not_a_new_turn(steering_runner):
    runner, adapter, agent, _registry = steering_runner
    messages = _tool_tail()

    async def next_tool_batch():
        assert attach_to_tool_tail(agent, messages) == 1
        await runner._settle_carried_completions()  # the watcher tick

    assert asyncio.run(_deliver_while_busy(runner, agent, next_tool_batch)) is True
    assert PROCESS_RESULT_MARKER in messages[-1]["content"] and "process report" in messages[-1]["content"]
    adapter.handle_message.assert_not_awaited()  # no queued next turn


def test_completion_the_turn_never_carried_arrives_as_the_next_turn(steering_runner):
    runner, adapter, agent, _registry = steering_runner

    async def turn_ends_without_a_tool_call():
        await runner._settle_carried_completions(agent, final=True)

    assert asyncio.run(_deliver_while_busy(runner, agent, turn_ends_without_a_tool_call)) is True
    adapter.handle_message.assert_awaited_once()
    assert attach_to_tool_tail(agent, _tool_tail()) == 0  # never also shown on a later tool result


def test_completion_already_read_with_process_wait_is_not_shown_again(steering_runner):
    runner, adapter, agent, registry = steering_runner
    messages = _tool_tail()

    async def agent_waits_on_the_process_then_calls_a_tool():
        registry._completion_consumed.add("proc_reused")
        assert attach_to_tool_tail(agent, messages) == 0
        await runner._settle_carried_completions()

    assert asyncio.run(_deliver_while_busy(runner, agent, agent_waits_on_the_process_then_calls_a_tool)) is True
    assert messages[-1]["content"] == "tool output"
    adapter.handle_message.assert_not_awaited()
