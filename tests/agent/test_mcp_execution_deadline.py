"""A server-owned deadline must survive the outer executor without widening a sibling batch."""
import json
import threading
from types import SimpleNamespace

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

from tools.registry import registry


def test_registered_deadline_survives_sequential_execution_and_bridge(tmp_path):
    from agent.tool_executor import execute_tool_calls_sequential
    from hermes_cli.config import atomic_config_write, get_config_path
    from run_agent import AIAgent

    atomic_config_write(get_config_path(), {"timeouts": {"tools": {"sequential_call": 2}}})
    release = threading.Event()
    finished = threading.Event()
    timers = []

    def blocked_tool(args, **kwargs):
        timer = threading.Timer(4, release.set)
        timers.append(timer)
        timer.start()
        release.wait(15)
        finished.set()
        return "released"

    names = ("deadline_probe_slow", "deadline_probe_ordinary")
    for name, budget in zip(names, (20, None)):
        registry.register(name=name, toolset="deadline-probe", schema={
            "name": name, "description": "Wait for the test-owned event.",
            "parameters": {"type": "object", "properties": {}}},
            handler=blocked_tool, execution_timeout=budget)
    agent = AIAgent(model="test-model", provider="openai", api_key="unused-test-key",
                    base_url="http://127.0.0.1:9/v1", enabled_toolsets=["deadline-probe"],
                    quiet_mode=True, skip_context_files=True, skip_memory=True,
                    save_trajectories=False)
    try:
        cases = [(names[0], {}, True),
                 ("tool_call", {"calls": [{"name": names[0], "arguments": {}}]}, True),
                 (names[1], {}, False)]
        for index, (name, args, succeeds) in enumerate(cases):
            release.clear()
            finished.clear()
            try:
                call = ChatCompletionMessageToolCall.model_validate({
                    "id": f"deadline-{index}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}})
                message = ChatCompletionMessage(role="assistant", content=None, tool_calls=[call])
                messages = []
                execute_tool_calls_sequential(agent, message, messages, "deadline-probe", finalize=False)
                result = messages[-1]["content"]
                if succeeds:
                    assert "released" in result, result
                    assert "timed out" not in result
                else:
                    assert "timed out after 2.0s" in result, result
            finally:
                release.set()
                assert finished.wait(5)
                for timer in timers:
                    timer.cancel()
                    timer.join()
                timers.clear()
    finally:
        for name in names:
            registry.deregister(name)


def test_extended_deadline_is_a_barrier_for_direct_and_bridged_batches():
    # Even a normally parallel-safe tool must not extend a batch containing unrelated work.
    name = "tool_describe"
    old = registry.get_entry(name)
    registry.register(name=name, toolset="deadline-probe", schema={"name": name},
                      handler=lambda args, **kw: "ok", execution_timeout=1000, override=True)
    try:
        from agent.tool_dispatch_helpers import _batch_admission

        for exposed, args in [(name, {}), ("tool_call", {"calls": [{"name": name, "arguments": {}}]})]:
            call = SimpleNamespace(function=SimpleNamespace(name=exposed, arguments=json.dumps(args)))
            assert _batch_admission(call, None) is None
    finally:
        current = registry.get_entry(name)
        assert current is not None
        registry.restore_registration(name, current, old)
