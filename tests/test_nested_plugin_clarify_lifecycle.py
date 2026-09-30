"""Real nested clarify waits retain the generic plugin deadline contract.

The platform wait is exercised through PluginContext/registry, not a mocked
handle_function_call. Only the executor's clock is advanced: no wall-clock
sleep or change to global time, and every worker is drained before teardown.
"""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from tests.agent.test_sequential_tool_timeout import _make_agent
from tools import clarify_gateway
from tools.registry import registry


@pytest.mark.parametrize("callback_source", ["framework", "legacy"])
@pytest.mark.parametrize("executor_kind", ["sequential", "concurrent"])
@pytest.mark.parametrize("outcome", ["submitted", "cancelled", "outer_timeout"])
def test_nested_gateway_wait_lifecycle(tmp_path, monkeypatch, executor_kind, outcome, callback_source):
    import agent.tool_executor as executor
    import hermes_cli.lifecycle as lifecycle
    import tools.clarify_tool  # noqa: F401 -- register the real clarify handler
    import tools.daemon_pool as daemon_pool
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    agent = _make_agent(tmp_path)
    name = "nested_wait_probe"
    session = "nested-wait-test"
    clarify_id = "nested-wait-question"
    entered = threading.Event()
    futures = []
    terminal_events = []
    continued = threading.Event()
    clock = SimpleNamespace(value=0.0)
    # A module-local clock leaves Event.wait, Future.result, and other modules'
    # clocks real; only the production executor deadline sees virtual elapsed time.
    monkeypatch.setattr(executor, "time", SimpleNamespace(
        monotonic=lambda: clock.value, time=time.time,
    ))
    monkeypatch.setattr(executor, "_resolve_sequential_tool_timeout", lambda: 420.0)
    monkeypatch.setattr(executor, "_resolve_concurrent_tool_timeout", lambda: 420.0)
    monkeypatch.setattr(lifecycle, "has_hook", lambda _name: True)
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda hook, **kw: (
        terminal_events.append(kw) if hook == "post_tool_call" else []
    ))

    real_pool = daemon_pool.DaemonThreadPoolExecutor

    class TrackedPool(real_pool):
        def submit(self, fn, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            futures.append(future)
            return future

    monkeypatch.setattr(daemon_pool, "DaemonThreadPoolExecutor", TrackedPool)

    def callback(questions):
        entry = clarify_gateway.register(
            clarify_id, session, questions[0]["question"], None,
        )
        real_wait = entry.event.wait

        def observed_wait(timeout=None):
            entered.set()
            return real_wait(timeout)

        entry.event.wait = observed_wait
        response = clarify_gateway.wait_for_response(clarify_id, timeout=0)
        if response == clarify_gateway.CANCELLED:
            return {"answers": {}, "outcome": "cancelled"}
        return {"answers": {"q0": response}, "outcome": "submitted"}

    agent.clarify_callback = callback if callback_source == "framework" else None
    manager = PluginManager()
    manager._cli_ref = None
    ctx = PluginContext(PluginManifest(name="nested-wait-plugin", source="user"), manager)

    def handler(args, **kwargs):
        if callback_source == "framework":
            assert kwargs["clarify_callback"] is callback
        else:
            # Existing direct-registry callers could already supply the legacy
            # framework callback; this control runs unchanged on pristine base.
            kwargs["callback"] = callback
        result = ctx.dispatch_tool("clarify", {
            "questions": [{"question": "Continue?"}], **args,
        }, **kwargs)
        continued.set()
        return result

    registry.register(
        name=name, toolset="nestedwaitprobe",
        schema={"name": name, "description": name,
                "parameters": {"type": "object", "properties": {}}},
        handler=handler,
    )
    agent.valid_tool_names = {name}

    def settle_wait():
        assert entered.wait(5), "real nested gateway wait was never reached"
        assert not futures[0].done()
        assert clarify_gateway.has_pending(session)
        if outcome == "outer_timeout":
            clock.value = 421.0
        else:
            clock.value = 100.0  # response genuinely follows an outstanding wait
            if outcome == "submitted":
                assert clarify_gateway.resolve_gateway_clarify(clarify_id, "YES")
            else:
                assert clarify_gateway.clear_session(session) == 1
            futures[0].result(timeout=5)

    # Synchronize at the actual executor wait boundary, then run its original
    # polling/abandonment implementation with its original future and deadline.
    if executor_kind == "sequential":
        original_poll = executor._poll_sequential_future

        def poll(*args, **kwargs):
            settle_wait()
            return original_poll(*args, **kwargs)

        monkeypatch.setattr(executor, "_poll_sequential_future", poll)
    else:
        original_await = executor._ConcurrentBatch.await_completion

        def await_completion(*args, **kwargs):
            settle_wait()
            return original_await(*args, **kwargs)

        monkeypatch.setattr(executor._ConcurrentBatch, "await_completion", await_completion)

    call = SimpleNamespace(id="nested-1", type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps({"callback": "EVIL", "clarify_callback": "EVIL"}),
    ))
    messages = []
    try:
        run = (executor.execute_tool_calls_sequential if executor_kind == "sequential"
               else executor.execute_tool_calls_concurrent)
        run(agent, SimpleNamespace(tool_calls=[call]), messages, "nested-task")
        assert len(messages) == 1
        assert messages[0]["tool_call_id"] == "nested-1"
        if outcome == "outer_timeout":
            assert "timed out after 420.0s" in messages[0]["content"]
            assert messages[0]["effect_disposition"] == "unknown"
            # The executor settles only this call's owned prompt before returning.
            assert not clarify_gateway.has_pending(session)
            assert not clarify_gateway.resolve_gateway_clarify(clarify_id, "late answer")
        else:
            payload = json.loads(messages[0]["content"])
            assert payload["outcome"] == outcome
            assert payload["responses"][0]["user_response"] == (
                "YES" if outcome == "submitted" else None
            )
        for future in futures:
            future.result(timeout=5)
        assert not clarify_gateway.has_pending(session)
        assert not clarify_gateway.resolve_gateway_clarify(clarify_id, "late answer")
        assert continued.is_set() == (outcome != "outer_timeout")
        assert not agent._tool_worker_threads
        # Draining a late worker cannot replace the committed result or publish
        # a second terminal event for the abandoned parent call.
        events = [e for e in terminal_events if e.get("tool_call_id") == "nested-1"]
        assert len(events) == 1
        assert events[0].get("error_type") == (
            "tool_timeout" if outcome == "outer_timeout" else None
        )
        assert len(messages) == 1
    finally:
        clarify_gateway.clear_session(session)
        for future in futures:
            future.result(timeout=5)
        registry.deregister(name)
