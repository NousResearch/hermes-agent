"""An abandoned concurrent worker cannot publish another terminal outcome."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from tests.agent.test_sequential_tool_timeout import _make_agent
from tools.registry import registry


@pytest.mark.parametrize("pause_at", ["handler", "after_dispatch"])
def test_late_worker_has_one_committed_terminal_event(tmp_path, monkeypatch, pause_at):
    import agent.tool_executor as executor
    import hermes_cli.lifecycle as lifecycle
    import tools.daemon_pool as daemon_pool

    agent = _make_agent(tmp_path)
    entered, release = threading.Event(), threading.Event()
    events, futures = [], []
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(executor, "time", SimpleNamespace(monotonic=lambda: clock.value, time=time.time))
    monkeypatch.setattr(executor, "_resolve_concurrent_tool_timeout", lambda: 420.0)
    monkeypatch.setattr(lifecycle, "has_hook", lambda _name: True)
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda hook, **kw: (
        events.append(kw) if hook == "post_tool_call" else []
    ))
    real_pool = daemon_pool.DaemonThreadPoolExecutor

    class TrackedPool(real_pool):
        def submit(self, fn, *args, **kwargs):
            f = super().submit(fn, *args, **kwargs)
            futures.append(f)
            return f

    monkeypatch.setattr(daemon_pool, "DaemonThreadPoolExecutor", TrackedPool)

    def pause():
        entered.set()
        assert release.wait(5)

    def handler(args, **kwargs):
        if pause_at == "handler":
            pause()
        return json.dumps({"result": "late"})

    name = "late_terminal_probe"
    registry.register(name=name, toolset=name, schema={"name": name, "parameters": {"type": "object"}}, handler=handler)
    agent.valid_tool_names = {name}
    original_dispatch = executor._ConcurrentBatch._dispatch_worker

    def dispatch(*args, **kwargs):
        result = original_dispatch(*args, **kwargs)
        if pause_at == "after_dispatch":
            pause()
        return result

    monkeypatch.setattr(executor._ConcurrentBatch, "_dispatch_worker", dispatch)
    original_await = executor._ConcurrentBatch.await_completion

    def await_completion(*args, **kwargs):
        assert entered.wait(5)
        clock.value = 421.0
        return original_await(*args, **kwargs)

    monkeypatch.setattr(executor._ConcurrentBatch, "await_completion", await_completion)
    call = SimpleNamespace(id="late-1", function=SimpleNamespace(name=name, arguments="{}"))
    messages = []
    try:
        executor.execute_tool_calls_concurrent(agent, SimpleNamespace(tool_calls=[call]), messages, "task")
        assert len(messages) == 1
        assert "timed out after 420.0s" in messages[0]["content"]
        release.set()
        for future in futures:
            future.result(timeout=5)
        own = [e for e in events if e.get("tool_call_id") == "late-1"]
        assert len(own) == 1
        # Preserve already-fired post-before-transform semantics. If dispatch has
        # finished, its success event cannot be retracted when outer work stalls.
        assert own[0]["error_type"] == ("tool_timeout" if pause_at == "handler" else None)
        assert not agent._tool_worker_threads
    finally:
        release.set()
        for future in futures:
            future.result(timeout=5)
        registry.deregister(name)


@pytest.mark.parametrize("inline", [False, True])
def test_real_concurrent_executor_preserves_post_before_transform(tmp_path, monkeypatch, inline):
    import agent.tool_executor as executor
    import hermes_cli.lifecycle as lifecycle
    agent = _make_agent(tmp_path)
    order = []
    def hook(name, **kwargs):
        if name == "post_tool_call":
            order.append("post")
        if name == "transform_tool_result":
            order.append("transform")
            return ["after-post" if order == ["post", "transform"] else "WRONG ORDER"]
        return []
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: True)
    monkeypatch.setattr(lifecycle, "invoke_hook", hook)
    name = "todo_list" if inline else "terminal_order_probe"
    if inline:
        monkeypatch.setattr("tools.todo_tool.todo_tool", lambda **kwargs: "original")
    else:
        registry.register(name=name, toolset=name, schema={"name": name, "parameters": {"type": "object"}},
                          handler=lambda args, **kwargs: "original")
    agent.valid_tool_names = {name}
    call = SimpleNamespace(id="order-1", function=SimpleNamespace(name=name, arguments="{}"))
    messages = []
    try:
        executor.execute_tool_calls_concurrent(agent, SimpleNamespace(tool_calls=[call]), messages, "task")
        assert order == ["post", "transform"]
        assert messages[0]["content"] == "after-post"
    finally:
        if not inline:
            registry.deregister(name)
