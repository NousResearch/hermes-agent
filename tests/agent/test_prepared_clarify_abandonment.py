"""Prepared terminal middleware must share the sequential executor's prompt owner."""
import threading
from contextlib import suppress
from types import SimpleNamespace

import pytest

from tests.agent.test_sequential_tool_timeout import _make_agent
from tools import clarify_gateway as cm
from tools.clarify_tool import clarify_tool


@pytest.mark.parametrize("phase", ["preparation", "execution"])
def test_prepared_nested_clarify_is_settled_on_abandonment(tmp_path, monkeypatch, phase):
    from agent import terminal_approval_batch as tab
    from agent import tool_executor as te
    agent = _make_agent(tmp_path)
    entered, continued = threading.Event(), threading.Event()
    batches = []
    original_init = tab._TerminalBatch.__init__
    def init(batch, *args, **kwargs):
        original_init(batch, *args, **kwargs)
        batches.append(batch)
    monkeypatch.setattr(tab._TerminalBatch, "__init__", init)
    monkeypatch.setattr("gateway.session_context.get_session_env", lambda key, default=None: "desktop" if key == "HERMES_SESSION_SOURCE" else default)
    monkeypatch.setattr("tools.approval._gateway_notify_cb", lambda key: object())
    monkeypatch.setattr("tools.terminal_tool._get_env_config", lambda: {"env_type": "local"})
    monkeypatch.setattr("tools.terminal_tool._docker_has_host_access", lambda cfg: False)
    monkeypatch.setattr("tools.terminal_tool._check_all_guards", lambda *a: {"approved": True})
    monkeypatch.setattr(te, "_resolve_sequential_tool_timeout", lambda: 0.2)

    def callback(questions):
        cm.register("prepared-owned", "prepared-session", "Continue?", None)
        entered.set()
        answer = cm.wait_for_response("prepared-owned", timeout=0)
        return {"answers": {"q0": answer}, "outcome": "submitted"}
    def nested():
        result = clarify_tool([{"question": "Continue?"}], callback=callback)
        continued.set()
        return result
    def middleware(name, args, execute, **kwargs):
        if phase == "preparation":
            return nested()
        return execute(args)
    monkeypatch.setattr("hermes_cli.middleware.run_tool_execution_middleware", middleware)
    monkeypatch.setattr(te, "_resolve_sequential_dispatch", lambda *a: SimpleNamespace(execute=lambda args: nested()))
    def poll(*args, **kwargs):
        assert entered.wait(3)
        return "timeout", None
    monkeypatch.setattr(te, "_poll_sequential_future", poll)
    calls = [SimpleNamespace(id=f"prepared-{i}", function=SimpleNamespace(name="terminal", arguments='{"command":"unused"}')) for i in range(2)]
    try:
        with tab.terminal_approval_batch(agent, calls, [], "prepared-task"):
            if phase == "execution":
                result = te._run_sequential_tool_execution_middleware(
                    agent, function_name="terminal", function_args={"command": "unused"},
                    effective_task_id="prepared-task", tool_call_id="prepared-0",
                    execute=lambda args: pytest.fail("prepared future was not reused"),
                )
                assert "timed out" in result.result
            assert entered.is_set()
            assert not cm.has_pending("prepared-session")
            assert not cm.resolve_gateway_clarify("prepared-owned", "late answer")
        for slot in batches[0].slots:
            if slot.future is not None and not slot.future.cancelled():
                with suppress(tab._CancelledPreparation):
                    slot.future.result(timeout=3)
        assert not continued.is_set()
        assert not agent._tool_worker_threads
    finally:
        cm.clear_session("prepared-session")
        for batch in batches:
            batch.close()
            for slot in batch.slots:
                if slot.future is not None and not slot.future.cancelled():
                    with suppress(tab._CancelledPreparation):
                        slot.future.result(timeout=3)
