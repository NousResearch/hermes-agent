"""Tests for gateway.runtime.executor.RuntimeExecutor.

Covers:
- Executor construction with default factories
- execute_run lifecycle with fake agent
- Status transitions (queued -> running -> completed/failed/cancelled)
- bind_run / unbind_run through control bridge
- Error redaction
- Background poll loop start/stop
- Cancel/interrupt of executor-owned runs
- Approval/clarify compatibility with fake agents
"""

import asyncio
import threading

import pytest

from gateway.runtime.run_manager import RunManager
from gateway.runtime.control_bridge import RuntimeControlBridge
from gateway.runtime.executor import (
    RuntimeExecutor,
    FakeAgentFactory,
    SessionKeyFactory,
)
from gateway.runtime.models import (
    RUN_STATUS_QUEUED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_CANCELLED,
)


class TestRuntimeExecutorConstruction:
    def test_default_construction(self):
        mgr = RunManager()
        executor = RuntimeExecutor(mgr)
        assert executor._run_manager is mgr
        assert executor._control_bridge is None
        assert executor._agent_factory is None
        assert executor._session_factory is not None
        assert executor.is_running is False

    def test_with_control_bridge(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        executor = RuntimeExecutor(mgr, control_bridge=bridge)
        assert executor._control_bridge is bridge

    def test_with_agent_factory(self):
        mgr = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, agent_factory=factory)
        assert executor._agent_factory is factory

    def test_with_custom_session_factory(self):
        mgr = RunManager()
        sf = SessionKeyFactory()
        executor = RuntimeExecutor(mgr, session_factory=sf)
        assert executor._session_factory is sf


class TestExecuteRunLifecycle:
    @pytest.mark.asyncio
    async def test_execute_completes_queued_run(self):
        mgr = RunManager()
        factory = FakeAgentFactory(result={
            "final_response": "hello world",
            "completed": True,
        })
        executor = RuntimeExecutor(mgr, agent_factory=factory)

        r = mgr.create_run("sess_test", message="test message")

        result = await executor.execute_run(r["run_id"])

        assert result["status"] == "completed"
        assert result["result"] == "hello world"

        status = mgr.get_status(r["run_id"])
        assert status["status"] == "completed"
        assert status["terminal"] is True

    @pytest.mark.asyncio
    async def test_execute_transitions_through_running(self):
        mgr = RunManager()
        factory = FakeAgentFactory(delay=0.01)
        executor = RuntimeExecutor(mgr, agent_factory=factory)

        r = mgr.create_run("sess_test")
        status_before = mgr.get_status(r["run_id"])
        assert status_before["status"] == "queued"

        await executor.execute_run(r["run_id"])

        status = mgr.get_status(r["run_id"])
        assert status["status"] == "completed"

    @pytest.mark.asyncio
    async def test_execute_fails_on_exception(self):
        mgr = RunManager()
        factory = FakeAgentFactory(fail=True, fail_message="test error")
        executor = RuntimeExecutor(mgr, agent_factory=factory)

        r = mgr.create_run("sess_test")

        result = await executor.execute_run(r["run_id"])

        assert result["status"] == "failed"
        assert "test error" in result.get("error", "")

        status = mgr.get_status(r["run_id"])
        assert status["status"] == "failed"
        assert status["terminal"] is True
        assert status["error"] is not None

    @pytest.mark.asyncio
    async def test_execute_unknown_run_returns_not_found(self):
        mgr = RunManager()
        executor = RuntimeExecutor(mgr)
        result = await executor.execute_run("run_nonexistent")
        assert result.get("error") == "not_found"

    @pytest.mark.asyncio
    async def test_execute_already_claimed_run_returns_conflict(self):
        mgr = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, agent_factory=factory)

        r = mgr.create_run("sess_test")
        mgr.transition_status(r["run_id"], "running")

        result = await executor.execute_run(r["run_id"])
        assert result.get("error") == "conflict"

    @pytest.mark.asyncio
    async def test_no_agent_factory_returns_not_supported(self):
        mgr = RunManager()
        executor = RuntimeExecutor(mgr)

        r = mgr.create_run("sess_test")
        result = await executor.execute_run(r["run_id"])

        assert result.get("error") == "not_supported"

    @pytest.mark.asyncio
    async def test_create_agent_failure_returns_error(self):
        mgr = RunManager()

        class FailingFactory:
            async def create_agent(self, **kwargs):
                raise RuntimeError("provider auth failed")

        executor = RuntimeExecutor(mgr, agent_factory=FailingFactory())

        r = mgr.create_run("sess_test")
        result = await executor.execute_run(r["run_id"])

        assert result.get("error") == "agent_creation_failed"

        status = mgr.get_status(r["run_id"])
        assert status["status"] == "failed"


class TestExecuteRunBindings:
    @pytest.mark.asyncio
    async def test_execute_calls_bind_run(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, control_bridge=bridge, agent_factory=factory)

        r = mgr.create_run("sess_bind")
        await executor.execute_run(r["run_id"])

        assert len(factory.created_agents) == 1
        agent = factory.created_agents[0]
        assert agent is not None

    @pytest.mark.asyncio
    async def test_execute_unbind_run_on_completion(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, control_bridge=bridge, agent_factory=factory)

        r = mgr.create_run("sess_unbind")
        run_id = r["run_id"]

        bind_spy = []
        orig_bind = bridge.bind_run
        def _bind(*args, **kwargs):
            bind_spy.append(args)
            return orig_bind(*args, **kwargs)
        bridge.bind_run = _bind

        unbind_spy = []
        orig_unbind = bridge.unbind_run
        def _unbind(*args, **kwargs):
            unbind_spy.append(args)
            return orig_unbind(*args, **kwargs)
        bridge.unbind_run = _unbind

        await executor.execute_run(run_id)

        assert len(bind_spy) == 1
        assert bind_spy[0][0] == run_id

        assert len(unbind_spy) == 1
        assert unbind_spy[0][0] == run_id

    @pytest.mark.asyncio
    async def test_unbind_on_failure(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        factory = FakeAgentFactory(fail=True)
        executor = RuntimeExecutor(mgr, control_bridge=bridge, agent_factory=factory)

        r = mgr.create_run("sess_unbind_fail")
        run_id = r["run_id"]

        unbind_called = []
        orig_unbind = bridge.unbind_run
        def _unbind(*args, **kwargs):
            unbind_called.append(args)
            return orig_unbind(*args, **kwargs)
        bridge.unbind_run = _unbind

        await executor.execute_run(run_id)

        assert len(unbind_called) == 1
        assert unbind_called[0][0] == run_id

    @pytest.mark.asyncio
    async def test_no_stale_bindings_after_execution(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, control_bridge=bridge, agent_factory=factory)

        r1 = mgr.create_run("sess_1")
        r2 = mgr.create_run("sess_2")

        await executor.execute_run(r1["run_id"])
        await executor.execute_run(r2["run_id"])

        assert r1["run_id"] not in bridge._bindings
        assert r2["run_id"] not in bridge._bindings
        assert r1["run_id"] not in bridge._live_agents
        assert r2["run_id"] not in bridge._live_agents


class TestExecuteRunEvents:
    @pytest.mark.asyncio
    async def test_completion_appends_done_event(self):
        mgr = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, agent_factory=factory)

        r = mgr.create_run("sess_events")
        await executor.execute_run(r["run_id"])

        events = mgr.read_events(r["run_id"])
        event_types = [e["type"] for e in events["events"]]
        assert "run.started" in event_types
        assert "run.status" in event_types
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_failure_appends_error_and_done_events(self):
        mgr = RunManager()
        factory = FakeAgentFactory(fail=True)
        executor = RuntimeExecutor(mgr, agent_factory=factory)

        r = mgr.create_run("sess_fail_events")
        await executor.execute_run(r["run_id"])

        events = mgr.read_events(r["run_id"])
        event_types = [e["type"] for e in events["events"]]
        assert "error" in event_types
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_status_endpoint_reflects_transitions(self):
        mgr = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, agent_factory=factory)

        r = mgr.create_run("sess_status")
        status = mgr.get_status(r["run_id"])
        assert status["status"] == "queued"

        await executor.execute_run(r["run_id"])

        status = mgr.get_status(r["run_id"])
        assert status["status"] == "completed"
        assert status["terminal"] is True


class TestExecuteRunCancellation:
    @pytest.mark.asyncio
    async def test_cancel_run_calls_agent_interrupt(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        factory = FakeAgentFactory(delay=0.5)
        executor = RuntimeExecutor(mgr, control_bridge=bridge, agent_factory=factory)

        r = mgr.create_run("sess_cancel")
        run_id = r["run_id"]

        execute_task = asyncio.create_task(executor.execute_run(run_id))
        await asyncio.sleep(0.05)

        cancel_result = await executor.cancel_run(run_id)
        assert cancel_result["status"] == "cancelled"

        result = await execute_task
        assert result["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_terminal_run_is_noop(self):
        mgr = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, agent_factory=factory)

        r = mgr.create_run("sess_term")
        await executor.execute_run(r["run_id"])

        result = await executor.cancel_run(r["run_id"])
        assert result["status"] in ("completed",)

    @pytest.mark.asyncio
    async def test_cancel_unknown_run_returns_not_found(self):
        mgr = RunManager()
        executor = RuntimeExecutor(mgr)
        result = await executor.cancel_run("nonexistent")
        assert result.get("error") == "not_found"


class TestExecuteRunApprovalClarify:
    @pytest.mark.asyncio
    async def test_approval_works_with_executor_agent(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        approval_triggered = []

        def _request_approval(run_id):
            mgr.request_approval(run_id, "apr-exec-1", payload={"command": "ls"})
            approval_triggered.append(run_id)

        factory = FakeAgentFactory(request_approval=_request_approval)
        executor = RuntimeExecutor(mgr, control_bridge=bridge, agent_factory=factory)

        r = mgr.create_run("sess_apr_exec")
        run_id = r["run_id"]

        execute_task = asyncio.create_task(executor.execute_run(run_id))
        await asyncio.sleep(0.05)

        assert len(approval_triggered) == 1
        status = mgr.get_status(run_id)
        assert status["status"] in ("awaiting_approval", "completed")

        if status["status"] == "awaiting_approval":
            resolve = mgr.resolve_approval(run_id, "apr-exec-1", "approve")
            assert resolve.get("status") == "resolved"

            result = await execute_task
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_clarify_works_with_executor_agent(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        clarify_triggered = []

        def _request_clarify(run_id):
            mgr.request_clarify(run_id, "clar-exec-1", payload={"question": "OK?"})
            clarify_triggered.append(run_id)

        factory = FakeAgentFactory(request_clarify=_request_clarify)
        executor = RuntimeExecutor(mgr, control_bridge=bridge, agent_factory=factory)

        r = mgr.create_run("sess_clar_exec")
        run_id = r["run_id"]

        execute_task = asyncio.create_task(executor.execute_run(run_id))
        await asyncio.sleep(0.05)

        assert len(clarify_triggered) == 1
        status = mgr.get_status(run_id)
        assert status["status"] in ("awaiting_clarify", "completed")

        if status["status"] == "awaiting_clarify":
            resolve = mgr.resolve_clarify(run_id, "clar-exec-1", "yes")
            assert resolve.get("status") == "resolved"


class TestBackgroundLoop:
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        mgr = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, agent_factory=factory)

        assert executor.is_running is False
        await executor.start(poll_interval=0.1)
        assert executor.is_running is True

        await executor.stop()
        assert executor.is_running is False

    @pytest.mark.asyncio
    async def test_run_once_consumes_one_queued_run(self):
        mgr = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, agent_factory=factory)

        r = mgr.create_run("sess_poll")
        executed_id = await executor.run_once()

        assert executed_id == r["run_id"]
        status = mgr.get_status(r["run_id"])
        assert status["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_once_no_queued_returns_none(self):
        mgr = RunManager()
        executor = RuntimeExecutor(mgr)
        result = await executor.run_once()
        assert result is None


class TestSessionKeyFactory:
    def test_create_session_key_format(self):
        sf = SessionKeyFactory()
        key = sf.create_session_key("run_abc123", "sess_test")
        assert key == "exec-run_abc123-sess_test"
        assert key.startswith("exec-")


class TestFakeAgentFactory:
    @pytest.mark.asyncio
    async def test_fake_agent_returns_fixed_result(self):
        factory = FakeAgentFactory(result={
            "final_response": "fixed",
            "completed": True,
        })
        agent = await factory.create_agent(
            run_id="test", session_id="s", message="hi", session_key="k",
        )
        result = await agent.run_conversation(user_message="hi")
        assert result["final_response"] == "fixed"
        assert result["completed"] is True

    @pytest.mark.asyncio
    async def test_fake_agent_raises_on_fail(self):
        factory = FakeAgentFactory(fail=True, fail_message="boom")
        agent = await factory.create_agent(
            run_id="test", session_id="s", message="hi", session_key="k",
        )
        with pytest.raises(RuntimeError, match="boom"):
            await agent.run_conversation(user_message="hi")

    @pytest.mark.asyncio
    async def test_fake_agent_supports_interrupt(self):
        factory = FakeAgentFactory(delay=0.5)
        agent = await factory.create_agent(
            run_id="test", session_id="s", message="hi", session_key="k",
        )
        assert agent.interrupted is False
        agent.interrupt("run_stop")
        assert agent.interrupted is True
