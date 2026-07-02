"""Tests for runtime executor integration with /v1/runs routes.

Covers:
- POST /v1/runs without executor remains control-plane-only
- POST /v1/runs with executor and execute=true enqueues execution
- Status and events reflect executor transitions when executed
- Stop/cancel works for executor-owned runs
- Approval/clarify works for executor-owned runs
- Backward compatibility
"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.runtime.routes import register_runtime_routes
from gateway.runtime.run_manager import RunManager
from gateway.runtime.control_bridge import RuntimeControlBridge
from gateway.runtime.executor import RuntimeExecutor, FakeAgentFactory


def _create_body(**overrides):
    base = {
        "session_id": "exec_test",
        "message": "test message",
        "workspace": "/tmp",
    }
    base.update(overrides)
    return base


class TestExecutorOptInRoutes:
    @pytest.mark.asyncio
    async def test_default_create_no_executor(self):
        """Without an executor, the route remains control-plane-only."""
        app = web.Application()
        register_runtime_routes(app)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=_create_body())
            data = await resp.json()
            assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_create_with_executor_and_execute_true(self):
        """With executor configured and execute=true, run gets executed."""
        mgr = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, agent_factory=factory)
        app = web.Application()
        register_runtime_routes(app, run_manager=mgr, executor=executor)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=_create_body(execute=True))
            data = await resp.json()
            assert data["status"] == "queued"
            run_id = data["run_id"]

            import asyncio
            await asyncio.sleep(0.1)

            status_resp = await cli.get(f"/v1/runs/{run_id}")
            status = await status_resp.json()
            assert status["status"] == "completed"
            assert status["terminal"] is True

    @pytest.mark.asyncio
    async def test_create_with_executor_no_execute_flag(self):
        """With executor configured but no execute flag, route stays CP-only."""
        mgr = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, agent_factory=factory)
        app = web.Application()
        register_runtime_routes(app, run_manager=mgr, executor=executor)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=_create_body())
            data = await resp.json()
            assert data["status"] == "queued"
            run_id = data["run_id"]

            import asyncio
            await asyncio.sleep(0.1)

            status_resp = await cli.get(f"/v1/runs/{run_id}")
            status = await status_resp.json()
            assert status["status"] == "queued"
            assert status["terminal"] is False

    @pytest.mark.asyncio
    async def test_executor_fail_transition_shows_in_status(self):
        mgr = RunManager()
        factory = FakeAgentFactory(fail=True)
        executor = RuntimeExecutor(mgr, agent_factory=factory)
        app = web.Application()
        register_runtime_routes(app, run_manager=mgr, executor=executor)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=_create_body(execute=True))
            data = await resp.json()
            run_id = data["run_id"]

            import asyncio
            await asyncio.sleep(0.1)

            status_resp = await cli.get(f"/v1/runs/{run_id}")
            status = await status_resp.json()
            assert status["status"] == "failed"
            assert status["terminal"] is True

    @pytest.mark.asyncio
    async def test_executor_events_visible_via_route(self):
        mgr = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(mgr, agent_factory=factory)
        app = web.Application()
        register_runtime_routes(app, run_manager=mgr, executor=executor)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=_create_body(execute=True))
            data = await resp.json()
            run_id = data["run_id"]

            import asyncio
            await asyncio.sleep(0.1)

            events_resp = await cli.get(f"/v1/runs/{run_id}/events")
            events = await events_resp.json()
            event_types = [e["type"] for e in events["events"]]
            assert "run.started" in event_types
            assert "run.status" in event_types
            assert "done" in event_types

    @pytest.mark.asyncio
    async def test_stop_works_on_executor_owned_run(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        factory = FakeAgentFactory(delay=0.5)
        executor = RuntimeExecutor(mgr, control_bridge=bridge, agent_factory=factory)
        app = web.Application()
        register_runtime_routes(app, run_manager=mgr, executor=executor, control_bridge=bridge)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=_create_body(execute=True))
            data = await resp.json()
            run_id = data["run_id"]

            import asyncio
            await asyncio.sleep(0.05)

            stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
            stop_data = await stop_resp.json()
            assert stop_data["status"] == "cancelled"

            await asyncio.sleep(0.2)

            status_resp = await cli.get(f"/v1/runs/{run_id}")
            status = await status_resp.json()
            assert status["status"] == "cancelled"
            assert status["terminal"] is True

    @pytest.mark.asyncio
    async def test_approval_via_route_on_executor_run(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        approval_triggered = []
        def _request_approval(run_id):
            mgr.request_approval(run_id, "apr-route-1", payload={"command": "ls"})
            approval_triggered.append(run_id)

        factory = FakeAgentFactory(request_approval=_request_approval)
        executor = RuntimeExecutor(mgr, control_bridge=bridge, agent_factory=factory)
        app = web.Application()
        register_runtime_routes(app, run_manager=mgr, executor=executor, control_bridge=bridge)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=_create_body(execute=True))
            data = await resp.json()
            run_id = data["run_id"]

            import asyncio
            await asyncio.sleep(0.05)

            assert len(approval_triggered) == 1

            status = mgr.get_status(run_id)
            if status["status"] == "awaiting_approval":
                apr_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"approval_id": "apr-route-1", "choice": "approve"},
                )
                apr_data = await apr_resp.json()
                assert apr_data.get("status") == "resolved"

    @pytest.mark.asyncio
    async def test_clarify_via_route_on_executor_run(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        clarify_triggered = []
        def _request_clarify(run_id):
            mgr.request_clarify(run_id, "clar-route-1", payload={"question": "OK?"})
            clarify_triggered.append(run_id)

        factory = FakeAgentFactory(request_clarify=_request_clarify)
        executor = RuntimeExecutor(mgr, control_bridge=bridge, agent_factory=factory)
        app = web.Application()
        register_runtime_routes(app, run_manager=mgr, executor=executor, control_bridge=bridge)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=_create_body(execute=True))
            data = await resp.json()
            run_id = data["run_id"]

            import asyncio
            await asyncio.sleep(0.05)

            assert len(clarify_triggered) == 1

            status = mgr.get_status(run_id)
            if status["status"] == "awaiting_clarify":
                clar_resp = await cli.post(
                    f"/v1/runs/{run_id}/clarify",
                    json={"clarify_id": "clar-route-1", "response": "yes"},
                )
                clar_data = await clar_resp.json()
                assert clar_data.get("status") == "resolved"
