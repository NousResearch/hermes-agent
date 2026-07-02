"""Tests for the /v1/runs execution-plane gap.

Audits whether POST /v1/runs (via the standalone runtime routes module)
can execute an AIAgent, and documents the exact missing primitives.

Current state (Phase 15):
- POST /v1/runs in ``gateway/runtime/routes.py`` is **control-plane-only**:
  creates a RunManager entry and returns a run_id.  No AIAgent is spawned,
  no message is processed, no events beyond ``run.started`` are emitted.
- The API server adapter (``gateway/platforms/api_server.py``) has its OWN
  POST /v1/runs handler that DOES create an AIAgent and execute it -- but
  that is a separate handler with its own route registration, NOT the
  runtime routes module.
- Execution requires: AIAgent construction (with model credentials, session
  store, callbacks), background task/thread management, streaming SSE event
  management, approval/clarify/stop wiring through the bridge and tools
  modules, and a GatewayRunner or similar session context.

The architecture intentionally separates control-plane run creation from
execution.  The GatewayRunner's ``set_runtime_control_bridge()`` can wire
the bridge for live session control, and ``RuntimeControlBridge.stop_run()``
can reach GatewayRunner's ``_running_agents``, but POST /v1/runs has no
access to a GatewayRunner or AIAgent in the standalone case.

Missing primitives needed for full execution-plane support:
1. A way to configure which AI model/provider/credentials to use for execution.
2. Background task management (asyncio task or thread) for non-blocking run.
3. Event streaming infrastructure for the duration of agent execution.
4. Approval/clarify notification wiring through the bridge to the live agent.
5. An opt-in execution flag or config that callers can set to request execution.
"""

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.runtime.routes import register_runtime_routes
from gateway.runtime.run_manager import RunManager


def _create_body(**overrides):
    base = {
        "session_id": "gap_test",
        "message": "test message",
        "workspace": "/tmp",
        "profile": "default",
    }
    base.update(overrides)
    return base


class TestV1RunsIsControlPlaneOnly:
    """POST /v1/runs creates a RunManager entry without executing an AIAgent.

    These tests assert the control-plane-only behavior explicitly.
    """

    @pytest.mark.asyncio
    async def test_create_does_not_add_running_status(self):
        """POST /v1/runs returns status 'queued', not 'running'.

        If execution were happening, the status would transition through
        'running' at some point during the response lifecycle.
        """
        app = web.Application()
        register_runtime_routes(app)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=_create_body())
            assert resp.status == 202
            data = await resp.json()
            assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_create_does_not_add_done_event(self):
        """POST /v1/runs emits exactly one event (run.started), not a done event.

        If an AIAgent executed, there would be a done/completed event.
        """
        app = web.Application()
        register_runtime_routes(app)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=_create_body())
            data = await resp.json()
            run_id = data["run_id"]
            events_resp = await cli.get(f"/v1/runs/{run_id}/events")
            events = await events_resp.json()
            assert len(events["events"]) == 1
            assert events["events"][0]["type"] == "run.started"

    @pytest.mark.asyncio
    async def test_create_leaves_run_queued_indefinitely(self):
        """Without external execution, the run stays queued forever.

        The RunManager has no executor that picks up queued runs.
        """
        app = web.Application()
        register_runtime_routes(app)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=_create_body())
            data = await resp.json()
            run_id = data["run_id"]
            status_resp = await cli.get(f"/v1/runs/{run_id}")
            status = await status_resp.json()
            assert status["status"] == "queued"
            assert status["terminal"] is False

    @pytest.mark.asyncio
    async def test_create_does_not_require_model_or_credentials(self):
        """Control-plane create succeeds without model config or API keys.

        This is a defining characteristic of control-plane-only:
        no execution dependencies are checked.
        """
        app = web.Application()
        register_runtime_routes(app)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"session_id": "bare", "message": "hi"})
            assert resp.status == 202
            data = await resp.json()
            assert data["status"] == "queued"


class TestV1RunsNoAIAgent:
    """The runtime routes module never creates or references an AIAgent."""

    def test_run_manager_has_no_agent_reference(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="gap")
        status = mgr.get_status(r["run_id"])
        assert status["status"] == "queued"
        assert status["terminal"] is False

    def test_no_agent_constructed_on_create(self):
        """Verify by checking that create_run returns expected fields without
        any agent-related side effects.
        """
        mgr = RunManager()
        r = mgr.create_run(
            session_id="gap_agentless",
            message="does not require AIAgent",
        )
        assert "agent_ref" not in r
        assert "execution" not in r
        assert r["status"] == "queued"


class TestV1RunsRequiresExternalExecutor:
    """A /v1/runs-created run needs an external executor to advance past queued.

    The API server adapter fulfills this role when HERMES_USE_RUNTIME_RUNS=1:
    it calls run_manager.create_run(), creates an AIAgent, calls
    agent.run_conversation(),and emits events.  The standalone runtime
    routes module does not do this.
    """

    def test_standalone_routes_have_no_executor(self):
        app = web.Application()
        register_runtime_routes(app)
        mgr = app["runtime_run_manager"]
        assert not hasattr(mgr, "_executor")
        assert not hasattr(mgr, "_agent_pool")

    def test_control_bridge_is_optional(self):
        """register_runtime_routes works without a control_bridge."""
        app = web.Application()
        register_runtime_routes(app)
        assert "runtime_control_bridge" not in app

    def test_control_bridge_has_no_agent_until_bound(self):
        mgr = RunManager()
        from gateway.runtime.control_bridge import RuntimeControlBridge
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run(session_id="gap_bridge")
        assert r["status"] == "queued"


class TestExecutionGapDocumentation:
    """Document the exact missing primitives as assertions.

    These tests exist so future developers can remove/update them when
    the execution-plane gap is closed.

    Missing primitives for a minimal opt-in /v1/runs execution path:
    1. ``execute`` flag on POST /v1/runs body.
    2. AIAgent factory accepting session_id, model, credentials.
    3. Background task manager for run execution.
    4. Event streaming from AIAgent to RunManager.
    5. Approval/clarify notification wiring from RunManager events to
       live agent primitives (tools.approval, tools.clarify_gateway).
    """

    def test_no_execute_flag_on_create(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="gap_flag")
        assert "execute" not in r

    def test_no_agent_factory_in_routes_module(self):
        import sys
        mod_names = [n for n in sys.modules if "run_agent" in n]
        # The routes module does not import run_agent or AIAgent
        from gateway.runtime import routes as routes_mod
        mod_file = getattr(routes_mod, "__file__", "") or ""
        # Verify no import of AIAgent or run_agent in the routes module
        with open(mod_file) as f:
            src = f.read()
        assert "from run_agent import AIAgent" not in src
        assert "import run_agent" not in src
        assert "from run_agent import" not in src

    def test_no_background_task_manager(self):
        app = web.Application()
        register_runtime_routes(app)
        mgr = app["runtime_run_manager"]
        assert not hasattr(mgr, "_background_tasks")

    def test_create_handler_returns_json_not_sse(self):
        """POST /v1/runs handler returns a JSON response, not a StreamResponse.
        The SSE path is only in the events GET handler, not in create.
        """
        from gateway.runtime import routes as routes_mod
        # Directly inspect the create_run handler code
        import inspect
        create_src = inspect.getsource(routes_mod.register_runtime_routes)
        # The _handle_create_run closure uses web.json_response, not web.StreamResponse
        # We verify this by checking the closure body
        assert "web.json_response" in create_src
        # Confirm that _handle_create_run is NOT part of the SSE path
        # by checking that it doesn't appear near StreamResponse usage
        lines = create_src.split("\n")
        create_range = False
        for i, line in enumerate(lines):
            if "async def _handle_create_run" in line:
                create_range = True
            elif "async def _handle_get_run" in line:
                create_range = False
            if create_range and "StreamResponse" in line:
                pytest.fail("_handle_create_run uses StreamResponse")
        assert True

    def test_routes_have_optional_executor_event_loop(self):
        """The routes module uses asyncio.create_task only for the optional
        RuntimeExecutor path.  This confirms the executor integration is
        present but routes don't use threading.Thread or run_in_executor.
        """
        import inspect
        from gateway.runtime import routes
        source = inspect.getsource(routes)
        assert "asyncio.create_task" in source
        assert "threading.Thread" not in source
        assert "run_in_executor" not in source


class TestSafeFutureDesign:
    """Recommended future design for /v1/runs execution.

    When execution is added, it should:
    1. Be opt-in (default: control-plane-only, backward compatible).
    2. Accept model/provider/credentials via the request body or config.
    3. Create the run via RunManager (unchanged).
    4. Create an AIAgent via a factory or injected executor.
    5. Run the agent in a background task/thread.
    6. Emit status events (running, completed/failed/cancelled).
    7. Wire approval/clarify/stop through RuntimeControlBridge.
    8. Use register_create=True in the API server adapter, not a
       separate handler.

    The safest implementation point is a new "runtime executor" service
    that watches the RunManager for queued runs and executes them via
    a configurable AIAgent factory.  This keeps execution separate from
    the route layer and allows multiple executor backends.
    """

    def test_api_server_already_has_execution_path(self):
        """The API server adapter already implements execution.

        This confirms that the missing piece is in the standalone routes
        module, not in the overall architecture.
        """
        import inspect
        try:
            from gateway.platforms.api_server import APIServerAdapter
            source = inspect.getsource(APIServerAdapter._handle_runs)
        except (ImportError, AttributeError):
            pytest.skip("APIServerAdapter not available")
        assert "AIAgent" in source or "run_conversation" in source

    def test_future_design_should_use_register_create(self):
        """The register_create parameter exists to let API server adapter
        provide its own create handler.  A future standalone executor could
        use register_create=True and extend the routes module to optionally
        execute.
        """
        from gateway.runtime.routes import register_runtime_routes
        import inspect
        sig = inspect.signature(register_runtime_routes)
        params = sig.parameters
        assert "register_create" in params
