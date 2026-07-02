"""Tests for gateway runtime live agent binding.

Covers:
- bind_run(run_id, session_key, agent) stores agent reference
- unbind_run(run_id) cleans up both mappings
- stop_run uses direct agent reference when available via bind_run
- Approval resolution uses bound session_key
- Clarify resolution works with globally unique clarify_id
- Binding cleanup on terminal run states
- Missing binding returns not_supported/fallback behavior
- Unknown run/action maps to 404
- Conflict maps to 409
- URL path run_id wins over body run_id
- Events are appended exactly once
- Pending IDs are added and removed correctly
- Secrets are redacted
- register_runtime_routes respects register_create/register_status/register_events flags
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.runtime.run_manager import RunManager
from gateway.runtime.control_bridge import RuntimeControlBridge
from gateway.runtime.routes import register_runtime_routes
from gateway.runtime.models import (
    EVENT_APPROVAL_REQUESTED,
    EVENT_APPROVAL_RESOLVED,
    EVENT_CLARIFY_REQUESTED,
    EVENT_CLARIFY_RESOLVED,
)


def _create_app():
    app = web.Application()
    register_runtime_routes(app)
    return app


def _create_app_control_only():
    app = web.Application()
    register_runtime_routes(
        app,
        register_create=False,
        register_status=False,
        register_events=False,
    )
    return app


class TestBindRunAgentRef:
    """bind_run stores agent reference for live interrupt."""

    def test_bind_run_stores_agent(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        mock_agent = MagicMock()
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "gw_key", mock_agent)
        assert bridge._live_agents.get(r["run_id"]) is mock_agent

    def test_bind_run_agent_is_optional(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "gw_key")
        assert r["run_id"] in bridge._bindings
        assert r["run_id"] not in bridge._live_agents

    def test_unbind_run_cleans_both_mappings(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        mock_agent = MagicMock()
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "gw_key", mock_agent)
        bridge.unbind_run(r["run_id"])
        assert r["run_id"] not in bridge._bindings
        assert r["run_id"] not in bridge._live_agents

    def test_unbind_run_nonexistent_is_noop(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        bridge.unbind_run("nonexistent")

    def test_stop_run_uses_direct_agent_reference(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "gw_key", mock_agent)
        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"
        mock_agent.interrupt.assert_called_once_with("run_stop")

    def test_stop_run_direct_agent_failure_does_not_raise(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        mock_agent.interrupt.side_effect = RuntimeError("interrupt failed")
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "gw_key", mock_agent)
        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"

    def test_stop_run_pure_runmanager_fallback_when_no_binding(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"


class TestApprovalResolutionWithBinding:
    """Bridge resolves approvals using bound session_key."""

    def test_approval_resolve_with_bound_session_key(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "sk_bound")
        mgr.request_approval(r["run_id"], "apr-001")
        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            mock_resolve.return_value = 1
            result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
            assert result["status"] == "resolved"
            mock_resolve.assert_called_once_with("sk_bound", "once")

    def test_approval_resolve_without_binding_still_succeeds(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
            assert result["status"] == "resolved"
            mock_resolve.assert_not_called()

    def test_approval_resolve_after_unbind_no_live_primitive(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "sk_temp")
        mgr.request_approval(r["run_id"], "apr-001")
        bridge.unbind_run(r["run_id"])
        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
            assert result["status"] == "resolved"
            mock_resolve.assert_not_called()

    def test_approval_not_found_on_unknown_run(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        result = bridge.resolve_approval("nonexistent", "apr-001", "once")
        assert result["error"] == "not_found"

    def test_approval_conflict_on_duplicate_resolve(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        bridge.resolve_approval(r["run_id"], "apr-001", "once")
        result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
        assert result["error"] == "conflict"


class TestClarifyResolutionWithBinding:
    """Bridge clarify resolution works with globally unique clarify_id."""

    def test_clarify_resolve_calls_live_primitive(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            mock_resolve.return_value = True
            result = bridge.resolve_clarify(r["run_id"], "clar-001", "yes")
            assert result["status"] == "resolved"
            mock_resolve.assert_called_once_with("clar-001", "yes")

    def test_clarify_not_found_on_unknown_run(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        result = bridge.resolve_clarify("nonexistent", "clar-001", "yes")
        assert result["error"] == "not_found"

    def test_clarify_conflict_on_duplicate_resolve(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        bridge.resolve_clarify(r["run_id"], "clar-001", "yes")
        result = bridge.resolve_clarify(r["run_id"], "clar-001", "yes")
        assert result["error"] == "conflict"


class TestBindingLifecycle:
    """bind/unbind lifetime in a realistic run lifecycle."""

    def test_full_lifecycle_create_run_approve_stop_cleanup(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "sk_life", mock_agent)

        assert r["run_id"] in bridge._bindings
        assert bridge._live_agents.get(r["run_id"]) is mock_agent

        mgr.request_approval(r["run_id"], "apr-life")
        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            mock_resolve.return_value = 1
            result = bridge.resolve_approval(r["run_id"], "apr-life", "once")
            assert result["status"] == "resolved"
            mock_resolve.assert_called_once_with("sk_life", "once")

        bridge.unbind_run(r["run_id"])
        assert r["run_id"] not in bridge._bindings
        assert r["run_id"] not in bridge._live_agents

    def test_unbind_then_stop_uses_runmanager_only(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "sk_life", mock_agent)
        bridge.unbind_run(r["run_id"])

        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"
        mock_agent.interrupt.assert_not_called()

    def test_callable_session_key_used_after_unbind(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        mock_runner = MagicMock()
        mock_runner._running_agents = {"callable_key": mock_agent}

        bridge = RuntimeControlBridge(
            mgr,
            get_session_key_for_run=lambda rid: "callable_key",
            gateway_runner_ref=lambda: mock_runner,
        )
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "direct_key")
        bridge.unbind_run(r["run_id"])
        mgr.request_approval(r["run_id"], "apr-001")
        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            mock_resolve.return_value = 1
            result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
            assert result["status"] == "resolved"
            mock_resolve.assert_called_once_with("callable_key", "once")


class TestSecretsAreRedacted:
    """Secrets stay redacted through bridge resolution."""

    def test_approval_payload_secrets_redacted(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-sec", payload={
            "command": "run safely",
            "api_key": "sk-abc123",
            "token": "tk-secret",
            "authorization": "Bearer secret",
            "access_token": "at-xyz",
            "secret": "s3cret",
            "password": "passw0rd",
            "credential": "cred-99",
        })
        result = bridge.resolve_approval(r["run_id"], "apr-sec", "once")
        assert result["status"] == "resolved"
        event = result.get("event", {})
        inner = event.get("payload", {}).get("payload", {})
        for secret_key in ("api_key", "token", "authorization", "access_token",
                           "secret", "password", "credential"):
            val = inner.get(secret_key, "")
            assert val != f"sk-abc123", f"{secret_key} leaked raw value"
            assert val != "tk-secret"
            assert isinstance(val, str)

    def test_clarify_payload_secrets_redacted(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-sec", payload={
            "question": "What next?",
            "api_key": "sk-xyz-789",
            "password": "hunter2",
        })
        result = bridge.resolve_clarify(r["run_id"], "clar-sec", "done")
        assert result["status"] == "resolved"
        event = result.get("event", {})
        inner = event.get("payload", {}).get("payload", {})
        assert inner.get("api_key") != "sk-xyz-789"
        assert inner.get("password") != "hunter2"


class TestRunIdBindingHttpIntegration:
    """HTTP-level binding tests: URL path run_id wins, error codes, etc."""

    async def _post_approval(self, client, run_id, approval_id, choice,
                             path_run_id=None, body_run_id=None):
        if path_run_id is None:
            path_run_id = run_id
        body = {"approval_id": approval_id, "choice": choice}
        if body_run_id is not None:
            body["run_id"] = body_run_id
        return await client.post(
            f"/v1/runs/{path_run_id}/approval",
            json=body,
        )

    @pytest.mark.asyncio
    async def test_url_path_run_id_wins_over_body_run_id(self):
        app = web.Application()
        run_manager = register_runtime_routes(app)
        bridge = RuntimeControlBridge(run_manager)
        app["runtime_control_bridge"] = bridge

        r = run_manager.create_run("sess_1")
        bridge.bind_run(r["run_id"], "sk_test")
        run_manager.request_approval(r["run_id"], "apr-001")

        async with TestServer(app) as server:
            async with TestClient(server) as client:
                resp = await self._post_approval(
                    client, r["run_id"], "apr-001", "once",
                    path_run_id=r["run_id"],
                    body_run_id="different_run_id",
                )
                assert resp.status == 400
                data = await resp.json()
                assert "run_id mismatch" in data.get("error", {}).get("message", "")

    @pytest.mark.asyncio
    async def test_approval_not_found_returns_404(self):
        app = web.Application()
        run_manager = register_runtime_routes(app)

        r = run_manager.create_run("sess_1")

        async with TestServer(app) as server:
            async with TestClient(server) as client:
                resp = await client.post(
                    f"/v1/runs/{r['run_id']}/approval",
                    json={"approval_id": "nonexistent", "choice": "once"},
                )
                assert resp.status == 404

    @pytest.mark.asyncio
    async def test_clarify_not_found_returns_404(self):
        app = web.Application()
        run_manager = register_runtime_routes(app)

        r = run_manager.create_run("sess_1")

        async with TestServer(app) as server:
            async with TestClient(server) as client:
                resp = await client.post(
                    f"/v1/runs/{r['run_id']}/clarify",
                    json={"clarify_id": "nonexistent", "response": "answer"},
                )
                assert resp.status == 404

    @pytest.mark.asyncio
    async def test_approval_duplicate_returns_409(self):
        app = web.Application()
        run_manager = register_runtime_routes(app)
        bridge = RuntimeControlBridge(run_manager)
        app["runtime_control_bridge"] = bridge

        r = run_manager.create_run("sess_1")
        run_manager.request_approval(r["run_id"], "apr-001")

        async with TestServer(app) as server:
            async with TestClient(server) as client:
                body = {"approval_id": "apr-001", "choice": "once"}
                resp1 = await client.post(
                    f"/v1/runs/{r['run_id']}/approval", json=body,
                )
                assert resp1.status == 200
                resp2 = await client.post(
                    f"/v1/runs/{r['run_id']}/approval", json=body,
                )
                assert resp2.status == 409

    @pytest.mark.asyncio
    async def test_approval_terminal_run_returns_409(self):
        app = web.Application()
        run_manager = register_runtime_routes(app)
        bridge = RuntimeControlBridge(run_manager)
        app["runtime_control_bridge"] = bridge

        r = run_manager.create_run("sess_1")
        run_manager.request_approval(r["run_id"], "apr-001")
        run_manager.stop_run(r["run_id"])

        async with TestServer(app) as server:
            async with TestClient(server) as client:
                resp = await client.post(
                    f"/v1/runs/{r['run_id']}/approval",
                    json={"approval_id": "apr-001", "choice": "once"},
                )
                assert resp.status == 409

    @pytest.mark.asyncio
    async def test_stop_nonexistent_returns_404(self):
        app = web.Application()
        register_runtime_routes(app)

        async with TestServer(app) as server:
            async with TestClient(server) as client:
                resp = await client.post("/v1/runs/nonexistent/stop")
                assert resp.status == 404

    @pytest.mark.asyncio
    async def test_events_exactly_once_after_approval(self):
        app = web.Application()
        run_manager = register_runtime_routes(app)
        bridge = RuntimeControlBridge(run_manager)
        app["runtime_control_bridge"] = bridge

        r = run_manager.create_run("sess_1")
        run_manager.request_approval(r["run_id"], "apr-001")

        with patch("tools.approval.resolve_gateway_approval", return_value=1):
            bridge.resolve_approval(r["run_id"], "apr-001", "once")

        events = run_manager.read_events(r["run_id"])
        assert events is not None
        approval_requested = [e for e in events["events"]
                              if e["type"] == EVENT_APPROVAL_REQUESTED]
        approval_resolved = [e for e in events["events"]
                             if e["type"] == EVENT_APPROVAL_RESOLVED]
        assert len(approval_requested) == 1
        assert len(approval_resolved) == 1


class TestRegisterRoutesFlags:
    """register_runtime_routes respects the per-route registration flags."""

    def test_register_create_false_skips_post_runs(self):
        app = web.Application()
        register_runtime_routes(app, register_create=False)

        routes = [r for r in app.router.routes()
                  if getattr(r, "method", "") == "POST"
                  and str(getattr(r.resource, "canonical", "")) == "/v1/runs"]
        assert len(routes) == 0

    def test_register_status_false_skips_get_runs(self):
        app = web.Application()
        register_runtime_routes(app, register_status=False)

        routes = [r for r in app.router.routes()
                  if getattr(r, "method", "") == "GET"
                  and str(getattr(r.resource, "canonical", "")) == "/v1/runs/{run_id}"]
        assert len(routes) == 0

    def test_register_events_false_skips_get_events(self):
        app = web.Application()
        register_runtime_routes(app, register_events=False)

        routes = [r for r in app.router.routes()
                  if getattr(r, "method", "") == "GET"
                  and str(getattr(r.resource, "canonical", "")) == "/v1/runs/{run_id}/events"]
        assert len(routes) == 0

    def test_control_routes_still_registered_when_flags_false(self):
        app = web.Application()
        register_runtime_routes(
            app, register_create=False, register_status=False,
            register_events=False,
        )

        route_paths = set()
        for r in app.router.routes():
            resource = getattr(r, "resource", None)
            if resource is None:
                continue
            canonical = str(getattr(resource, "canonical", ""))
            if canonical:
                route_paths.add(f"{getattr(r, 'method', '?')} {canonical}")

        assert "POST /v1/runs/{run_id}/stop" in route_paths
        assert "POST /v1/runs/{run_id}/approval" in route_paths
        assert "POST /v1/runs/{run_id}/clarify" in route_paths

    def test_default_flags_register_all_six(self):
        app = web.Application()
        register_runtime_routes(app)

        route_paths = set()
        for r in app.router.routes():
            resource = getattr(r, "resource", None)
            if resource is None:
                continue
            canonical = str(getattr(resource, "canonical", ""))
            if canonical:
                route_paths.add(f"{getattr(r, 'method', '?')} {canonical}")

        assert "POST /v1/runs" in route_paths
        assert "GET /v1/runs/{run_id}" in route_paths
        assert "GET /v1/runs/{run_id}/events" in route_paths
        assert "POST /v1/runs/{run_id}/stop" in route_paths
        assert "POST /v1/runs/{run_id}/approval" in route_paths
        assert "POST /v1/runs/{run_id}/clarify" in route_paths


class TestPendingIdTracking:
    """Pending IDs are added and removed correctly."""

    def test_pending_approval_ids_added_and_removed(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        mgr.request_approval(r["run_id"], "apr-002")

        status = mgr.get_status(r["run_id"])
        assert "apr-001" in status["pending_approval_ids"]
        assert "apr-002" in status["pending_approval_ids"]

        mgr.resolve_approval(r["run_id"], "apr-001", "once")
        status = mgr.get_status(r["run_id"])
        assert "apr-001" not in status["pending_approval_ids"]
        assert "apr-002" in status["pending_approval_ids"]

        mgr.resolve_approval(r["run_id"], "apr-002", "once")
        status = mgr.get_status(r["run_id"])
        assert status["pending_approval_ids"] == []

    def test_pending_clarify_ids_added_and_removed(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        mgr.request_clarify(r["run_id"], "clar-002")

        status = mgr.get_status(r["run_id"])
        assert "clar-001" in status["pending_clarify_ids"]
        assert "clar-002" in status["pending_clarify_ids"]

        mgr.resolve_clarify(r["run_id"], "clar-001", "ok")
        status = mgr.get_status(r["run_id"])
        assert "clar-001" not in status["pending_clarify_ids"]
        assert "clar-002" in status["pending_clarify_ids"]


class TestApiServerBindingIntegrationShape:
    """Verify the control bridge can be used from the API server runtime mode."""

    def test_control_bridge_stored_on_app(self):
        app = web.Application()
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        app["runtime_control_bridge"] = bridge
        app["runtime_run_manager"] = mgr
        assert app.get("runtime_control_bridge") is bridge
        assert app.get("runtime_run_manager") is mgr

    def test_runmanager_accepts_explicit_run_id(self):
        mgr = RunManager()
        custom_id = "custom_run_abc123"
        r = mgr.create_run("sess_1", run_id=custom_id)
        assert r["run_id"] == custom_id

    def test_runmanager_auto_generates_run_id_when_not_provided(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        assert r["run_id"].startswith("run_")

    def test_create_run_with_same_explicit_run_id_does_not_duplicate(self):
        mgr = RunManager()
        rid = "fixed_run_id"
        r1 = mgr.create_run("sess_1", run_id=rid)
        r2 = mgr.create_run("sess_1", run_id=rid)
        assert r1["run_id"] == rid
        assert r2["run_id"] == rid
