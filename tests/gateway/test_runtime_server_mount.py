"""Server-mount integration tests for gateway.runtime.routes.

Validates that the runtime /v1/runs route module can be mounted onto
an aiohttp ``web.Application`` and that all six endpoints produce the
correct HTTP responses, status codes, and error shapes.

Covers:
- POST /v1/runs — create a run (202)
- GET  /v1/runs/{run_id} — poll status
- GET  /v1/runs/{run_id}/events — JSON event replay
- POST /v1/runs/{run_id}/stop — interrupt
- POST /v1/runs/{run_id}/approval — resolve lifecycle
- POST /v1/runs/{run_id}/clarify — resolve lifecycle
- Unknown run → 404
- Secret redaction in responses
- Malformed requests → 400
- Query parameter validation
"""

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.runtime.routes import register_runtime_routes
from gateway.runtime.run_manager import RunManager


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _create_runtime_app() -> web.Application:
    app = web.Application()
    register_runtime_routes(app)
    return app


def _create_body(**overrides):
    base = {
        "session_id": "sess_123",
        "message": "User request",
        "workspace": "/home/user/workspace",
        "profile": "default",
        "model": "provider/model",
        "toolsets": ["terminal", "file"],
        "metadata": {"client": "webui", "client_version": "unknown"},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# POST /v1/runs — create
# ---------------------------------------------------------------------------


class TestCreateRunViaServer:

    @pytest.mark.asyncio
    async def test_creates_run_returns_202(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            body = _create_body()
            resp = await cli.post("/v1/runs", json=body)
            assert resp.status == 202
            data = await resp.json()
            assert data["run_id"].startswith("run_")
            assert data["session_id"] == "sess_123"
            assert data["status"] == "queued"
            assert "events_url" in data
            assert "status_url" in data
            assert data["object"] == "hermes.run"

    @pytest.mark.asyncio
    async def test_creates_with_minimal_body(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={})
            assert resp.status == 202
            data = await resp.json()
            assert data["run_id"].startswith("run_")
            assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_handles_input_as_array(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            body = _create_body()
            body["input"] = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "world"},
            ]
            resp = await cli.post("/v1/runs", json=body)
            assert resp.status == 202
            data = await resp.json()
            assert data["run_id"].startswith("run_")
            events_resp = await cli.get(data["events_url"])
            events_data = await events_resp.json()
            assert len(events_data["events"]) == 1
            assert events_data["events"][0]["type"] == "run.started"

    @pytest.mark.asyncio
    async def test_redacts_secrets_in_payload(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            body = _create_body()
            body["metadata"]["api_key"] = "sk-secret-key-12345"
            resp = await cli.post("/v1/runs", json=body)
            assert resp.status == 202
            data = await resp.json()
            events_resp = await cli.get(data["events_url"])
            events_data = await events_resp.json()
            started = events_data["events"][0]
            meta = started["payload"]["metadata"]
            assert meta.get("api_key") != "sk-secret-key-12345"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — status
# ---------------------------------------------------------------------------


class TestGetRunViaServer:

    @pytest.mark.asyncio
    async def test_returns_status(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.get(f"/v1/runs/{run_id}")
            assert resp.status == 200
            status = await resp.json()
            assert status["run_id"] == run_id
            assert status["object"] == "hermes.run"
            assert status["session_id"] == "sess_123"
            assert status["status"] == "queued"
            assert "last_event_id" in status
            assert "last_seq" in status
            assert "terminal" in status
            assert "controls" in status

    @pytest.mark.asyncio
    async def test_unknown_run_returns_404(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent")
            assert resp.status == 404
            data = await resp.json()
            assert "error" in data
            assert data["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_status_redacts_secrets(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            body = _create_body()
            body["metadata"]["token"] = "sk-abc-secret-key"
            create_resp = await cli.post("/v1/runs", json=body)
            data = await create_resp.json()

            resp = await cli.get(f"/v1/runs/{data['run_id']}")
            assert resp.status == 200
            text = await resp.text()
            assert "sk-abc-secret-key" not in text


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — replay
# ---------------------------------------------------------------------------


class TestGetEventsViaServer:

    @pytest.mark.asyncio
    async def test_returns_events(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.get(f"/v1/runs/{run_id}/events")
            assert resp.status == 200
            events = await resp.json()
            assert events["run_id"] == run_id
            assert events["object"] == "hermes.run.events"
            assert len(events["events"]) == 1
            assert events["events"][0]["type"] == "run.started"

    @pytest.mark.asyncio
    async def test_events_support_after_seq(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.get(
                f"/v1/runs/{run_id}/events", params={"after_seq": "0"}
            )
            assert resp.status == 200
            events = await resp.json()
            assert len(events["events"]) == 1

    @pytest.mark.asyncio
    async def test_events_support_limit(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.get(
                f"/v1/runs/{run_id}/events", params={"limit": "1"}
            )
            assert resp.status == 200
            events = await resp.json()
            assert len(events["events"]) == 1

    @pytest.mark.asyncio
    async def test_after_seq_invalid_returns_400(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.get(
                f"/v1/runs/{run_id}/events", params={"after_seq": "not_a_number"}
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_limit_invalid_returns_400(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.get(
                f"/v1/runs/{run_id}/events", params={"limit": "bad"}
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_unknown_run_returns_404(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent/events")
            assert resp.status == 404
            data = await resp.json()
            assert data["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_events_redact_secrets(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            body = _create_body()
            body["metadata"]["bearer"] = "Bearer sk-secret-redact_test"
            create_resp = await cli.post("/v1/runs", json=body)
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.get(f"/v1/runs/{run_id}/events")
            assert resp.status == 200
            text = await resp.text()
            assert "sk-secret-redact_test" not in text

    @pytest.mark.asyncio
    async def test_events_sse_when_accept_header(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.get(
                f"/v1/runs/{run_id}/events",
                headers={"Accept": "text/event-stream"},
            )
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "text/event-stream"
            text = await resp.text()
            assert "data: {" in text
            assert "run.started" in text
            assert ": stream closed" in text


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt
# ---------------------------------------------------------------------------


class TestStopRunViaServer:

    @pytest.mark.asyncio
    async def test_stop_transitions_to_cancelled(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.post(f"/v1/runs/{run_id}/stop")
            assert resp.status == 200
            body = await resp.json()
            assert body["run_id"] == run_id
            assert body["status"] == "cancelled"
            assert body["terminal"] is True

    @pytest.mark.asyncio
    async def test_unknown_run_returns_404(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_nonexistent/stop")
            assert resp.status == 404
            data = await resp.json()
            assert data["error"]["code"] == "run_not_found"


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/approval — resolve (not_found without request)
# ---------------------------------------------------------------------------


class TestApprovalViaServer:

    @pytest.mark.asyncio
    async def test_without_request_returns_404(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.post(
                f"/v1/runs/{run_id}/approval",
                json={"choice": "once", "approval_id": "apr_test"},
            )
            assert resp.status == 404
            body = await resp.json()
            assert body["error"]["code"] == "action_not_found"

    @pytest.mark.asyncio
    async def test_missing_choice_returns_400(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.post(
                f"/v1/runs/{run_id}/approval",
                json={},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_unknown_run_returns_404(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_nonexistent/approval",
                json={"choice": "deny", "approval_id": "apr-1"},
            )
            assert resp.status == 404
            data = await resp.json()
            assert data["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_url_body_run_id_mismatch_returns_400(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.post(
                f"/v1/runs/{run_id}/approval",
                json={"choice": "once", "run_id": "different_run"},
            )
            assert resp.status == 400


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/clarify — resolve (not_found without request)
# ---------------------------------------------------------------------------


class TestClarifyViaServer:

    @pytest.mark.asyncio
    async def test_without_request_returns_404(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.post(
                f"/v1/runs/{run_id}/clarify",
                json={"response": "my answer", "clarify_id": "clar_test"},
            )
            assert resp.status == 404
            body = await resp.json()
            assert body["error"]["code"] == "action_not_found"

    @pytest.mark.asyncio
    async def test_accepts_answer_field(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.post(
                f"/v1/runs/{run_id}/clarify",
                json={"answer": "my answer", "clarify_id": "clar_test"},
            )
            assert resp.status == 404
            body = await resp.json()
            assert body["error"]["code"] == "action_not_found"

    @pytest.mark.asyncio
    async def test_missing_response_returns_400(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.post(
                f"/v1/runs/{run_id}/clarify",
                json={},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_unknown_run_returns_404(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_nonexistent/clarify",
                json={"response": "text", "clarify_id": "clar-1"},
            )
            assert resp.status == 404
            data = await resp.json()
            assert data["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_url_body_run_id_mismatch_returns_400(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.post(
                f"/v1/runs/{run_id}/clarify",
                json={"response": "yes", "run_id": "different_run"},
            )
            assert resp.status == 400


# ---------------------------------------------------------------------------
# Route present check — all six endpoints are registered
# ---------------------------------------------------------------------------


class TestAllRoutesRegistered:

    def test_all_six_endpoints_registered(self):
        app = _create_runtime_app()
        routes = [
            (r.method, r.resource.canonical) for r in app.router.routes()
        ]
        registered = set(routes)
        expected = {
            ("POST", "/v1/runs"),
            ("GET", "/v1/runs/{run_id}"),
            ("GET", "/v1/runs/{run_id}/events"),
            ("POST", "/v1/runs/{run_id}/stop"),
            ("POST", "/v1/runs/{run_id}/approval"),
            ("POST", "/v1/runs/{run_id}/clarify"),
        }
        missing = expected - registered
        assert not missing, f"Missing routes: {missing}"


# ---------------------------------------------------------------------------
# RunManager stored on app
# ---------------------------------------------------------------------------


class TestRunManagerOnApp:

    def test_run_manager_stored_on_app(self):
        app = _create_runtime_app()
        assert "runtime_run_manager" in app
        from gateway.runtime.run_manager import RunManager
        assert isinstance(app["runtime_run_manager"], RunManager)

    @pytest.mark.asyncio
    async def test_same_manager_across_requests(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            data = await create_resp.json()
            run_id = data["run_id"]

            resp = await cli.get(f"/v1/runs/{run_id}")
            assert resp.status == 200
            assert data["run_id"] == (await resp.json())["run_id"]


# ---------------------------------------------------------------------------
# Malformed request handling
# ---------------------------------------------------------------------------


class TestMalformedRequests:

    @pytest.mark.asyncio
    async def test_unknown_run_all_endpoints_404(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            segments = ["", "events", "stop", "approval", "clarify"]
            for seg in segments:
                url = "/v1/runs/run_nonexistent"
                if seg:
                    url += f"/{seg}"
                if seg in ("stop", "approval", "clarify"):
                    resp = await cli.post(
                        url,
                        json={"choice": "deny", "response": "x"},
                    )
                else:
                    resp = await cli.get(url)
                assert resp.status == 404, f"{url} expected 404 got {resp.status}"

    @pytest.mark.asyncio
    async def test_multiple_creates_independent_runs(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            r1 = await cli.post("/v1/runs", json=_create_body())
            r2 = await cli.post("/v1/runs", json=_create_body())
            d1 = await r1.json()
            d2 = await r2.json()
            assert d1["run_id"] != d2["run_id"]
            assert d1["run_id"].startswith("run_")
            assert d2["run_id"].startswith("run_")

    @pytest.mark.asyncio
    async def test_empty_body_not_json_returns_400(self):
        app = _create_runtime_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                data="",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400


# ---------------------------------------------------------------------------
# Control bridge mount tests
# ---------------------------------------------------------------------------


class TestBridgeMount:
    """Routes can receive a RuntimeControlBridge."""

    def test_bridge_stored_on_app_when_provided(self):
        from gateway.runtime.control_bridge import RuntimeControlBridge

        app = web.Application()
        bridge = RuntimeControlBridge(RunManager())
        register_runtime_routes(app, control_bridge=bridge)
        assert "runtime_control_bridge" in app
        assert app["runtime_control_bridge"] is bridge

    def test_no_bridge_stored_when_not_provided(self):
        app = web.Application()
        register_runtime_routes(app)
        assert "runtime_control_bridge" not in app

    @pytest.mark.asyncio
    async def test_approval_uses_bridge_when_present(self):
        from unittest.mock import MagicMock

        app = web.Application()
        mock_bridge = MagicMock()
        mock_bridge.resolve_approval.return_value = {
            "status": "resolved",
            "type": "approval",
        }
        register_runtime_routes(app, control_bridge=mock_bridge)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            run_id = (await create_resp.json())["run_id"]

            resp = await cli.post(
                f"/v1/runs/{run_id}/approval",
                json={"choice": "once", "approval_id": "apr-1"},
            )
            assert resp.status == 200
            mock_bridge.resolve_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_clarify_uses_bridge_when_present(self):
        from unittest.mock import MagicMock

        app = web.Application()
        mock_bridge = MagicMock()
        mock_bridge.resolve_clarify.return_value = {
            "status": "resolved",
            "type": "clarify",
        }
        register_runtime_routes(app, control_bridge=mock_bridge)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            run_id = (await create_resp.json())["run_id"]

            resp = await cli.post(
                f"/v1/runs/{run_id}/clarify",
                json={"response": "yes", "clarify_id": "clar-1"},
            )
            assert resp.status == 200
            mock_bridge.resolve_clarify.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_uses_bridge_when_present(self):
        from unittest.mock import MagicMock

        app = web.Application()
        mock_bridge = MagicMock()
        mock_bridge.stop_run.return_value = {
            "status": "cancelled",
            "terminal": True,
            "controls": [],
        }
        register_runtime_routes(app, control_bridge=mock_bridge)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            run_id = (await create_resp.json())["run_id"]

            resp = await cli.post(f"/v1/runs/{run_id}/stop", json={})
            assert resp.status == 200
            mock_bridge.stop_run.assert_called_once_with(run_id)

    @pytest.mark.asyncio
    async def test_bridge_not_found_maps_404(self):
        from unittest.mock import MagicMock

        app = web.Application()
        mock_bridge = MagicMock()
        mock_bridge.resolve_approval.return_value = {
            "error": "not_found",
            "message": "Approval apr-001 not found for run run_x.",
        }
        register_runtime_routes(app, control_bridge=mock_bridge)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_x/approval",
                json={"choice": "once", "approval_id": "apr-001"},
            )
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_bridge_conflict_maps_409(self):
        from unittest.mock import MagicMock

        app = web.Application()
        mock_bridge = MagicMock()
        mock_bridge.resolve_approval.return_value = {
            "error": "conflict",
            "message": "Approval apr-001 has already been resolved.",
        }
        register_runtime_routes(app, control_bridge=mock_bridge)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_x/approval",
                json={"choice": "once", "approval_id": "apr-001"},
            )
            assert resp.status == 409

    @pytest.mark.asyncio
    async def test_url_run_id_wins_over_body_when_bridge_present(self):
        from unittest.mock import MagicMock

        app = web.Application()
        mock_bridge = MagicMock()
        mock_bridge.resolve_approval.return_value = {
            "status": "resolved",
        }
        register_runtime_routes(app, control_bridge=mock_bridge)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/url_run/approval",
                json={
                    "choice": "once",
                    "run_id": "body_run",
                    "approval_id": "apr-1",
                },
            )
            assert resp.status == 400


class TestBridgeDynamicMount:
    """Bridge can be set on app AFTER route registration."""

    @pytest.mark.asyncio
    async def test_bridge_set_after_mount_used_by_handlers(self):
        from unittest.mock import MagicMock

        app = web.Application()
        register_runtime_routes(app)

        mock_bridge = MagicMock()
        mock_bridge.resolve_approval.return_value = {
            "status": "resolved",
            "type": "approval",
        }
        app["runtime_control_bridge"] = mock_bridge

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/runs", json=_create_body())
            run_id = (await create_resp.json())["run_id"]

            resp = await cli.post(
                f"/v1/runs/{run_id}/approval",
                json={"choice": "once", "approval_id": "apr-1"},
            )
            assert resp.status == 200
            mock_bridge.resolve_approval.assert_called_once()

