"""
Tests for the GET /healthz endpoint on the API server adapter.

The /healthz endpoint is intended for monitoring/probing tools (kube probes,
uptime monitors). It MUST be unauthenticated and never 500 — even when
runtime_status is missing or partial — and returns a small JSON envelope:

    {
      "status": "ok",
      "version": "<hermes_cli.__version__ or 'unknown'>",
      "uptime_seconds": <float, >=0>,
      "connected_platforms": <int, >=0>
    }
"""

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

_MOD = "gateway.platforms.api_server"


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Minimal app — just /healthz plus /v1/capabilities for the discovery test."""
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/healthz", adapter._handle_healthz)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    """Adapter with API key configured — proves /healthz still skips auth."""
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# Happy-path shape tests
# ---------------------------------------------------------------------------


class TestHealthzShape:
    @pytest.mark.asyncio
    async def test_returns_200(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/healthz")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_response_is_json(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/healthz")
            assert resp.headers.get("Content-Type", "").startswith("application/json")

    @pytest.mark.asyncio
    async def test_has_all_four_keys(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/healthz")
            data = await resp.json()
            assert set(data.keys()) >= {
                "status",
                "version",
                "uptime_seconds",
                "connected_platforms",
            }

    @pytest.mark.asyncio
    async def test_status_is_ok(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/healthz")
            data = await resp.json()
            assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_version_is_nonempty_string(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/healthz")
            data = await resp.json()
            assert isinstance(data["version"], str)
            assert data["version"] != ""

    @pytest.mark.asyncio
    async def test_uptime_seconds_is_float_nonnegative(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/healthz")
            data = await resp.json()
            assert isinstance(data["uptime_seconds"], (int, float))
            assert not isinstance(data["uptime_seconds"], bool)  # bool subclasses int
            assert float(data["uptime_seconds"]) >= 0.0

    @pytest.mark.asyncio
    async def test_connected_platforms_is_int_nonnegative(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/healthz")
            data = await resp.json()
            assert isinstance(data["connected_platforms"], int)
            assert not isinstance(data["connected_platforms"], bool)
            assert data["connected_platforms"] >= 0


# ---------------------------------------------------------------------------
# Auth: /healthz must NEVER require auth, even when API_SERVER_KEY is set
# ---------------------------------------------------------------------------


class TestHealthzAuth:
    @pytest.mark.asyncio
    async def test_no_auth_header_returns_200(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # Explicitly send no Authorization header.
            resp = await cli.get("/healthz")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_unauthenticated_when_api_key_configured(self, auth_adapter):
        """Even with an API key configured, /healthz must remain open."""
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/healthz")
            assert resp.status == 200, (
                "/healthz must not require auth even when API_SERVER_KEY is set; "
                f"got {resp.status}"
            )
            data = await resp.json()
            assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Runtime data shaping
# ---------------------------------------------------------------------------


class TestHealthzRuntimeShaping:
    @pytest.mark.asyncio
    async def test_counts_connected_platforms(self, adapter):
        """2 connected + 1 disconnected -> count of 2."""
        runtime = {
            "started_at": 0.0,  # ancient -> uptime is large but >= 0
            "platforms": {
                "telegram": {"state": "connected"},
                "discord": {"state": "connected"},
                "slack": {"state": "disconnected"},
            },
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}.read_runtime_status", return_value=runtime):
                resp = await cli.get("/healthz")
                assert resp.status == 200
                data = await resp.json()
                assert data["connected_platforms"] == 2

    @pytest.mark.asyncio
    async def test_running_state_also_counts_as_connected(self, adapter):
        """Some platforms report 'running' instead of 'connected'."""
        runtime = {
            "started_at": 0.0,
            "platforms": {
                "api_server": {"state": "running"},
                "telegram": {"state": "connected"},
            },
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}.read_runtime_status", return_value=runtime):
                resp = await cli.get("/healthz")
                data = await resp.json()
                assert data["connected_platforms"] == 2

    @pytest.mark.asyncio
    async def test_missing_platforms_dict_yields_zero(self, adapter):
        runtime = {"started_at": 0.0}  # no 'platforms' key
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}.read_runtime_status", return_value=runtime):
                resp = await cli.get("/healthz")
                data = await resp.json()
                assert data["connected_platforms"] == 0

    @pytest.mark.asyncio
    async def test_runtime_status_none_does_not_500(self, adapter):
        """read_runtime_status() can return None when the file isn't written yet."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}.read_runtime_status", return_value=None):
                resp = await cli.get("/healthz")
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["connected_platforms"] == 0
                assert float(data["uptime_seconds"]) >= 0.0

    @pytest.mark.asyncio
    async def test_uptime_clamped_when_started_at_is_future(self, adapter):
        """Clock skew can yield started_at > now; uptime must clamp to >= 0."""
        import time as _time

        runtime = {
            "started_at": _time.time() + 10_000.0,  # in the future
            "platforms": {},
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}.read_runtime_status", return_value=runtime):
                resp = await cli.get("/healthz")
                data = await resp.json()
                assert float(data["uptime_seconds"]) >= 0.0


# ---------------------------------------------------------------------------
# Capabilities map should advertise /healthz so dashboards can discover it
# ---------------------------------------------------------------------------


class TestHealthzCapabilities:
    @pytest.mark.asyncio
    async def test_healthz_listed_in_capabilities(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            endpoints = data.get("endpoints", {})
            assert "healthz" in endpoints, (
                "Capabilities map must list /healthz so external UIs can discover it"
            )
            assert endpoints["healthz"]["method"] == "GET"
            assert endpoints["healthz"]["path"] == "/healthz"
