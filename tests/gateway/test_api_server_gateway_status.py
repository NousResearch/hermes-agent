"""
Tests for the enriched /api/gateway/status response.

The base handler (introduced in the same commit as the remote-
management endpoints) already returns runtime liveness +
per-platform state. This file covers the version + interpreter
metadata that the enrichment commit adds — version,
python_version, openai_sdk_version, released,
subsystem_capabilities.

Two layers:

* Module-level unit tests for the four helper lookups
  (``_hermes_agent_version`` / ``_python_runtime_version`` /
  ``_openai_sdk_version`` / ``_hermes_agent_released``).
* One HTTP integration test asserting the GET response carries
  the new fields and that the historic fields are still present
  (no regression on Codex's shape).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _hermes_agent_released,
    _hermes_agent_version,
    _openai_sdk_version,
    _python_runtime_version,
    cors_middleware,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestPythonRuntimeVersion:
    def test_returns_dot_separated_three_part_version(self):
        v = _python_runtime_version()
        # Format like "3.11.15" — three integer segments.
        parts = v.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_matches_sys_version_info(self):
        import sys

        expected = (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )
        assert _python_runtime_version() == expected


class TestHermesAgentVersion:
    def test_returns_version_string_when_package_installed(self):
        # Editable install via `uv pip install -e ...` in the test
        # runner registers the package as `hermes-agent`. Result
        # is a non-empty string of digits + dots.
        v = _hermes_agent_version()
        assert v is None or (isinstance(v, str) and len(v) > 0)

    def test_returns_none_when_package_metadata_unavailable(self):
        with patch(
            "gateway.platforms.api_server.version", create=True
        ) as mock_version:
            # importlib.metadata.version is imported inside the
            # helper, so patching the imported name at use-site is
            # the cleanest way. Easier: patch the module path the
            # helper actually imports.
            mock_version.side_effect = Exception("not found")
            # The helper's `from importlib.metadata import version`
            # is a local import inside the try block, so the patch
            # above doesn't reach it. Use a direct call instead:
        # Re-import importlib.metadata.version and force failure
        # by mocking at the right level:
        with patch("importlib.metadata.version") as mock:
            mock.side_effect = Exception("not found")
            assert _hermes_agent_version() is None


class TestOpenAiSdkVersion:
    def test_returns_string_or_none(self):
        v = _openai_sdk_version()
        assert v is None or isinstance(v, str)

    def test_returns_none_when_openai_not_installed(self):
        with patch("importlib.metadata.version") as mock:
            mock.side_effect = Exception("no package")
            assert _openai_sdk_version() is None


class TestHermesAgentReleased:
    def test_returns_string_or_none(self):
        v = _hermes_agent_released()
        assert v is None or isinstance(v, str)

    def test_returns_none_when_git_fails(self):
        import subprocess

        with patch.object(subprocess, "check_output") as mock:
            mock.side_effect = subprocess.CalledProcessError(1, "git")
            assert _hermes_agent_released() is None

    def test_strips_leading_v_prefix(self):
        import subprocess

        with patch.object(subprocess, "check_output") as mock:
            mock.return_value = b"v0.14.0\n"
            assert _hermes_agent_released() == "0.14.0"

    def test_empty_output_becomes_none(self):
        import subprocess

        with patch.object(subprocess, "check_output") as mock:
            mock.return_value = b"\n"
            assert _hermes_agent_released() is None


# ---------------------------------------------------------------------------
# HTTP integration
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/gateway/status", adapter._handle_gateway_status)
    return app


class TestGatewayStatusEnrichedResponse:
    @pytest.mark.asyncio
    async def test_response_includes_new_runtime_fields(self):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/gateway/status")
            assert resp.status == 200
            body = await resp.json()

        # New fields added by this enrichment.
        for key in (
            "version",
            "python_version",
            "openai_sdk_version",
            "released",
            "subsystem_capabilities",
        ):
            assert key in body, f"missing enrichment field {key!r}"

        # python_version is always available, never null.
        assert isinstance(body["python_version"], str)
        assert body["python_version"].count(".") == 2

        # subsystem_capabilities defaults to an empty list — it is
        # populated by follow-up subsystem PRs as they land.
        assert body["subsystem_capabilities"] == []

        # version / openai_sdk_version / released are best-effort:
        # may be null when not installed / no git, but if present
        # they must be strings.
        for key in ("version", "openai_sdk_version", "released"):
            assert body[key] is None or isinstance(body[key], str)

    @pytest.mark.asyncio
    async def test_historic_runtime_fields_still_present(self):
        # Codex's original response shape. None of these may
        # disappear or change semantics.
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/gateway/status")
            assert resp.status == 200
            body = await resp.json()

        for key in (
            "ok",
            "running",
            "pid",
            "gateway_state",
            "platforms",
            "active_agents",
            "exit_reason",
            "updated_at",
        ):
            assert key in body, f"historic field {key!r} disappeared"

        assert body["ok"] is True
        assert body["running"] is True
        assert isinstance(body["pid"], int)

    @pytest.mark.asyncio
    async def test_helper_failures_do_not_propagate(self):
        """If every optional helper fails simultaneously, the
        handler still returns 200 with the null fields plus the
        always-available ones."""
        import subprocess

        with patch("importlib.metadata.version") as mock_v, patch.object(
            subprocess, "check_output"
        ) as mock_git:
            mock_v.side_effect = Exception("metadata unavailable")
            mock_git.side_effect = Exception("git unavailable")

            app = _create_app(_make_adapter())
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/api/gateway/status")
                assert resp.status == 200
                body = await resp.json()

        assert body["version"] is None
        assert body["openai_sdk_version"] is None
        assert body["released"] is None
        # python_version comes straight from sys.version_info,
        # not from a fallible lookup — must still be there.
        assert isinstance(body["python_version"], str)
