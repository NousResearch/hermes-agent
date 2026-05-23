"""
Tests for the /api/providers read-only inventory endpoint.

The endpoint reports which LLM providers are configured for the
current profile (env-var or config.yaml) and which one is the
currently-selected model.provider. **It never returns the API
keys themselves.**

Coverage:

* Unit: ``_list_provider_status`` helper across the three
  detection layers (env var, yaml providers map, active
  model.provider).
* Integration: HTTP responses, auth gate, and a regression
  guard asserting no fixture API key appears in the response.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _list_provider_status,
    cors_middleware,
)


# ---------------------------------------------------------------------------
# Module-level — _list_provider_status
# ---------------------------------------------------------------------------


class TestListProviderStatusViaEnv:
    def test_env_with_filled_api_key_marks_provider_configured(self, tmp_path):
        (tmp_path / ".env").write_text(
            "ANTHROPIC_API_KEY=sk-test-anthropic\n", encoding="utf-8"
        )
        providers, active = _list_provider_status(tmp_path)
        by_key = {p["key"]: p for p in providers}
        assert by_key["anthropic"]["configured"] is True
        assert active is None

    def test_env_with_empty_value_does_not_mark_configured(self, tmp_path):
        (tmp_path / ".env").write_text("OPENAI_API_KEY=\n", encoding="utf-8")
        providers, _ = _list_provider_status(tmp_path)
        by_key = {p["key"]: p for p in providers}
        assert by_key["openai"]["configured"] is False

    def test_dashed_provider_name_matches_underscored_env_var(self, tmp_path):
        # openai-codex should be detected by OPENAI_CODEX_API_KEY
        (tmp_path / ".env").write_text(
            "OPENAI_CODEX_API_KEY=sk-codex\n", encoding="utf-8"
        )
        providers, _ = _list_provider_status(tmp_path)
        by_key = {p["key"]: p for p in providers}
        assert by_key["openai-codex"]["configured"] is True

    def test_missing_env_file_yields_no_configured_providers(self, tmp_path):
        providers, active = _list_provider_status(tmp_path)
        # All known providers come back with configured=False
        assert all(p["configured"] is False for p in providers)
        assert active is None


class TestListProviderStatusViaConfigYaml:
    def test_yaml_providers_map_marks_configured(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "providers:\n  anthropic:\n    base_url: https://api.anthropic.com\n",
            encoding="utf-8",
        )
        providers, active = _list_provider_status(tmp_path)
        by_key = {p["key"]: p for p in providers}
        assert by_key["anthropic"]["configured"] is True
        assert active is None  # providers map alone doesn't mark active

    def test_yaml_model_provider_marks_active_and_configured(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "model:\n  provider: openai-codex\n  default: gpt-5.5\n",
            encoding="utf-8",
        )
        providers, active = _list_provider_status(tmp_path)
        by_key = {p["key"]: p for p in providers}
        assert active == "openai-codex"
        assert by_key["openai-codex"]["configured"] is True
        assert by_key["openai-codex"]["active"] is True
        # Other known providers stay inactive / unconfigured
        assert by_key["anthropic"]["active"] is False
        assert by_key["anthropic"]["configured"] is False

    def test_unknown_provider_in_yaml_is_appended(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "providers:\n  my-custom-thing:\n    base_url: x\n",
            encoding="utf-8",
        )
        providers, _ = _list_provider_status(tmp_path)
        keys = [p["key"] for p in providers]
        assert "my-custom-thing" in keys

    def test_malformed_yaml_does_not_crash_resolver(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "this is not :: valid yaml [", encoding="utf-8"
        )
        # Should silently return the known-providers list with no
        # configured entries, rather than raising.
        providers, active = _list_provider_status(tmp_path)
        assert isinstance(providers, list)
        assert active is None


class TestListProviderStatusNoKeyLeakage:
    def test_response_payload_never_echoes_api_key_value(self, tmp_path):
        secret = "sk-FIXTURE-DO-NOT-LEAK-7m4p"
        (tmp_path / ".env").write_text(
            f"ANTHROPIC_API_KEY={secret}\nOPENAI_API_KEY={secret}-v2\n",
            encoding="utf-8",
        )
        providers, _ = _list_provider_status(tmp_path)
        flat = repr(providers)
        assert secret not in flat, (
            "API key fixture leaked into the providers listing"
        )


# ---------------------------------------------------------------------------
# HTTP integration
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra: Dict[str, Any] = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/providers", adapter._handle_list_providers)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


class TestProvidersEndpoint:
    @pytest.mark.asyncio
    async def test_response_shape(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "model:\n  provider: openai-codex\n", encoding="utf-8"
        )

        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/providers")
            assert resp.status == 200
            body = await resp.json()

        assert "providers" in body
        assert "active" in body
        assert body["active"] == "openai-codex"

        # Each entry has the documented fields, nothing more.
        for entry in body["providers"]:
            assert set(entry.keys()) == {"key", "label", "configured", "active"}

    @pytest.mark.asyncio
    async def test_requires_auth_when_key_configured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        app = _create_app(_make_adapter(api_key="sk-secret"))
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/providers")
            assert resp.status == 401
            resp = await cli.get(
                "/api/providers", headers={"Authorization": "Bearer sk-secret"}
            )
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_response_never_contains_real_api_key(self, tmp_path, monkeypatch):
        secret = "sk-CANARY-providers-test-99zz"
        (tmp_path / ".env").write_text(
            f"ANTHROPIC_API_KEY={secret}\n", encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/providers")
            raw = await resp.text()

        assert secret not in raw, "API key leaked into HTTP response body"


class TestCapabilitiesAdvertisesProviders:
    @pytest.mark.asyncio
    async def test_capabilities_lists_providers_endpoint(self):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            body = await resp.json()

        assert body["features"]["remote_providers"] is True
        assert body["endpoints"]["providers"]["path"] == "/api/providers"
        assert body["endpoints"]["providers"]["method"] == "GET"
