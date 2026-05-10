"""Tests for /api/config/model — backend default model + provider read/write.

The handler reads/writes the {primary, provider} pair into ~/.hermes/config.yaml
via hermes_cli.config.{load_config, save_config}. The gateway's per-request
runtime kwargs cache is keyed on config.yaml mtime, so a successful PUT takes
effect on the next /v1/chat/completions / /v1/responses / /v1/runs call without
restarting the gateway.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _CORS_HEADERS,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/config/model", adapter._handle_model_config_get)
    app.router.add_put("/api/config/model", adapter._handle_model_config_put)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


@pytest.fixture
def temp_config(tmp_path: Path, monkeypatch):
    """Redirect HERMES_HOME to a tmp dir so save_config writes there."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Some helpers cache the path lazily — clear the load cache to be safe.
    try:
        from hermes_cli import config as _cfg
        _cfg._RAW_CONFIG_CACHE.clear()
        _cfg._LOAD_CONFIG_CACHE.clear()
        _cfg._LAST_EXPANDED_CONFIG_BY_PATH.clear()
        # Reset module-level _hermes_home reference used in gateway.run helpers
        from gateway import run as _gateway_run
        _gateway_run._hermes_home = home
    except Exception:
        pass
    yield home


def _seed_config(temp_config: Path, content: str) -> Path:
    config_path = temp_config / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Static helper
# ---------------------------------------------------------------------------


class TestResolveModelConfigView:
    def test_full_dict(self):
        view = APIServerAdapter._resolve_model_config_view(
            {"model": {"default": "kimi-k2.6", "provider": "opencode-go"}}
        )
        assert view == {"primary": "kimi-k2.6", "provider": "opencode-go"}

    def test_legacy_string_form(self):
        view = APIServerAdapter._resolve_model_config_view({"model": "gpt-5.5"})
        assert view == {"primary": "gpt-5.5", "provider": None}

    def test_legacy_inner_model_alias(self):
        view = APIServerAdapter._resolve_model_config_view(
            {"model": {"model": "claude-3-5-haiku", "provider": "anthropic"}}
        )
        assert view == {"primary": "claude-3-5-haiku", "provider": "anthropic"}

    def test_missing_section(self):
        view = APIServerAdapter._resolve_model_config_view({})
        assert view == {"primary": None, "provider": None}

    def test_non_dict_input(self):
        view = APIServerAdapter._resolve_model_config_view(None)
        assert view == {"primary": None, "provider": None}


# ---------------------------------------------------------------------------
# GET /api/config/model
# ---------------------------------------------------------------------------


class TestModelConfigGet:
    @pytest.mark.asyncio
    async def test_returns_current_config(self, adapter, temp_config):
        _seed_config(
            temp_config,
            "model:\n  default: kimi-k2.6\n  provider: opencode-go\n",
        )
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.get("/api/config/model")
            assert resp.status == 200
            data = await resp.json()
            assert data == {"primary": "kimi-k2.6", "provider": "opencode-go"}

    @pytest.mark.asyncio
    async def test_returns_null_when_section_missing(self, adapter, temp_config):
        _seed_config(temp_config, "agent:\n  max_turns: 10\n")
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.get("/api/config/model")
            assert resp.status == 200
            data = await resp.json()
            assert data["primary"] is None
            assert data["provider"] is None

    @pytest.mark.asyncio
    async def test_returns_null_when_load_raises(self, adapter, temp_config):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                resp = await cli.get("/api/config/model")
                assert resp.status == 200
                data = await resp.json()
                assert data == {"primary": None, "provider": None}

    @pytest.mark.asyncio
    async def test_requires_auth_when_key_set(self, auth_adapter, temp_config):
        _seed_config(temp_config, "model:\n  default: gpt-5.5\n")
        async with TestClient(TestServer(_create_app(auth_adapter))) as cli:
            resp = await cli.get("/api/config/model")
            assert resp.status == 401

            resp = await cli.get(
                "/api/config/model",
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 200


# ---------------------------------------------------------------------------
# PUT /api/config/model
# ---------------------------------------------------------------------------


class TestModelConfigPut:
    @pytest.mark.asyncio
    async def test_writes_primary_and_provider(self, adapter, temp_config):
        _seed_config(
            temp_config,
            "model:\n  default: gpt-5.5\n  provider: openai-codex\n",
        )
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.put(
                "/api/config/model",
                json={"primary": "kimi-k2.6", "provider": "opencode-go"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data == {
                "ok": True,
                "primary": "kimi-k2.6",
                "provider": "opencode-go",
            }

        # Round-trip via load_config
        from hermes_cli.config import (
            _LOAD_CONFIG_CACHE,
            _RAW_CONFIG_CACHE,
            load_config,
        )
        _LOAD_CONFIG_CACHE.clear()
        _RAW_CONFIG_CACHE.clear()
        cfg = load_config()
        assert cfg["model"]["default"] == "kimi-k2.6"
        assert cfg["model"]["provider"] == "opencode-go"

    @pytest.mark.asyncio
    async def test_writes_primary_only_keeps_existing_provider(self, adapter, temp_config):
        _seed_config(
            temp_config,
            "model:\n  default: gpt-5.5\n  provider: openai-codex\n",
        )
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.put(
                "/api/config/model",
                json={"primary": "claude-3-5-haiku"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["primary"] == "claude-3-5-haiku"
            assert data["provider"] == "openai-codex"

    @pytest.mark.asyncio
    async def test_preserves_unrelated_config_keys(self, adapter, temp_config):
        _seed_config(
            temp_config,
            "model:\n"
            "  default: gpt-5.5\n"
            "  provider: openai-codex\n"
            "agent:\n"
            "  max_turns: 42\n"
            "toolsets:\n"
            "- hermes-cli\n",
        )
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.put(
                "/api/config/model",
                json={"primary": "kimi-k2.6", "provider": "opencode-go"},
            )
            assert resp.status == 200

        from hermes_cli.config import (
            _LOAD_CONFIG_CACHE,
            _RAW_CONFIG_CACHE,
            load_config,
        )
        _LOAD_CONFIG_CACHE.clear()
        _RAW_CONFIG_CACHE.clear()
        cfg = load_config()
        assert cfg["model"]["default"] == "kimi-k2.6"
        assert cfg["agent"]["max_turns"] == 42
        assert "hermes-cli" in cfg["toolsets"]

    @pytest.mark.asyncio
    async def test_rejects_non_dict_body(self, adapter, temp_config):
        _seed_config(temp_config, "")
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.put("/api/config/model", json=["not", "a", "dict"])
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_rejects_missing_primary(self, adapter, temp_config):
        _seed_config(temp_config, "")
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.put("/api/config/model", json={"provider": "openai-codex"})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_rejects_empty_primary(self, adapter, temp_config):
        _seed_config(temp_config, "")
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.put("/api/config/model", json={"primary": "   "})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_rejects_path_traversal_in_primary(self, adapter, temp_config):
        _seed_config(temp_config, "")
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            for bad in ["../etc/passwd", "model;rm -rf /", "model with spaces", "<script>"]:
                resp = await cli.put("/api/config/model", json={"primary": bad})
                assert resp.status == 400, f"expected 400 for {bad!r}"

    @pytest.mark.asyncio
    async def test_rejects_path_traversal_in_provider(self, adapter, temp_config):
        _seed_config(temp_config, "")
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            for bad in ["..", "with space", "semi;colon", "slash/here"]:
                resp = await cli.put(
                    "/api/config/model",
                    json={"primary": "kimi-k2.6", "provider": bad},
                )
                assert resp.status == 400, f"expected 400 for provider={bad!r}"

    @pytest.mark.asyncio
    async def test_rejects_oversized_primary(self, adapter, temp_config):
        _seed_config(temp_config, "")
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.put(
                "/api/config/model",
                json={"primary": "a" * 201},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_rejects_invalid_json_body(self, adapter, temp_config):
        _seed_config(temp_config, "")
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.put(
                "/api/config/model",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_returns_409_when_managed(self, adapter, temp_config):
        _seed_config(temp_config, "model:\n  default: gpt-5.5\n")
        with patch("hermes_cli.config.is_managed", return_value=True):
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                resp = await cli.put(
                    "/api/config/model",
                    json={"primary": "kimi-k2.6"},
                )
                assert resp.status == 409

    @pytest.mark.asyncio
    async def test_returns_500_on_save_error(self, adapter, temp_config):
        _seed_config(temp_config, "model:\n  default: gpt-5.5\n")
        with patch(
            "hermes_cli.config.save_config",
            side_effect=PermissionError("read-only fs"),
        ):
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                resp = await cli.put(
                    "/api/config/model",
                    json={"primary": "kimi-k2.6"},
                )
                assert resp.status == 500

    @pytest.mark.asyncio
    async def test_requires_auth_when_key_set(self, auth_adapter, temp_config):
        _seed_config(temp_config, "model:\n  default: gpt-5.5\n")
        async with TestClient(TestServer(_create_app(auth_adapter))) as cli:
            resp = await cli.put(
                "/api/config/model",
                json={"primary": "kimi-k2.6"},
            )
            assert resp.status == 401

            resp = await cli.put(
                "/api/config/model",
                headers={"Authorization": "Bearer sk-secret"},
                json={"primary": "kimi-k2.6"},
            )
            assert resp.status == 200
