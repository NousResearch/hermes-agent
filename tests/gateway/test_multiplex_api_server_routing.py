"""Multiplex /p/<profile>/ routing for the api_server adapter.

Mirrors ``test_multiplex_http_routing.py`` (webhook): the default listener
owns the port, and secondary profiles are reached via a URL prefix when
``gateway.multiplex_profiles`` is on.
"""
from __future__ import annotations

from typing import Any, cast

import pytest

from agent import secret_scope as ss
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _PROFILE_REJECTED,
    _api_request_profile,
)


def _make_adapter(multiplex: bool = True) -> APIServerAdapter:
    cfg = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 8642, "key": "test-key"})
    adapter = APIServerAdapter(cfg)

    class _Runner:
        config = GatewayConfig(multiplex_profiles=multiplex)

    adapter.gateway_runner = _Runner()
    return adapter


class _FakeReq:
    def __init__(self, profile=None):
        self.match_info = {"profile": profile} if profile is not None else {}


class TestApiServerProfileResolution:
    def test_no_prefix_returns_none(self):
        adapter = _make_adapter(multiplex=True)
        assert adapter._resolve_request_profile(_FakeReq(None)) is None

    def test_unserved_prefix_is_rejected(self, monkeypatch):
        adapter = _make_adapter(multiplex=True)
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex: [
                ("default", "/profiles/default"),
                ("worker", "/profiles/worker"),
            ],
        )

        assert (
            adapter._resolve_request_profile(cast(Any, _FakeReq("worker")))
            == "worker"
        )
        assert (
            adapter._resolve_request_profile(cast(Any, _FakeReq("restricted")))
            is _PROFILE_REJECTED
        )




class TestApiServerModelsUnderProfile:
    def test_resolve_model_name_follows_active_profile(self, monkeypatch):
        """When the request is scoped to a named profile, advertise that name."""
        adapter = _make_adapter(multiplex=True)
        adapter._model_name = "hermes-agent"
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name",
            lambda: "coder",
        )
        token_prof = _api_request_profile.set("coder")
        try:
            assert adapter._resolve_model_name("") == "coder"
        finally:
            _api_request_profile.reset(token_prof)

    @pytest.mark.asyncio
    async def test_models_uses_profile_model_name_override(self, tmp_path, monkeypatch):
        """A multiplexed profile exposes its configured API model alias."""
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        profile_home = tmp_path / "profiles" / "worker"
        profile_home.mkdir(parents=True)
        profile_key = "worker-profile-api-key-123456"
        (profile_home / ".env").write_text(
            f"API_SERVER_KEY={profile_key}\nAPI_SERVER_MODEL_NAME=worker-api\n",
            encoding="utf-8",
        )
        adapter = _make_adapter(multiplex=True)
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex: [("default", tmp_path), ("worker", profile_home)],
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path if name == "default" else profile_home,
        )
        ss.set_multiplex_active(True)

        app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
        app.router.add_get("/p/{profile}/v1/models", adapter._handle_models)
        try:
            async with TestClient(TestServer(app)) as client:
                response = await client.get(
                    "/p/worker/v1/models",
                    headers={"Authorization": f"Bearer {profile_key}"},
                )
                assert response.status == 200
                assert (await response.json())["data"][0]["id"] == "worker-api"
        finally:
            ss.set_multiplex_active(False)


class TestApiServerSessionProfileBinding:
    """HERMES_SESSION_PROFILE must be bound per /p/<profile>/ request.

    Regression guard for cross-profile sandbox reuse: before the fix,
    _bind_api_server_session never passed ``profile`` to set_session_vars,
    so every API-server turn bound HERMES_SESSION_PROFILE="" and the
    terminal tool collapsed ALL api_server sessions (default AND org
    profiles) onto the shared "default" container key — letting org-profile
    turns reuse the default profile's sandbox (SSH key / secrets exposure).
    """

    def test_profile_is_bound_into_session_vars(self):
        from gateway.session_context import clear_session_vars, get_session_env

        adapter = _make_adapter(multiplex=True)
        tokens = adapter._bind_api_server_session(
            chat_id="chat",
            session_key="key",
            session_id="sess",
            profile="nm-media",
        )
        try:
            assert get_session_env("HERMES_SESSION_PROFILE") == "nm-media"
        finally:
            clear_session_vars(tokens)

    def test_bound_profile_selects_profile_scoped_container_key(self, monkeypatch):
        """Persistent-Docker container keys honour the api_server-bound profile (real resolver chain)."""
        import tools.terminal_tool as tt
        from gateway.session_context import clear_session_vars

        adapter = _make_adapter(multiplex=True)
        monkeypatch.setattr(tt, "_ensure_terminal_env_bridged", lambda: None)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
        monkeypatch.delenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", raising=False)

        tokens = adapter._bind_api_server_session(
            chat_id="chat", session_key="key", session_id="sess", profile="nm-media",
        )
        try:
            assert tt._resolve_container_task_id(None) == "profile:nm-media"
        finally:
            clear_session_vars(tokens)
