"""Multiplex /p/<profile>/ routing for the api_server adapter.

Mirrors ``test_multiplex_http_routing.py`` (webhook): the default listener
owns the port, and secondary profiles are reached via a URL prefix when
``gateway.multiplex_profiles`` is on.
"""
from __future__ import annotations

from typing import Any, cast

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _PROFILE_REJECTED,
    _api_request_profile,
)


def _make_adapter(
    multiplex: bool = True, allowlist: list[str] | None = None
) -> APIServerAdapter:
    cfg = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 8642, "key": "test-key"})
    adapter = APIServerAdapter(cfg)

    class _Runner:
        config = GatewayConfig(
            multiplex_profiles=multiplex,
            multiplex_profile_allowlist=allowlist,
        )

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
        adapter = _make_adapter(multiplex=True, allowlist=["worker"])
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex, profile_allowlist=None: [
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


class TestApiServerRouteTable:
    def test_route_table_includes_models_options_and_chat(self):
        """Model discovery and chat routes must survive profile multiplexing."""
        adapter = _make_adapter(multiplex=True)
        paths = {path for _method, path, _handler in adapter._http_route_table()}
        assert "/v1/models" in paths
        assert "/api/model/options" in paths
        assert "/v1/chat/completions" in paths
        assert "/api/sessions/{session_id}/model" in paths
        # connect() mirrors every native path under /p/{profile}/…
        mirrored = {f"/p/{{profile}}{path}" for path in paths}
        assert "/p/{profile}/v1/models" in mirrored
        assert "/p/{profile}/api/model/options" in mirrored
        assert "/p/{profile}/v1/chat/completions" in mirrored
        assert "/p/{profile}/api/sessions/{session_id}/model" in mirrored


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

    def test_default_profile_binds_empty_profile(self):
        from gateway.session_context import clear_session_vars, get_session_env

        adapter = _make_adapter(multiplex=True)
        tokens = adapter._bind_api_server_session(
            chat_id="chat",
            session_key="key",
            session_id="sess",
            profile="",
        )
        try:
            assert get_session_env("HERMES_SESSION_PROFILE") == ""
        finally:
            clear_session_vars(tokens)

    def test_bound_profile_selects_profile_scoped_container_key(self, monkeypatch):
        """Terminal container resolution honours the api_server-bound profile."""
        import tools.terminal_tool as tt
        from gateway.session_context import clear_session_vars, get_session_env

        adapter = _make_adapter(multiplex=True)
        monkeypatch.setattr(tt, "_ensure_terminal_env_bridged", lambda: None)
        monkeypatch.setattr(
            tt, "_docker_persistent_profile_scoped", lambda: True
        )
        monkeypatch.setattr(tt, "_current_session_key", lambda: "sess-key")
        monkeypatch.setattr(
            tt, "_current_session_profile",
            lambda: get_session_env("HERMES_SESSION_PROFILE", ""),
        )

        tokens = adapter._bind_api_server_session(
            chat_id="chat",
            session_key="key",
            session_id="sess",
            profile="nm-media",
        )
        try:
            assert tt._resolve_container_task_id(None) == "profile:nm-media"
        finally:
            clear_session_vars(tokens)
