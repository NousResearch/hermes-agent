"""Multiplex /p/<profile>/ routing for the api_server adapter.

Mirrors ``test_multiplex_http_routing.py`` (webhook): the default listener
owns the port, and secondary profiles are reached via a URL prefix when
``gateway.multiplex_profiles`` is on.
"""
from __future__ import annotations

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

    def test_prefix_ignored_when_multiplex_off(self):
        adapter = _make_adapter(multiplex=False)
        # Even a bogus profile is ignored (not 404'd) when multiplexing is off.
        assert adapter._resolve_request_profile(_FakeReq("anything")) is None

    def test_known_profile_accepted(self, monkeypatch):
        adapter = _make_adapter(multiplex=True)
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex: [("default", None), ("coder", None)],
        )
        assert adapter._resolve_request_profile(_FakeReq("coder")) == "coder"

    def test_unknown_profile_rejected(self, monkeypatch):
        adapter = _make_adapter(multiplex=True)
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex: [("default", None), ("coder", None)],
        )
        assert adapter._resolve_request_profile(_FakeReq("ghost")) is _PROFILE_REJECTED


class TestApiServerRouteTable:
    def test_route_table_includes_models_and_chat(self):
        """ /p/{profile}/v1/models must be registered — this is the 404 Fadeway hit. """
        adapter = _make_adapter(multiplex=True)
        paths = {path for _method, path, _handler in adapter._http_route_table()}
        assert "/v1/models" in paths
        assert "/v1/chat/completions" in paths
        # connect() mirrors every native path under /p/{profile}/…
        mirrored = {f"/p/{{profile}}{path}" for path in paths}
        assert "/p/{profile}/v1/models" in mirrored
        assert "/p/{profile}/v1/chat/completions" in mirrored


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
