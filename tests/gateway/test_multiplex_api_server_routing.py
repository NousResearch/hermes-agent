"""Multiplex /p/<profile>/ routing for the api_server adapter.

Mirrors ``test_multiplex_http_routing.py`` (webhook): the process-primary
listener owns the port, and every other profile is reached via a URL prefix
when ``gateway.multiplex_profiles`` is on.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _PROFILE_REJECTED,
    _api_request_profile,
)


def _make_adapter(
    multiplex: bool = True,
    *,
    primary_profile: str = "default",
) -> APIServerAdapter:
    cfg = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 8642, "key": "test-key"})
    process_root = Path("/process-hermes")
    process_home = (
        process_root
        if primary_profile == "default"
        else process_root / "profiles" / primary_profile
    )
    with patch(
        "hermes_constants.get_process_hermes_home",
        return_value=process_home,
    ), patch(
        "hermes_constants.get_default_hermes_root",
        return_value=process_root,
    ):
        adapter = APIServerAdapter(cfg)

    class _Runner:
        config = GatewayConfig(multiplex_profiles=multiplex)

    adapter.gateway_runner = _Runner()
    return adapter


class _FakeReq:
    def __init__(self, profile=None):
        self.match_info = {"profile": profile} if profile is not None else {}


class _FakeHttpEventAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.verify_headers: list[str] = []
        self.payloads: list[dict] = []

    def verify_http_event_request(self, authorization: str) -> tuple[bool, str]:
        self.verify_headers.append(authorization)
        return True, ""

    async def dispatch_http_event(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {"adapter": self.name}


def _install_multiplex_profiles(monkeypatch, tmp_path) -> None:
    default_home = tmp_path / "default"
    coder_home = tmp_path / "profiles" / "coder"
    for home in (default_home, coder_home):
        home.mkdir(parents=True)
        (home / ".env").write_text("", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", default_home), ("coder", coder_home)],
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: {"default": default_home, "coder": coder_home}[name],
    )


def _callback_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post(
        "/api/platforms/{platform}/events",
        adapter._handle_platform_event_callback,
    )
    app.router.add_post(
        "/p/{profile}/api/platforms/{platform}/events",
        adapter._handle_platform_event_callback,
    )
    return app


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


class TestApiServerPlatformCallbacksUnderProfile:
    def test_primary_profile_resolution_ignores_request_scope(self, tmp_path, monkeypatch):
        """Process-level ownership must not follow a request-local profile."""
        default_home = tmp_path / "hermes"
        coder_home = default_home / "profiles" / "coder"
        for home in (default_home, coder_home):
            home.mkdir(parents=True)
            (home / ".env").write_text("", encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(coder_home))

        from gateway.run import _profile_runtime_scope

        with _profile_runtime_scope(default_home):
            assert APIServerAdapter._resolve_primary_profile_name() == "coder"

    def test_primary_profile_is_captured_before_request_scope(self, tmp_path, monkeypatch):
        """The shared listener's map owner must outlive request-local scope."""
        default_home = tmp_path / "hermes"
        coder_home = default_home / "profiles" / "coder"
        for home in (default_home, coder_home):
            home.mkdir(parents=True)
            (home / ".env").write_text("", encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(coder_home))
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        try:
            assert adapter._primary_profile == "coder"

            from gateway.run import _profile_runtime_scope
            from hermes_cli.profiles import get_active_profile_name

            with _profile_runtime_scope(default_home):
                assert get_active_profile_name() == "default"
                assert adapter._primary_profile == "coder"
        finally:
            adapter._response_store.close()

    @pytest.mark.asyncio
    async def test_nonprimary_profile_callback_uses_its_profile_adapter(
        self, tmp_path, monkeypatch
    ):
        _install_multiplex_profiles(monkeypatch, tmp_path)
        platform = Platform("google_chat")
        default = _FakeHttpEventAdapter("default")
        coder = _FakeHttpEventAdapter("coder")
        adapter = _make_adapter(multiplex=True)
        adapter.gateway_runner = SimpleNamespace(
            config=GatewayConfig(multiplex_profiles=True),
            adapters={platform: default},
            _profile_adapters={"coder": {platform: coder}},
        )

        try:
            async with TestClient(TestServer(_callback_app(adapter))) as client:
                response = await client.post(
                    "/p/coder/api/platforms/google_chat/events",
                    headers={"Authorization": "Bearer x"},
                    json={"type": "MESSAGE"},
                )
                body = await response.json()

            assert response.status == 200
            assert body == {"adapter": "coder"}
            assert coder.verify_headers == ["Bearer x"]
            assert coder.payloads == [{"type": "MESSAGE"}]
            assert default.verify_headers == []
            assert default.payloads == []
        finally:
            adapter._response_store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("profile_adapters", [{}, {"coder": {}}])
    async def test_nonprimary_profile_callback_does_not_fallback_to_primary_adapter(
        self,
        tmp_path,
        monkeypatch,
        profile_adapters,
    ):
        _install_multiplex_profiles(monkeypatch, tmp_path)
        platform = Platform("google_chat")
        default = _FakeHttpEventAdapter("default")
        adapter = _make_adapter(multiplex=True)
        adapter.gateway_runner = SimpleNamespace(
            config=GatewayConfig(multiplex_profiles=True),
            adapters={platform: default},
            _profile_adapters=profile_adapters,
        )

        try:
            async with TestClient(TestServer(_callback_app(adapter))) as client:
                response = await client.post(
                    "/p/coder/api/platforms/google_chat/events",
                    headers={"Authorization": "Bearer x"},
                    json={"type": "MESSAGE"},
                )
                body = await response.json()

            assert response.status == 503
            assert body["error"]["code"] == "platform_unavailable"
            assert default.verify_headers == []
            assert default.payloads == []
        finally:
            adapter._response_store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/api/platforms/google_chat/events",
            "/p/default/api/platforms/google_chat/events",
        ],
    )
    async def test_primary_profile_callback_keeps_primary_adapter(
        self, tmp_path, monkeypatch, path
    ):
        _install_multiplex_profiles(monkeypatch, tmp_path)
        platform = Platform("google_chat")
        default = _FakeHttpEventAdapter("default")
        coder = _FakeHttpEventAdapter("coder")
        adapter = _make_adapter(multiplex=True)
        adapter.gateway_runner = SimpleNamespace(
            config=GatewayConfig(multiplex_profiles=True),
            adapters={platform: default},
            _profile_adapters={"coder": {platform: coder}},
        )

        try:
            async with TestClient(TestServer(_callback_app(adapter))) as client:
                response = await client.post(
                    path,
                    headers={"Authorization": "Bearer x"},
                    json={"type": "MESSAGE"},
                )
                body = await response.json()

            assert response.status == 200
            assert body == {"adapter": "default"}
            assert default.verify_headers == ["Bearer x"]
            assert default.payloads == [{"type": "MESSAGE"}]
            assert coder.verify_headers == []
            assert coder.payloads == []
        finally:
            adapter._response_store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/api/platforms/google_chat/events",
            "/p/coder/api/platforms/google_chat/events",
        ],
    )
    async def test_named_primary_profile_callback_uses_primary_adapter(
        self, tmp_path, monkeypatch, path
    ):
        _install_multiplex_profiles(monkeypatch, tmp_path)
        platform = Platform("google_chat")
        coder = _FakeHttpEventAdapter("coder")
        default = _FakeHttpEventAdapter("default")
        adapter = _make_adapter(multiplex=True, primary_profile="coder")
        adapter.gateway_runner = SimpleNamespace(
            config=GatewayConfig(multiplex_profiles=True),
            adapters={platform: coder},
            _profile_adapters={"default": {platform: default}},
        )

        try:
            async with TestClient(TestServer(_callback_app(adapter))) as client:
                response = await client.post(
                    path,
                    headers={"Authorization": "Bearer x"},
                    json={"type": "MESSAGE"},
                )
                body = await response.json()

            assert response.status == 200
            assert body == {"adapter": "coder"}
            assert coder.verify_headers == ["Bearer x"]
            assert coder.payloads == [{"type": "MESSAGE"}]
            assert default.verify_headers == []
            assert default.payloads == []
        finally:
            adapter._response_store.close()

    @pytest.mark.asyncio
    async def test_default_callback_under_named_primary_uses_profile_adapter(
        self, tmp_path, monkeypatch
    ):
        _install_multiplex_profiles(monkeypatch, tmp_path)
        platform = Platform("google_chat")
        coder = _FakeHttpEventAdapter("coder")
        default = _FakeHttpEventAdapter("default")
        adapter = _make_adapter(multiplex=True, primary_profile="coder")
        adapter.gateway_runner = SimpleNamespace(
            config=GatewayConfig(multiplex_profiles=True),
            adapters={platform: coder},
            _profile_adapters={"default": {platform: default}},
        )

        try:
            async with TestClient(TestServer(_callback_app(adapter))) as client:
                response = await client.post(
                    "/p/default/api/platforms/google_chat/events",
                    headers={"Authorization": "Bearer x"},
                    json={"type": "MESSAGE"},
                )
                body = await response.json()

            assert response.status == 200
            assert body == {"adapter": "default"}
            assert default.verify_headers == ["Bearer x"]
            assert default.payloads == [{"type": "MESSAGE"}]
            assert coder.verify_headers == []
            assert coder.payloads == []
        finally:
            adapter._response_store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("profile_adapters", [{}, {"default": {}}])
    async def test_default_callback_under_named_primary_does_not_fallback(
        self,
        tmp_path,
        monkeypatch,
        profile_adapters,
    ):
        _install_multiplex_profiles(monkeypatch, tmp_path)
        platform = Platform("google_chat")
        coder = _FakeHttpEventAdapter("coder")
        adapter = _make_adapter(multiplex=True, primary_profile="coder")
        adapter.gateway_runner = SimpleNamespace(
            config=GatewayConfig(multiplex_profiles=True),
            adapters={platform: coder},
            _profile_adapters=profile_adapters,
        )

        try:
            async with TestClient(TestServer(_callback_app(adapter))) as client:
                response = await client.post(
                    "/p/default/api/platforms/google_chat/events",
                    headers={"Authorization": "Bearer x"},
                    json={"type": "MESSAGE"},
                )
                body = await response.json()

            assert response.status == 503
            assert body["error"]["code"] == "platform_unavailable"
            assert coder.verify_headers == []
            assert coder.payloads == []
        finally:
            adapter._response_store.close()
