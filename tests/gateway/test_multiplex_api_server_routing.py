"""Multiplex /p/<profile>/ routing for the api_server adapter.

Mirrors ``test_multiplex_http_routing.py`` (webhook): the default listener
owns the port, and secondary profiles are reached via a URL prefix when
``gateway.multiplex_profiles`` is on.
"""
from __future__ import annotations

from typing import Any, cast

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

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

    def test_hermes_handle_aliases_to_default_profile(self, monkeypatch):
        """A peer advertises its default profile's @handle as 'hermes' (botHandle()), but the
        mirror is keyed by profile name — '/p/hermes/' must reach the same profile '/p/default/'
        does (#118076), not 404."""
        adapter = _make_adapter(multiplex=True)
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex: [
                ("default", "/profiles/default"),
                ("worker", "/profiles/worker"),
            ],
        )

        assert (
            adapter._resolve_request_profile(cast(Any, _FakeReq("hermes")))
            == "default"
        )

    def test_real_hermes_profile_is_not_shadowed_by_the_alias(self, monkeypatch):
        """A gateway that genuinely serves a profile literally named 'hermes' must keep
        resolving to itself, never silently redirect to 'default'."""
        adapter = _make_adapter(multiplex=True)
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex: [
                ("default", "/profiles/default"),
                ("hermes", "/profiles/hermes"),
            ],
        )

        assert (
            adapter._resolve_request_profile(cast(Any, _FakeReq("hermes")))
            == "hermes"
        )


_UNKNOWN = {"error": "Unknown or unconfigured profile"}
_NOT_SERVED = "profile_not_served"


@pytest.mark.asyncio
@pytest.mark.parametrize("multiplex,gateway_home,legacy_hermes,answers", [
    # Single-profile gateways: the handle is the default profile's, and nobody else's.
    (False, "", None, {"hermes": None, "default": None, "worker": _UNKNOWN}),
    (False, "profiles/worker", None, {"hermes": _UNKNOWN, "default": _UNKNOWN, "worker": None}),
    (False, "profiles/hermes", "live", {"hermes": None, "default": _UNKNOWN}),
    # A real legacy 'hermes' profile the gateway does not serve is never answered by default.
    (False, "", "live", {"hermes": _NOT_SERVED, "default": None}),
    (True, "", None, {"hermes": "default", "worker": "worker"}),
    (True, "", "live", {"hermes": "hermes", "default": "default"}),
    (True, "", "parked", {"hermes": _NOT_SERVED, "default": "default"}),
])
async def test_hermes_handle_reaches_the_default_profile_only_when_no_hermes_profile_exists(
        tmp_path, monkeypatch, multiplex, gateway_home, legacy_hermes, answers):
    """'/p/hermes/' is Bot Mode's handle for the default profile (#118076): a gateway serving the
    default profile answers it, one running another profile refuses it with the unknown-profile
    404 (peer clients report it as ambiguous, never reroute), and a host with a real pre-reservation
    'hermes' profile keeps the name for it (refusing distinctly when this gateway does not serve it)."""
    root = tmp_path / "root"
    homes = [root, root / "profiles" / "worker"] + ([root / "profiles" / "hermes"] if legacy_hermes else [])
    for home in homes:
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    if legacy_hermes == "parked":
        (root / "profiles" / "hermes" / "gateway.parked").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root / gateway_home))
    adapter = _make_adapter(multiplex=multiplex)

    async def handler(request):
        return web.json_response({"profile": _api_request_profile.get()})

    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_get("/p/{profile}/v1/test", handler)
    async with TestClient(TestServer(app)) as cli:
        for prefix, expected in answers.items():
            resp = await cli.get(f"/p/{prefix}/v1/test")
            body = await resp.json()
            if expected == _NOT_SERVED:
                assert (prefix, resp.status, body["error"]["code"]) == (prefix, 404, _NOT_SERVED)
            elif expected == _UNKNOWN:
                assert (prefix, resp.status, body) == (prefix, 404, _UNKNOWN)
            else:
                assert (prefix, resp.status, body) == (prefix, 200, {"profile": expected})


@pytest.mark.asyncio
@pytest.mark.parametrize("multiplex,legacy_hermes,answers", [
    (True, True, {"Hermes": "hermes", "HERMES": "hermes", "Worker": "worker", "Default": "default"}),
    (True, False, {"Hermes": "default", "Worker": "worker"}),
    (False, True, {"Hermes": _NOT_SERVED}),
    (False, False, {"Hermes": None}),
])
async def test_profile_prefix_names_the_same_profile_in_any_case(tmp_path, monkeypatch, multiplex, legacy_hermes, answers):
    """Profile ids are lowercase (``normalize_profile_name``), so ``/p/Hermes/`` is ``/p/hermes/``:
    a real 'hermes' profile keeps it, and only without one does it alias the default. An
    exact-case comparison answered it with the unknown-profile 404, so box/Hermes reached no
    agent (and older peer clients, which rerouted on that 404, sent it to the default profile)."""
    root = tmp_path / "root"
    for home in [root, root / "profiles" / "worker"] + ([root / "profiles" / "hermes"] if legacy_hermes else []):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    adapter = _make_adapter(multiplex=multiplex)

    async def handler(request):
        return web.json_response({"profile": _api_request_profile.get()})

    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_get("/p/{profile}/v1/test", handler)
    async with TestClient(TestServer(app)) as cli:
        for prefix, expected in answers.items():
            resp = await cli.get(f"/p/{prefix}/v1/test")
            body = await resp.json()
            if expected == _NOT_SERVED:
                assert (prefix, resp.status, body["error"]["code"]) == (prefix, 404, _NOT_SERVED)
            else:
                assert (prefix, resp.status, body) == (prefix, 200, {"profile": expected})





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
