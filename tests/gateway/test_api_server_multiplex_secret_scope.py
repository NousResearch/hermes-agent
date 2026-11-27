"""Regression for #61276: api_server agent entry under multiplex isolation.

When gateway.multiplex_profiles is on, get_secret fails closed without a
profile secret scope. Requests with a ``/p/<profile>/`` prefix are scoped by
``_profile_scope(profile)``, but plain requests on the default listener used
to get ``nullcontext()`` — so agent runs crashed with UnscopedSecretError on
their first credential read (e.g. OPENROUTER_BASE_URL). ``_profile_scope``
now enters the DEFAULT profile's runtime scope when multiplex is active and
no profile was requested.

Adapted from PR #61283 by @giggling-ginger (originally targeting a
pre-``_profile_scope`` helper); no live gateway or network.
"""

from __future__ import annotations

import gateway.run  # noqa: F401  settle gateway.run's module init at collection time
import pytest

from agent import secret_scope as ss
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


class TestProfileScopeDefaultFallback:
    def test_noop_when_multiplex_off(self, adapter, monkeypatch):
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://from-environ.example/v1")
        with adapter._profile_scope(None):
            # Legacy single-profile path: unscoped get_secret reads os.environ.
            assert ss.get_secret("OPENROUTER_BASE_URL") == "https://from-environ.example/v1"
        assert ss.current_secret_scope() is None


# Regression coverage for #134095: the prefix-less default scope is a routed-profile
# decision and must resolve from the launch home, not a live-mirrored HERMES_HOME.
class TestProfileScopeResolvesLaunchHome:
    def test_default_scope_ignores_live_hermes_home_mirror(self, adapter, tmp_path, monkeypatch):
        """An embedding host that mirrors the active turn's profile into HERMES_HOME while
        the listener serves a prefix-less request must not make that request swap scopes:
        auth was checked against the launch profile's key, so the runtime scope (session
        DB, secret scope, home override) has to stay the launch profile's."""
        from hermes_constants import pin_process_hermes_home

        launch_home = tmp_path / "profiles" / "default"
        other_home = tmp_path / "profiles" / "worker"
        launch_home.mkdir(parents=True)
        other_home.mkdir(parents=True)
        (launch_home / ".env").write_text("LAUNCH_MARKER=launch\n", encoding="utf-8")
        (other_home / ".env").write_text("WORKER_MARKER=worker\n", encoding="utf-8")

        pin_process_hermes_home(launch_home)
        monkeypatch.setenv("HERMES_HOME", str(other_home))
        ss.set_multiplex_active(True)
        try:
            with adapter._profile_scope(None):
                assert ss.get_secret("LAUNCH_MARKER") == "launch"
                assert ss.get_secret("WORKER_MARKER") is None
        finally:
            pin_process_hermes_home(None)

    def test_default_scope_follows_process_home_when_not_pinned(self, adapter, tmp_path, monkeypatch):
        """Without a pin the routing home is the process home, so an ordinary multiplex
        gateway keeps entering the home HERMES_HOME names (unchanged behavior)."""
        home = tmp_path / "profiles" / "default"
        home.mkdir(parents=True)
        (home / ".env").write_text("HOME_MARKER=present\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        ss.set_multiplex_active(True)
        with adapter._profile_scope(None):
            assert ss.get_secret("HOME_MARKER") == "present"


# Regression coverage for #72041: profile-bound API authentication
class TestProfileScopedApiAuthentication:
    @staticmethod
    def _request(token: str):
        from types import SimpleNamespace

        return SimpleNamespace(
            headers={"Authorization": f"Bearer {token}"},
            remote="127.0.0.1",
            transport=None,
            method="GET",
            path_qs="/p/worker/v1/models",
        )

    def test_named_profile_rejects_default_listener_key(
        self, adapter, tmp_path, monkeypatch
    ):
        from gateway.platforms.api_server import _api_request_profile

        profile_home = tmp_path / "profiles" / "worker"
        profile_home.mkdir(parents=True)
        profile_key = "worker-profile-api-key-123456"
        default_key = "default-listener-api-key-123456"
        (profile_home / ".env").write_text(
            f"API_SERVER_KEY={profile_key}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: profile_home,
        )
        adapter._api_key = default_key
        ss.set_multiplex_active(True)

        profile_token = _api_request_profile.set("worker")
        try:
            with adapter._profile_scope("worker"):
                assert adapter._check_auth(self._request(profile_key)) is None

                rejected = adapter._check_auth(self._request(default_key))
                assert rejected is not None
                assert rejected.status == 401
        finally:
            _api_request_profile.reset(profile_token)


@pytest.mark.asyncio
async def test_profile_middleware_binds_auth_before_handler(
    adapter, tmp_path, monkeypatch
):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import GatewayConfig
    from gateway.platforms.api_server import _api_request_profile

    worker_home = tmp_path / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    profile_key = "a" * 32
    default_key = "b" * 32
    (worker_home / ".env").write_text(
        f"API_SERVER_KEY={profile_key}\n", encoding="utf-8"
    )
    adapter._api_key = default_key
    adapter.gateway_runner = type(
        "_Runner", (), {"config": GatewayConfig(multiplex_profiles=True)}
    )()
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [
            ("default", tmp_path), ("worker", worker_home)
        ],
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: tmp_path if name == "default" else worker_home,
    )
    ss.set_multiplex_active(True)

    async def authenticated(request):
        auth_error = adapter._check_auth(request)
        if auth_error is not None:
            return auth_error
        return web.json_response(
            {"profile": _api_request_profile.get() or "default"}
        )

    app = web.Application(
        middlewares=[adapter._make_profile_prefix_middleware()]
    )
    app.router.add_get("/v1/test", authenticated)
    app.router.add_get("/p/{profile}/v1/test", authenticated)

    async with TestClient(TestServer(app)) as client:
        default_response = await client.get(
            "/v1/test",
            headers={"Authorization": f"Bearer {default_key}"},
        )
        assert default_response.status == 200

        default_alias = await client.get(
            "/p/default/v1/test",
            headers={"Authorization": f"Bearer {default_key}"},
        )
        assert default_alias.status == 200

        rejected = await client.get(
            "/p/worker/v1/test",
            headers={"Authorization": f"Bearer {default_key}"},
        )
        assert rejected.status == 401

        accepted = await client.get(
            "/p/worker/v1/test",
            headers={"Authorization": f"Bearer {profile_key}"},
        )
        assert accepted.status == 200
        assert (await accepted.json())["profile"] == "worker"


