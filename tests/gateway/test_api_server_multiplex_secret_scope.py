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
        lambda multiplex, profile_allowlist=None: [
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

        header_accepted = await client.get(
            "/v1/test",
            headers={
                "Authorization": f"Bearer {profile_key}",
                "X-Hermes-Profile": "worker",
            },
        )
        assert header_accepted.status == 200
        assert (await header_accepted.json())["profile"] == "worker"

        header_rejects_default_key = await client.get(
            "/v1/test",
            headers={
                "Authorization": f"Bearer {default_key}",
                "X-Hermes-Profile": "worker",
            },
        )
        assert header_rejects_default_key.status == 401

        conflict = await client.get(
            "/p/default/v1/test",
            headers={
                "Authorization": f"Bearer {default_key}",
                "X-Hermes-Profile": "worker",
            },
        )
        assert conflict.status == 400
        assert (await conflict.json())["error"]["code"] == "profile_selector_conflict"


@pytest.mark.asyncio
async def test_profile_header_scopes_session_chat_and_stream_to_profile_db(
    tmp_path, monkeypatch
):
    """Both native session-chat routes must load history from the selected
    profile even when two profiles use the same session id."""
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import GatewayConfig
    from hermes_state import SessionDB

    default_home = tmp_path / "default"
    worker_home = tmp_path / "profiles" / "worker"
    default_home.mkdir(parents=True)
    worker_home.mkdir(parents=True)
    default_key = "d" * 32
    worker_key = "w" * 32
    (worker_home / ".env").write_text(
        f"API_SERVER_KEY={worker_key}\n", encoding="utf-8"
    )

    session_id = "shared-session-id"
    default_db = SessionDB(default_home / "state.db")
    worker_db = SessionDB(worker_home / "state.db")
    try:
        default_db.create_session(session_id, "api_server")
        default_db.append_message(session_id, "user", "default history")
        worker_db.create_session(session_id, "api_server")
        worker_db.append_message(session_id, "user", "worker history")
    finally:
        default_db.close()
        worker_db.close()

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._api_key = default_key
    adapter.gateway_runner = type(
        "_Runner", (), {"config": GatewayConfig(multiplex_profiles=True)}
    )()
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex, profile_allowlist=None: [
            ("default", default_home),
            ("worker", worker_home),
        ],
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: default_home if name == "default" else worker_home,
    )
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    ss.set_multiplex_active(True)

    observed = []

    async def fake_run(**kwargs):
        observed.append(
            [message.get("content") for message in kwargs["conversation_history"]]
        )
        if kwargs.get("stream_delta_callback"):
            kwargs["stream_delta_callback"]("ok")
        return (
            {
                "final_response": "ok",
                "session_id": session_id,
                "messages": [{"role": "assistant", "content": "ok"}],
            },
            {"total_tokens": 1},
        )

    monkeypatch.setattr(adapter, "_run_agent", fake_run)
    app = web.Application(
        middlewares=[adapter._make_profile_prefix_middleware()]
    )
    app.router.add_post(
        "/api/sessions/{session_id}/chat", adapter._handle_session_chat
    )
    app.router.add_post(
        "/api/sessions/{session_id}/chat/stream",
        adapter._handle_session_chat_stream,
    )

    async with TestClient(TestServer(app)) as client:
        default_response = await client.post(
            f"/api/sessions/{session_id}/chat",
            headers={"Authorization": f"Bearer {default_key}"},
            json={"message": "default turn"},
        )
        assert default_response.status == 200, await default_response.text()

        profile_headers = {
            "Authorization": f"Bearer {worker_key}",
            "X-Hermes-Profile": "worker",
        }
        worker_response = await client.post(
            f"/api/sessions/{session_id}/chat",
            headers=profile_headers,
            json={"message": "worker turn"},
        )
        assert worker_response.status == 200, await worker_response.text()

        worker_stream = await client.post(
            f"/api/sessions/{session_id}/chat/stream",
            headers=profile_headers,
            json={"message": "worker stream turn"},
        )
        assert worker_stream.status == 200, await worker_stream.text()
        await worker_stream.read()

    assert observed == [
        ["default history"],
        ["worker history"],
        ["worker history"],
    ]
