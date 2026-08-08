"""Tests for headless provider OAuth on the API server.

Covers the flow module (session lifecycle, code parsing, persistence) and
the two HTTP handlers (auth gate, validation, error mapping).
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import provider_oauth
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.fixture(autouse=True)
def _clear_sessions():
    with provider_oauth._sessions_lock:
        provider_oauth._sessions.clear()
    yield
    with provider_oauth._sessions_lock:
        provider_oauth._sessions.clear()


def _app(api_key: str = "k" * 32) -> web.Application:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": api_key}))
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post(
        "/api/providers/{provider}/oauth/start",
        adapter._handle_provider_oauth_start,
    )
    app.router.add_post(
        "/api/providers/{provider}/oauth/submit",
        adapter._handle_provider_oauth_submit,
    )
    return app


class TestFlow:

    def test_start_returns_auth_url_and_session(self):
        out = provider_oauth.start("anthropic")
        assert out["flow"] == "pkce"
        assert out["auth_url"].startswith("https://claude.ai/oauth/authorize?")
        assert "code_challenge=" in out["auth_url"]
        assert "code_challenge_method=S256" in out["auth_url"]
        assert out["session_id"] in provider_oauth._sessions

    def test_start_rejects_unknown_provider(self):
        with pytest.raises(provider_oauth.ProviderOAuthError) as exc:
            provider_oauth.start("openai")
        assert exc.value.status == 400

    def test_submit_unknown_session_is_404(self):
        with pytest.raises(provider_oauth.ProviderOAuthError) as exc:
            provider_oauth.submit("nope", "code#state")
        assert exc.value.status == 404

    def test_submit_requires_a_code(self):
        sid = provider_oauth.start("anthropic")["session_id"]
        with pytest.raises(provider_oauth.ProviderOAuthError) as exc:
            provider_oauth.submit(sid, "   ")
        assert exc.value.status == 400

    def test_submit_exchanges_and_persists(self):
        started = provider_oauth.start("anthropic")
        sid = started["session_id"]
        verifier = provider_oauth._sessions[sid]["verifier"]

        captured = {}

        def _fake_exchange(token_urls, payload):
            captured.update(json.loads(payload.decode()))
            return {
                "access_token": "at",
                "refresh_token": "rt",
                "expires_in": 3600,
            }

        with patch.object(provider_oauth, "_exchange", _fake_exchange), patch.object(
            provider_oauth, "_persist_anthropic"
        ) as persist:
            out = provider_oauth.submit(sid, "  thecode#thestate  ")

        assert out["ok"] is True
        # code/state split on '#', verifier bound to the session
        assert captured["code"] == "thecode"
        assert captured["state"] == "thestate"
        assert captured["code_verifier"] == verifier
        assert captured["grant_type"] == "authorization_code"
        persist.assert_called_once()
        assert persist.call_args[0][0] == "at"
        assert persist.call_args[0][1] == "rt"
        # session is consumed — a code cannot be replayed
        assert sid not in provider_oauth._sessions

    def test_submit_without_state_falls_back_to_session_state(self):
        sid = provider_oauth.start("anthropic")["session_id"]
        verifier = provider_oauth._sessions[sid]["verifier"]
        captured = {}

        def _fake_exchange(token_urls, payload):
            captured.update(json.loads(payload.decode()))
            return {"access_token": "at", "refresh_token": "rt", "expires_in": 60}

        with patch.object(provider_oauth, "_exchange", _fake_exchange), patch.object(
            provider_oauth, "_persist_anthropic"
        ):
            provider_oauth.submit(sid, "onlycode")

        assert captured["state"] == verifier

    def test_missing_access_token_is_502(self):
        sid = provider_oauth.start("anthropic")["session_id"]
        with patch.object(
            provider_oauth, "_exchange", lambda *_: {"refresh_token": "rt"}
        ):
            with pytest.raises(provider_oauth.ProviderOAuthError) as exc:
                provider_oauth.submit(sid, "code#state")
        assert exc.value.status == 502

    def test_sessions_are_capped(self):
        for _ in range(provider_oauth._MAX_SESSIONS + 5):
            provider_oauth.start("anthropic")
        assert len(provider_oauth._sessions) <= provider_oauth._MAX_SESSIONS

    def test_expired_session_is_pruned(self):
        sid = provider_oauth.start("anthropic")["session_id"]
        with provider_oauth._sessions_lock:
            provider_oauth._sessions[sid]["created_at"] -= (
                provider_oauth.SESSION_TTL_SECONDS + 1
            )
        with pytest.raises(provider_oauth.ProviderOAuthError) as exc:
            provider_oauth.submit(sid, "code#state")
        assert exc.value.status == 404


class TestHandlers:

    @pytest.mark.asyncio
    async def test_start_requires_auth(self):
        async with TestClient(TestServer(_app())) as cli:
            resp = await cli.post("/api/providers/anthropic/oauth/start")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_start_returns_url_with_auth(self):
        key = "k" * 32
        async with TestClient(TestServer(_app(key))) as cli:
            resp = await cli.post(
                "/api/providers/anthropic/oauth/start",
                headers={"Authorization": f"Bearer {key}"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["auth_url"].startswith("https://claude.ai/oauth/authorize?")
            assert body["session_id"]

    @pytest.mark.asyncio
    async def test_unsupported_provider_maps_to_400(self):
        key = "k" * 32
        async with TestClient(TestServer(_app(key))) as cli:
            resp = await cli.post(
                "/api/providers/openai/oauth/start",
                headers={"Authorization": f"Bearer {key}"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_submit_requires_session_id(self):
        key = "k" * 32
        async with TestClient(TestServer(_app(key))) as cli:
            resp = await cli.post(
                "/api/providers/anthropic/oauth/submit",
                headers={"Authorization": f"Bearer {key}"},
                json={"code": "abc"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_submit_success_roundtrip(self):
        key = "k" * 32
        async with TestClient(TestServer(_app(key))) as cli:
            started = await (
                await cli.post(
                    "/api/providers/anthropic/oauth/start",
                    headers={"Authorization": f"Bearer {key}"},
                )
            ).json()
            with patch.object(
                provider_oauth,
                "_exchange",
                lambda *_: {
                    "access_token": "at",
                    "refresh_token": "rt",
                    "expires_in": 3600,
                },
            ), patch.object(provider_oauth, "_persist_anthropic"):
                resp = await cli.post(
                    "/api/providers/anthropic/oauth/submit",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"session_id": started["session_id"], "code": "c#s"},
                )
            assert resp.status == 200
            assert (await resp.json())["ok"] is True


class TestPersistence:

    def test_persist_writes_file_and_pool(self, tmp_path):
        oauth_file = tmp_path / ".anthropic_oauth.json"
        pool = MagicMock()
        pool.entries.return_value = []

        with patch(
            "agent.anthropic_adapter._get_hermes_oauth_file", return_value=oauth_file
        ), patch("agent.credential_pool.load_pool", return_value=pool):
            provider_oauth._persist_anthropic("at", "rt", 1234)

        written = json.loads(oauth_file.read_text())
        assert written["accessToken"] == "at"
        assert written["refreshToken"] == "rt"
        assert written["expiresAt"] == 1234
        # file must not be world/group readable
        assert oct(oauth_file.stat().st_mode)[-3:] == "600"
        pool.add_entry.assert_called_once()

    def test_pool_failure_does_not_break_login(self, tmp_path):
        oauth_file = tmp_path / ".anthropic_oauth.json"
        with patch(
            "agent.anthropic_adapter._get_hermes_oauth_file", return_value=oauth_file
        ), patch("agent.credential_pool.load_pool", side_effect=RuntimeError("boom")):
            provider_oauth._persist_anthropic("at", "rt", 1234)
        assert json.loads(oauth_file.read_text())["accessToken"] == "at"
