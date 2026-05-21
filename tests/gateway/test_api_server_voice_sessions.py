"""Tests for the API server mobile voice-session MVP endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/voice/sessions", adapter._handle_create_voice_session)
    app.router.add_get("/v1/voice/sessions/{voice_session_id}", adapter._handle_get_voice_session)
    app.router.add_delete("/v1/voice/sessions/{voice_session_id}", adapter._handle_delete_voice_session)
    app.router.add_post("/v1/voice/sessions/{voice_session_id}/turns", adapter._handle_voice_turn)
    return app


def _adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


@pytest.mark.asyncio
async def test_capabilities_advertise_voice_session_contract():
    adapter = _adapter()
    async with TestClient(TestServer(_make_app(adapter))) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()

    assert data["features"]["voice_sessions"] is True
    assert data["features"]["voice_text_turns"] is True
    assert data["features"]["voice_audio_upload"] is False
    assert data["endpoints"]["voice_turns"]["path"] == "/v1/voice/sessions/{voice_session_id}/turns"


@pytest.mark.asyncio
async def test_create_get_and_hangup_voice_session():
    adapter = _adapter()
    async with TestClient(TestServer(_make_app(adapter))) as cli:
        create = await cli.post("/v1/voice/sessions", json={"system_prompt": "Be concise"})
        assert create.status == 201
        created = await create.json()
        voice_session_id = created["id"]

        assert created["object"] == "hermes.voice_session"
        assert created["status"] == "active"
        assert created["capabilities"]["audio_upload"] is False
        assert created["hermes_session_id"].startswith("api-voice-")

        got = await cli.get(f"/v1/voice/sessions/{voice_session_id}")
        assert got.status == 200
        assert (await got.json())["id"] == voice_session_id

        ended = await cli.delete(f"/v1/voice/sessions/{voice_session_id}")
        assert ended.status == 200
        assert (await ended.json())["status"] == "ended"

        rejected = await cli.post(f"/v1/voice/sessions/{voice_session_id}/turns", json={"text": "hello"})
        assert rejected.status == 409


@pytest.mark.asyncio
async def test_text_turn_runs_agent_tracks_history_and_optional_tts(tmp_path):
    adapter = _adapter()
    audio_path = tmp_path / "reply.mp3"
    audio_path.write_bytes(b"fake mp3")

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        create = await cli.post("/v1/voice/sessions", json={})
        voice_session_id = (await create.json())["id"]

        run_agent = AsyncMock(return_value=({"final_response": "Hi there", "session_id": "rotated-session"}, {"total_tokens": 3}))
        with patch.object(adapter, "_run_agent", run_agent), patch(
            "tools.tts_tool.text_to_speech_tool",
            return_value=(
                '{"success": true, "file_path": "'
                + str(audio_path)
                + '", "media_tag": "MEDIA:'
                + str(audio_path)
                + '", "provider": "test"}'
            ),
        ):
            turn = await cli.post(
                f"/v1/voice/sessions/{voice_session_id}/turns",
                json={"text": "Hello Jeeves", "tts": True, "include_audio_base64": True},
            )

        assert turn.status == 200
        data = await turn.json()

    assert data["object"] == "hermes.voice_turn"
    assert data["transcript"] == "Hello Jeeves"
    assert data["reply"] == "Hi there"
    assert data["hermes_session_id"] == "rotated-session"
    assert data["audio"]["success"] is True
    assert data["audio"]["mime_type"] == "audio/mpeg"
    assert data["audio"]["base64"] == "ZmFrZSBtcDM="
    run_agent.assert_awaited_once()
    awaited = run_agent.await_args
    assert awaited is not None
    assert awaited.kwargs["conversation_history"] == []

    session = adapter._voice_sessions[voice_session_id]
    assert session["history"] == [
        {"role": "user", "content": "Hello Jeeves"},
        {"role": "assistant", "content": "Hi there"},
    ]


@pytest.mark.asyncio
async def test_audio_only_turn_documents_mvp_boundary():
    adapter = _adapter()
    async with TestClient(TestServer(_make_app(adapter))) as cli:
        create = await cli.post("/v1/voice/sessions", json={})
        voice_session_id = (await create.json())["id"]

        resp = await cli.post(
            f"/v1/voice/sessions/{voice_session_id}/turns",
            json={"audio": {"base64": "AAAA", "mime_type": "audio/wav"}},
        )
        assert resp.status == 501
        data = await resp.json()

    assert data["error"]["code"] == "voice_audio_not_supported"
    assert "Submit a transcript" in data["error"]["message"]


@pytest.mark.asyncio
async def test_json_session_key_requires_api_key_authentication():
    adapter = _adapter()
    async with TestClient(TestServer(_make_app(adapter))) as cli:
        resp = await cli.post("/v1/voice/sessions", json={"session_key": "ios-user-1"})
        assert resp.status == 403


@pytest.mark.asyncio
async def test_json_session_key_allowed_when_authenticated():
    adapter = _adapter(api_key="sk-test")
    async with TestClient(TestServer(_make_app(adapter))) as cli:
        resp = await cli.post(
            "/v1/voice/sessions",
            headers={"Authorization": "Bearer sk-test"},
            json={"session_key": "ios-user-1"},
        )
        assert resp.status == 201
        data = await resp.json()

    assert adapter._voice_sessions[data["id"]]["gateway_session_key"] == "ios-user-1"
