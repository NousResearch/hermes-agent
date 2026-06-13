"""Tests for the API server audio transcription endpoint."""

from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/audio/transcriptions", adapter._handle_audio_transcriptions)
    return app


@pytest.mark.asyncio
async def test_audio_transcription_uploads_raw_audio_to_stt_tool():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    captured: dict[str, str] = {}

    def fake_transcribe(path: str):
        captured["path"] = path
        assert Path(path).exists()
        assert Path(path).read_bytes() == b"webm audio"
        return {"success": True, "transcript": "hello world", "provider": "openai"}

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch("tools.transcription_tools.transcribe_audio", side_effect=fake_transcribe):
            resp = await cli.post(
                "/v1/audio/transcriptions",
                data=b"webm audio",
                headers={"Content-Type": "audio/webm"},
            )
            assert resp.status == 200, await resp.text()
            body = await resp.json()

    assert body == {
        "object": "audio.transcription",
        "text": "hello world",
        "transcript": "hello world",
        "provider": "openai",
    }
    assert captured["path"].endswith(".webm")
    assert not Path(captured["path"]).exists()


@pytest.mark.asyncio
async def test_audio_transcription_rejects_non_audio_payload():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/audio/transcriptions",
            data=b"not audio",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status == 400
        body = await resp.json()

    assert body["error"]["code"] == "invalid_audio_type"
