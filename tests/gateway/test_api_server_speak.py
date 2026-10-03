"""Tests for the /api/audio/speak TTS endpoint on the API server adapter.

Covers:
- Successful synthesis returns a base64 data URL and cleans up the temp file.
- Empty text is rejected with 400.
- A TTS provider failure (success=False) is surfaced as 400 with the error.
- Auth enforcement (401 when API_SERVER_KEY is set and no bearer is supplied).

Mirrors the dashboard's hermes_cli/web_routers/audio.py::speak_text contract so
remote-desktop sessions routed through the gateway get the same response shape.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

_TTS = "tools.tts_tool.text_to_speech_tool"


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/audio/speak", adapter._handle_speak)
    return app


class TestSpeak:
    @pytest.mark.asyncio
    async def test_speak_success_returns_data_url(self, tmp_path):
        adapter = _make_adapter(api_key="sk-secret")
        audio = tmp_path / "out.mp3"
        audio.write_bytes(b"ID3\x04\x00" + b"\x00" * 32)  # plausible mp3-ish bytes
        mock_tts = MagicMock(return_value=json.dumps({
            "success": True,
            "file_path": str(audio),
            "provider": "edge-tts",
        }))
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(_TTS, mock_tts):
                resp = await cli.post(
                    "/api/audio/speak",
                    json={"text": "hello"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True
            assert data["mime_type"] == "audio/mpeg"
            assert data["provider"] == "edge-tts"
            assert data["data_url"].startswith("data:audio/mpeg;base64,")
            # the temp file is unlinked off-loop after being read
            assert not audio.exists()

    @pytest.mark.asyncio
    async def test_speak_empty_text_returns_400(self):
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(_TTS, MagicMock()):
                resp = await cli.post(
                    "/api/audio/speak",
                    json={"text": "   "},
                    headers={"Authorization": "Bearer sk-secret"},
                )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_speak_tts_failure_returns_400(self):
        adapter = _make_adapter(api_key="sk-secret")
        mock_tts = MagicMock(return_value=json.dumps({
            "success": False,
            "error": "no provider configured",
        }))
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(_TTS, mock_tts):
                resp = await cli.post(
                    "/api/audio/speak",
                    json={"text": "hi"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
            assert resp.status == 400
            data = await resp.json()
            assert data["error"] == "no provider configured"

    @pytest.mark.asyncio
    async def test_speak_requires_auth(self):
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(_TTS, MagicMock()):
                resp = await cli.post("/api/audio/speak", json={"text": "hi"})
            assert resp.status == 401
