"""Tests for voice mode in POST /v1/runs (D13).

Covers:
- message_type='voice' in body → TTS called, audio_path in run.completed SSE event
- X-Hermes-Voice: 'true' header → TTS called
- Normal run (no voice flag) → TTS not called, no audio_path
"""
import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_adapter() -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={})
    return APIServerAdapter(config)


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    return app


def _fake_agent(response: str = "hello from forge") -> MagicMock:
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": response}
    agent.session_prompt_tokens = 1
    agent.session_completion_tokens = 2
    agent.session_total_tokens = 3
    return agent


async def _collect_run_completed(cli, run_id: str) -> dict:
    """Poll SSE stream until run.completed; return that event."""
    async with cli.get(f"/v1/runs/{run_id}/events") as resp:
        assert resp.status == 200
        async for raw in resp.content:
            line = raw.decode().strip()
            if line.startswith("data:"):
                ev = json.loads(line[5:])
                if ev.get("event") == "run.completed":
                    return ev
    return {}


# ── tests ─────────────────────────────────────────────────────────────────────

class TestVoiceModeInRuns:

    @pytest.mark.asyncio
    async def test_voice_body_flag_calls_tts_and_includes_audio_base64(self, tmp_path):
        """message_type='voice' in body → TTS fires, audio_base64 in run.completed."""
        adapter = _make_adapter()
        # Write a tiny valid WAV to a real temp file so the code can read it
        fake_wav = tmp_path / "tts_out.mp3"
        fake_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        fake_tts = json.dumps({"file_path": str(fake_wav), "text": "hello from forge"})

        with (
            patch.object(adapter, "_create_agent", return_value=_fake_agent()),
            patch("tools.tts_tool.check_tts_requirements", return_value=True),
            patch("tools.tts_tool.text_to_speech_tool", return_value=fake_tts) as mock_tts,
        ):
            app = _create_runs_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "message_type": "voice"},
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                await asyncio.sleep(0.3)
                ev = await _collect_run_completed(cli, run_id)

        assert ev.get("event") == "run.completed"
        assert "audio_base64" in ev
        assert len(ev["audio_base64"]) > 0
        mock_tts.assert_called_once()

    @pytest.mark.asyncio
    async def test_voice_header_flag_calls_tts(self):
        """X-Hermes-Voice: 'true' header → TTS fires."""
        adapter = _make_adapter()
        fake_tts = json.dumps({"file_path": "/tmp/tts_hdr.mp3"})

        with (
            patch.object(adapter, "_create_agent", return_value=_fake_agent()),
            patch("tools.tts_tool.check_tts_requirements", return_value=True),
            patch("tools.tts_tool.text_to_speech_tool", return_value=fake_tts) as mock_tts,
        ):
            app = _create_runs_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"X-Hermes-Voice": "true"},
                )
                assert resp.status == 202
                await asyncio.sleep(0.3)

        mock_tts.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_voice_run_skips_tts_and_no_audio_path(self):
        """Normal run (no voice flag) → TTS not called, no audio_path in event."""
        adapter = _make_adapter()

        with (
            patch.object(adapter, "_create_agent", return_value=_fake_agent()),
            patch("tools.tts_tool.text_to_speech_tool") as mock_tts,
        ):
            app = _create_runs_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                await asyncio.sleep(0.3)
                ev = await _collect_run_completed(cli, run_id)

        assert ev.get("event") == "run.completed"
        assert "audio_path" not in ev
        mock_tts.assert_not_called()

    @pytest.mark.asyncio
    async def test_false_voice_header_skips_tts(self):
        """X-Hermes-Voice: 'false' → not voice mode, TTS not called."""
        adapter = _make_adapter()

        with (
            patch.object(adapter, "_create_agent", return_value=_fake_agent()),
            patch("tools.tts_tool.text_to_speech_tool") as mock_tts,
        ):
            app = _create_runs_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"X-Hermes-Voice": "false"},
                )
                assert resp.status == 202
                await asyncio.sleep(0.2)

        mock_tts.assert_not_called()
