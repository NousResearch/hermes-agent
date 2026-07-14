"""Tests for voice mode in POST /v1/runs.

Covers:
- message_type='voice' in body → _voice_mode detected, TTS called
- X-Hermes-Voice: 'true' header → _voice_mode detected
- non-voice runs → TTS not called, no audio_path in event
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_voice_flag_via_body_calls_tts(aiohttp_client, api_server_app):
    """message_type='voice' in body → TTS fires, audio_path in run.completed event."""
    fake_audio = "/tmp/tts_fake.mp3"
    fake_tts_result = json.dumps({"file_path": fake_audio, "text": "hi"})

    with (
        patch("tools.tts_tool.check_tts_requirements", return_value=True),
        patch("tools.tts_tool.text_to_speech_tool", return_value=fake_tts_result) as mock_tts,
    ):
        client = await aiohttp_client(api_server_app)
        resp = await client.post(
            "/v1/runs",
            json={"input": "hello forge", "message_type": "voice"},
            headers={"Authorization": "Bearer testkey"},
        )
        assert resp.status == 202
        body = await resp.json()
        run_id = body["run_id"]

        # Poll events
        events = []
        async with client.get(f"/v1/runs/{run_id}/events") as sse:
            async for line in sse.content:
                line = line.decode().strip()
                if line.startswith("data:"):
                    ev = json.loads(line[5:])
                    events.append(ev)
                    if ev.get("event") == "run.completed":
                        break

    completed = next((e for e in events if e.get("event") == "run.completed"), None)
    assert completed is not None
    assert "audio_path" in completed
    assert completed["audio_path"] == fake_audio
    mock_tts.assert_called_once()


@pytest.mark.asyncio
async def test_voice_flag_via_header_calls_tts(aiohttp_client, api_server_app):
    """X-Hermes-Voice: 'true' header → TTS fires."""
    fake_tts_result = json.dumps({"file_path": "/tmp/tts_hdr.mp3"})

    with (
        patch("tools.tts_tool.check_tts_requirements", return_value=True),
        patch("tools.tts_tool.text_to_speech_tool", return_value=fake_tts_result) as mock_tts,
    ):
        client = await aiohttp_client(api_server_app)
        resp = await client.post(
            "/v1/runs",
            json={"input": "hello"},
            headers={"Authorization": "Bearer testkey", "X-Hermes-Voice": "true"},
        )
        assert resp.status == 202

    mock_tts.assert_called_once()


@pytest.mark.asyncio
async def test_non_voice_run_skips_tts(aiohttp_client, api_server_app):
    """Normal run (no voice flag) → TTS not called, no audio_path."""
    with patch("tools.tts_tool.text_to_speech_tool") as mock_tts:
        client = await aiohttp_client(api_server_app)
        resp = await client.post(
            "/v1/runs",
            json={"input": "hello"},
            headers={"Authorization": "Bearer testkey"},
        )
        assert resp.status == 202
        body = await resp.json()
        run_id = body["run_id"]

        events = []
        async with client.get(f"/v1/runs/{run_id}/events") as sse:
            async for line in sse.content:
                line = line.decode().strip()
                if line.startswith("data:"):
                    ev = json.loads(line[5:])
                    events.append(ev)
                    if ev.get("event") == "run.completed":
                        break

    completed = next((e for e in events if e.get("event") == "run.completed"), None)
    assert completed is not None
    assert "audio_path" not in completed
    mock_tts.assert_not_called()
