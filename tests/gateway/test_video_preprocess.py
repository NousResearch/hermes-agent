"""Inbound video attachments: aux auto-enrichment per ``agent.video_input_mode``.

Mirrors tests/gateway/test_vision_preprocess.py for the video path.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest


def _make_runner():
    from gateway.run import GatewayRunner

    return GatewayRunner.__new__(GatewayRunner)


def _patch_load_config(monkeypatch, cfg):
    from hermes_cli import config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)


@pytest.mark.asyncio
async def test_video_enrichment_off_by_default(monkeypatch):
    """Default install: no auxiliary.video configured → mode 'off' → path note only."""
    _patch_load_config(monkeypatch, {"agent": {}, "auxiliary": {}})

    with patch(
        "tools.vision_tools.video_analyze_tool",
        new_callable=AsyncMock,
    ) as mock_video:
        result = await _make_runner()._enrich_inbound_videos(
            source=None, session_key="s1", message_text="look at this", video_paths=["/tmp/v.mp4"],
        )

    mock_video.assert_not_awaited()
    assert result == "look at this"


@pytest.mark.asyncio
async def test_video_enrichment_auto_with_aux_video_config(monkeypatch):
    """auxiliary.video.provider set + mode auto → auto-analyze and prepend description."""
    _patch_load_config(monkeypatch, {
        "agent": {},
        "auxiliary": {"video": {"provider": "custom", "base_url": "http://127.0.0.1:18081/v1",
                                "model": "GLM-5.3-Flash-512K"}},
    })

    with patch(
        "tools.vision_tools.video_analyze_tool",
        new_callable=AsyncMock,
        return_value=json.dumps({"success": True, "analysis": "A ball sweeps left to right."}),
    ) as mock_video:
        result = await _make_runner()._enrich_inbound_videos(
            source=None, session_key="s2", message_text="", video_paths=["/tmp/v.mp4"],
        )

    mock_video.assert_awaited_once()
    assert "A ball sweeps left to right." in result
    assert "video_analyze with" in result and "/tmp/v.mp4" in result


@pytest.mark.asyncio
async def test_video_enrichment_text_mode_always_analyzes(monkeypatch):
    """video_input_mode: text → auto-analyze even without auxiliary.video config."""
    _patch_load_config(monkeypatch, {"agent": {"video_input_mode": "text"}, "auxiliary": {}})

    with patch(
        "tools.vision_tools.video_analyze_tool",
        new_callable=AsyncMock,
        return_value=json.dumps({"success": True, "analysis": "Countdown 5 4 3 2 1."}),
    ):
        result = await _make_runner()._enrich_inbound_videos(
            source=None, session_key="s3", message_text="what do you see", video_paths=["/tmp/v.mp4"],
        )

    assert "Countdown 5 4 3 2 1." in result
    assert "what do you see" in result


@pytest.mark.asyncio
async def test_video_enrichment_off_mode_explicit(monkeypatch):
    """video_input_mode: off → never auto-analyze, even with auxiliary.video set."""
    _patch_load_config(monkeypatch, {"agent": {"video_input_mode": "off"},
                                     "auxiliary": {"video": {"provider": "custom"}}})

    with patch(
        "tools.vision_tools.video_analyze_tool",
        new_callable=AsyncMock,
    ) as mock_video:
        result = await _make_runner()._enrich_inbound_videos(
            source=None, session_key="s4", message_text="hello", video_paths=["/tmp/v.mp4"],
        )

    mock_video.assert_not_awaited()
    assert result == "hello"


@pytest.mark.asyncio
async def test_video_enrichment_tool_failure_leaves_retry_note(monkeypatch):
    """Analysis failure → neutral retry note with the path, never a raised exception."""
    _patch_load_config(monkeypatch, {"agent": {"video_input_mode": "text"}, "auxiliary": {}})

    with patch(
        "tools.vision_tools.video_analyze_tool",
        new_callable=AsyncMock,
        return_value=json.dumps({"success": False, "analysis": "backend down"}),
    ):
        result = await _make_runner()._enrich_inbound_videos(
            source=None, session_key="s5", message_text="look", video_paths=["/tmp/v.mp4"],
        )

    assert "couldn't analyze it" in result
    assert "/tmp/v.mp4" in result
    assert "video_analyze" in result


@pytest.mark.asyncio
async def test_video_enrichment_prompt_covers_temporal_content():
    """The auto-analysis prompt asks for chronology and brief/flashing text."""
    from gateway.run_inbound import VIDEO_ANALYSIS_PROMPT

    lowered = VIDEO_ANALYSIS_PROMPT.lower()
    assert "chronological" in lowered
    assert "flashing" in lowered
    assert "roughly when" in lowered


@pytest.mark.asyncio
async def test_video_enrichment_video_exception_is_contained(monkeypatch):
    """A raised exception from the tool is caught per-video, not propagated."""
    _patch_load_config(monkeypatch, {"agent": {"video_input_mode": "text"}, "auxiliary": {}})

    with patch(
        "tools.vision_tools.video_analyze_tool",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        result = await _make_runner()._enrich_inbound_videos(
            source=None, session_key="s6", message_text="look", video_paths=["/tmp/v.mp4"],
        )

    assert "something went wrong" in result
    assert "/tmp/v.mp4" in result
