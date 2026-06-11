from types import SimpleNamespace

import pytest

from gateway.platforms.base import MessageEvent, MessageType
from gateway.video_analysis import VideoAnalysisLimits, analyze_video_local, format_video_analysis_context


def test_analyze_video_local_skips_oversized_file(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")

    result = analyze_video_local(str(video), VideoAnalysisLimits(max_bytes=1))

    assert result["status"] == "skipped"
    assert "too large" in result["reason"]


def test_format_video_analysis_context_includes_frames_and_transcript():
    context = format_video_analysis_context(
        {
            "status": "ok",
            "duration_seconds": 3.4,
            "outdir": "/tmp/analysis",
            "frames": ["/tmp/frame-1.jpg", "/tmp/frame-2.jpg"],
            "audio": "/tmp/audio.wav",
            "transcript": "привет",
        }
    )

    assert "Telegram video local analysis" in context
    assert "/tmp/frame-1.jpg" in context
    assert "transcript:" in context
    assert "привет" in context


@pytest.mark.asyncio
async def test_telegram_video_hook_is_opt_in(monkeypatch):
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._video_analysis_enabled = False
    adapter._video_analysis_limits = VideoAnalysisLimits()

    called = False

    def fake_analyze(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"status": "ok", "frames": []}

    monkeypatch.setattr("gateway.platforms.telegram.analyze_video_local", fake_analyze)
    event = MessageEvent(text="", message_type=MessageType.VIDEO, source=SimpleNamespace())

    await adapter._maybe_enrich_video_event(event, "/tmp/video.mp4")

    assert called is False
    assert event.text == ""


@pytest.mark.asyncio
async def test_telegram_video_hook_adds_context_and_frame_paths(monkeypatch):
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._video_analysis_enabled = True
    adapter._video_analysis_limits = VideoAnalysisLimits()

    def fake_analyze(*_args, **_kwargs):
        return {
            "status": "ok",
            "duration_seconds": 5.0,
            "outdir": "/tmp/video-analysis",
            "frames": ["/tmp/frame-1.jpg", "/tmp/frame-2.jpg"],
            "audio": "/tmp/audio.wav",
            "transcript": "video transcript",
        }

    monkeypatch.setattr("gateway.platforms.telegram.analyze_video_local", fake_analyze)
    event = MessageEvent(
        text="caption",
        message_type=MessageType.VIDEO,
        source=SimpleNamespace(),
        media_urls=["/tmp/video.mp4"],
        media_types=["video/mp4"],
    )

    await adapter._maybe_enrich_video_event(event, "/tmp/video.mp4")

    assert "caption" in event.text
    assert "video transcript" in event.text
    assert event.media_urls == ["/tmp/video.mp4", "/tmp/frame-1.jpg", "/tmp/frame-2.jpg"]
    assert event.media_types == ["video/mp4", "image/jpeg", "image/jpeg"]
