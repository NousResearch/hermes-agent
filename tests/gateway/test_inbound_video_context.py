import asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.WEIXIN: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "gpt-5.5"
    runner._base_url = None
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.WEIXIN,
        chat_id="wx-user",
        chat_type="private",
        user_name="wx-user",
    )


def test_inbound_video_injects_agent_visible_analysis_context():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="",
        message_type=MessageType.VIDEO,
        source=source,
        media_urls=["/tmp/doc_abc123_video.mp4"],
        media_types=["video/mp4"],
    )

    result = asyncio.run(
        runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )
    )

    assert result is not None
    assert "The user sent a video" in result
    assert "video.mp4" in result
    assert "/tmp/doc_abc123_video.mp4" in result
    assert "video_analyze" in result
    assert "summarize" in result


def test_inbound_document_video_file_injects_analysis_context_when_mime_is_generic():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="",
        message_type=MessageType.DOCUMENT,
        source=source,
        media_urls=["/tmp/doc_abc123_wechat-upload.mp4"],
        media_types=["application/octet-stream"],
    )

    result = asyncio.run(
        runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )
    )

    assert result is not None
    assert "The user sent a video" in result
    assert "wechat-upload.mp4" in result
    assert "/tmp/doc_abc123_wechat-upload.mp4" in result
    assert "Ask the user what they'd like" not in result


def test_douyin_link_injects_short_video_analysis_context():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="https://v.douyin.com/AbCdEf/ 帮我总结这个视频",
        message_type=MessageType.TEXT,
        source=source,
    )

    result = asyncio.run(
        runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )
    )

    assert result is not None
    assert "short-video link" in result
    assert "https://v.douyin.com/AbCdEf/" in result
    assert "browser_navigate" in result
    assert "browser_vision" in result
    assert "subtitle-aware visual analysis" in result
    assert "not fixed time intervals" in result
    assert "Capture each distinct subtitle line" in result
    assert "Do not make a high-frequency screen recording" in result
    assert "untrusted data" in result
    assert "Do not bypass" in result


def test_regular_link_does_not_inject_short_video_analysis_context():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="https://example.com/article please summarize",
        message_type=MessageType.TEXT,
        source=source,
    )

    result = asyncio.run(
        runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )
    )

    assert result == "https://example.com/article please summarize"


def test_lookalike_short_video_domain_does_not_inject_context():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="https://tiktok.com.evil.example/video please summarize",
        message_type=MessageType.TEXT,
        source=source,
    )

    result = asyncio.run(
        runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )
    )

    assert result == "https://tiktok.com.evil.example/video please summarize"
