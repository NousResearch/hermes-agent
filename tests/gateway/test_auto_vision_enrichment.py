import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import tools.vision_tools as vision_tools_module
from gateway.config import Platform
from gateway.media_pipeline import MessageAttachment
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


async def _wait_until(predicate, *, timeout: float = 0.2, interval: float = 0.005):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if predicate():
            return
        await asyncio.sleep(interval)
    assert predicate()


def test_auto_vision_timeout_uses_aux_timeout_with_cap(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()

    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"auxiliary": {"vision": {"timeout": 120}}},
    )

    assert runner._auto_vision_analysis_timeout_seconds() == 45.0


def test_auto_vision_inline_wait_uses_longer_budget_for_image_only_turns(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()

    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"auxiliary": {"vision": {"auto_timeout": 12, "image_only_inline_wait": 9}}},
    )

    source = SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1", chat_type="group")

    wait_seconds = runner._auto_vision_inline_wait_seconds(
        source,
        has_user_text=False,
    )

    assert wait_seconds == 9.0


def test_image_vision_inputs_prefer_remote_url_for_qq_media_when_vision_backend_needs_url(
    monkeypatch,
):
    event = MessageEvent(
        text="看图",
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1"),
        media_urls=["/tmp/cached-image.jpg"],
        media_sources=["https://cdn.example.com/image.jpg"],
        media_types=["image/jpeg"],
    )
    monkeypatch.setattr(
        "tools.vision_tools.should_prefer_remote_vision_source",
        lambda image_url, **_: image_url == "https://cdn.example.com/image.jpg",
    )

    assert GatewayRunner.__module__  # keep import visible for static analyzers
    from gateway.run import _image_vision_inputs_from_event

    assert _image_vision_inputs_from_event(event) == ["https://cdn.example.com/image.jpg"]


def test_image_vision_inputs_fall_back_to_local_cache_for_qq_media_without_remote_source(
    monkeypatch,
):
    event = MessageEvent(
        text="看图",
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1"),
        media_urls=["/tmp/cached-image.jpg"],
        media_sources=[""],
        media_types=["image/jpeg"],
    )
    monkeypatch.setattr(
        "tools.vision_tools.should_prefer_remote_vision_source",
        lambda *_args, **_kwargs: True,
    )

    from gateway.run import _image_vision_inputs_from_event

    assert _image_vision_inputs_from_event(event) == ["/tmp/cached-image.jpg"]


def test_image_vision_inputs_fall_back_to_local_cache_for_qq_signed_remote_source(
    monkeypatch,
):
    event = MessageEvent(
        text="看图",
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1"),
        media_urls=["/tmp/cached-image.jpg"],
        media_sources=[
            "https://multimedia.nt.qq.com.cn/download?appid=1406&fileid=abc&rkey=def"
        ],
        media_types=["image/jpeg"],
    )
    monkeypatch.setattr(
        "tools.vision_tools.should_prefer_remote_vision_source",
        lambda *_args, **_kwargs: False,
    )

    from gateway.run import _image_vision_inputs_from_event

    assert _image_vision_inputs_from_event(event) == ["/tmp/cached-image.jpg"]


def test_image_vision_inputs_fall_back_to_local_cache_for_generic_custom_v1_backend(
    monkeypatch,
):
    event = MessageEvent(
        text="看图",
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1"),
        media_urls=["/tmp/cached-image.jpg"],
        media_sources=["https://cdn.example.com/image.jpg"],
        media_types=["image/jpeg"],
    )
    monkeypatch.setattr(
        "tools.vision_tools.should_prefer_remote_vision_source",
        lambda *_args, **_kwargs: False,
    )

    from gateway.run import _image_vision_inputs_from_event

    assert _image_vision_inputs_from_event(event) == ["/tmp/cached-image.jpg"]


def test_image_vision_inputs_prefer_preserved_media_source_for_non_qq():
    event = MessageEvent(
        text="看图",
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1"),
        media_urls=["/tmp/cached-image.jpg"],
        media_sources=["https://cdn.example.com/image.jpg"],
        media_types=["image/jpeg"],
    )

    from gateway.run import _image_vision_inputs_from_event

    assert _image_vision_inputs_from_event(event) == [
        "https://cdn.example.com/image.jpg"
    ]


def test_image_vision_inputs_skip_animated_gif_media():
    from gateway.run import _image_vision_inputs_from_event

    event = MessageEvent(
        text="看图",
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1"),
        media_urls=["/tmp/animated.gif", "/tmp/static.jpg"],
        media_sources=[
            "https://cdn.example.com/animated.gif",
            "https://cdn.example.com/static.jpg",
        ],
        media_types=["image/gif", "image/jpeg"],
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "tools.vision_tools.should_prefer_remote_vision_source",
            lambda image_url, **_: image_url == "https://cdn.example.com/static.jpg",
        )

        assert _image_vision_inputs_from_event(event) == [
            "https://cdn.example.com/static.jpg"
        ]


def test_image_vision_inputs_skip_gif_by_path_even_if_mime_is_wrong():
    event = MessageEvent(
        text="看图",
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1"),
        media_urls=["/tmp/animated.gif", "/tmp/static.jpg"],
        media_sources=["", ""],
        media_types=["image/jpeg", "image/jpeg"],
    )

    from gateway.run import _image_vision_inputs_from_event

    assert _image_vision_inputs_from_event(event) == ["/tmp/static.jpg"]


def test_auto_vision_degraded_note_redacts_remote_signed_url():
    runner = GatewayRunner.__new__(GatewayRunner)

    note = runner._auto_vision_degraded_note(
        "https://multimedia.nt.qq.com.cn/download?appid=1406&fileid=abc&rkey=def",
        pending=False,
    )

    assert "multimedia.nt.qq.com.cn" not in note
    assert note == "[Image attached; no verified image description is available for this turn.]"
    assert "vision_analyze" not in note


def test_auto_vision_degraded_note_for_local_cache_does_not_suggest_tool_loop():
    runner = GatewayRunner.__new__(GatewayRunner)

    note = runner._auto_vision_degraded_note("/tmp/demo-image.jpg", pending=False)

    assert note == "[Image attached; no verified image description is available for this turn.]"
    assert "vision_analyze" not in note
    assert "/tmp/demo-image.jpg" not in note


@pytest.mark.asyncio
async def test_foreground_turn_image_only_falls_back_to_degraded_note_when_enrichment_returns_empty(
    monkeypatch,
):
    from gateway.foreground_turn_runtime_service import prepare_gateway_foreground_message

    event = MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1", chat_type="group"),
        attachments=[
            MessageAttachment(
                kind="image",
                mime_type="image/png",
                local_path="/tmp/demo.png",
                analysis_ref="/tmp/demo.png",
            )
        ],
    )

    async def _fake_prelude(**kwargs):
        return SimpleNamespace(
            hook_ctx=kwargs.get("hook_ctx") or {},
            message_text=kwargs["message_text"],
            blocked=False,
        )

    monkeypatch.setattr(
        "gateway.foreground_turn_runtime_service.run_gateway_agent_prelude",
        _fake_prelude,
    )

    prepared = await prepare_gateway_foreground_message(
        event=event,
        source=event.source,
        session_id="sess-1",
        history=[],
        thread_sessions_per_user=False,
        hooks=SimpleNamespace(),
        adapters={},
        image_vision_inputs_from_event=lambda _event: ["/tmp/demo.png"],
        enrich_message_with_vision=AsyncMock(return_value=""),
        auto_vision_degraded_note=lambda _path, pending: (
            "[Image attached; no verified image description is available yet.]"
            if pending
            else "[Image attached; no verified image description is available for this turn.]"
        ),
        enrich_message_with_transcription=AsyncMock(side_effect=lambda text, _paths: text),
        has_setup_skill=lambda: False,
        expand_context_references=AsyncMock(return_value=""),
    )

    assert (
        prepared.message_text
        == "[Image attached; no verified image description is available for this turn.]"
    )


@pytest.mark.asyncio
async def test_auto_vision_enrichment_times_out_and_degrades_gracefully(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()

    async def _slow_vision_analyze_tool(*, image_url, user_prompt, model=None):
        await asyncio.sleep(0.05)
        return (
            '{"success": true, "analysis": "should not arrive before timeout"}'
        )

    monkeypatch.setattr(
        "tools.vision_tools.vision_analyze_tool",
        _slow_vision_analyze_tool,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"auxiliary": {"vision": {"auto_timeout": 0.01}}},
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_INLINE_WAIT_SECONDS",
        0.01,
        raising=False,
    )

    result = await asyncio.wait_for(
        runner._enrich_message_with_vision(
            "看看这个图",
            ["/tmp/demo-image.jpg"],
        ),
        timeout=0.1,
    )

    assert "[Image attached; no verified image description is available" in result
    assert "vision_analyze" not in result
    assert "/tmp/demo-image.jpg" not in result
    assert result.endswith("看看这个图")


@pytest.mark.asyncio
async def test_auto_vision_enrichment_skips_retries_during_timeout_cooldown(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()
    runner._auto_vision_unhealthy_until = 0.0
    runner._auto_vision_unhealthy_reason = ""

    calls = {"count": 0}

    async def _slow_vision_analyze_tool(*, image_url, user_prompt, model=None):
        calls["count"] += 1
        await asyncio.sleep(0.05)
        return (
            '{"success": true, "analysis": "should not arrive before timeout"}'
        )

    monkeypatch.setattr(
        "tools.vision_tools.vision_analyze_tool",
        _slow_vision_analyze_tool,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"auxiliary": {"vision": {"auto_timeout": 0.01}}},
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_FAILURE_COOLDOWN_SECONDS",
        60.0,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_INLINE_WAIT_SECONDS",
        0.01,
        raising=False,
    )

    first = await asyncio.wait_for(
        runner._enrich_message_with_vision(
            "第一张图",
            ["/tmp/first.jpg"],
        ),
        timeout=0.1,
    )
    await _wait_until(
        lambda: calls["count"] == 1
        and bool(getattr(runner, "_auto_vision_unhealthy_until", 0.0))
    )

    started = time.monotonic()
    second = await asyncio.wait_for(
        runner._enrich_message_with_vision(
            "第二张图",
            ["/tmp/second.jpg"],
        ),
        timeout=0.1,
    )
    elapsed = time.monotonic() - started

    assert calls["count"] == 1
    assert "[Image attached; no verified image description is available" in first
    assert "[Image attached; no verified image description is available for this turn.]" in second
    assert "vision_analyze" not in second
    assert "/tmp/second.jpg" not in second
    assert second.endswith("第二张图")
    assert elapsed < 0.05


@pytest.mark.asyncio
async def test_auto_vision_timeout_cooldown_expires_quickly_and_allows_retry(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()
    runner._background_tasks = set()
    runner._auto_vision_unhealthy_until = 0.0
    runner._auto_vision_unhealthy_reason = ""
    runner._auto_vision_cache = {}
    runner._auto_vision_tasks = {}

    calls = {"count": 0}

    async def _timeout_then_success(*, image_url, user_prompt, model=None):
        calls["count"] += 1
        if calls["count"] == 1:
            await asyncio.sleep(0.05)
            return '{"success": true, "analysis": "late timeout result"}'
        return '{"success": true, "analysis": "第二次重试成功"}'

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _timeout_then_success)
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_TRANSIENT_COOLDOWN_SECONDS",
        1.0,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"auxiliary": {"vision": {"auto_timeout": 0.01}}},
    )

    fake_now = {"value": 100.0}
    monkeypatch.setattr("gateway.run.time.time", lambda: fake_now["value"])

    first = await asyncio.wait_for(
        runner._enrich_message_with_vision("第一张", ["/tmp/first.jpg"]),
        timeout=0.1,
    )
    await _wait_until(
        lambda: calls["count"] == 1
        and bool(getattr(runner, "_auto_vision_unhealthy_until", 0.0))
    )
    assert "[Image attached; no verified image description is available" in first

    fake_now["value"] += 1.5
    second = await asyncio.wait_for(
        runner._enrich_message_with_vision("第二张", ["/tmp/second.jpg"]),
        timeout=0.1,
    )

    assert calls["count"] == 2
    assert "第二次重试成功" in second


@pytest.mark.asyncio
async def test_auto_vision_timeout_same_image_retries_after_cooldown(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()
    runner._background_tasks = set()
    runner._auto_vision_unhealthy_until = 0.0
    runner._auto_vision_unhealthy_reason = ""
    runner._auto_vision_cache = {}
    runner._auto_vision_tasks = {}

    calls = {"count": 0}

    async def _timeout_then_success(*, image_url, user_prompt, model=None):
        calls["count"] += 1
        if calls["count"] == 1:
            await asyncio.sleep(0.05)
            return '{"success": true, "analysis": "late timeout result"}'
        return '{"success": true, "analysis": "同一张图第二次成功"}'

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _timeout_then_success)
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_TRANSIENT_COOLDOWN_SECONDS",
        1.0,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_INLINE_WAIT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"auxiliary": {"vision": {"auto_timeout": 0.01}}},
    )

    fake_now = {"value": 100.0}
    monkeypatch.setattr("gateway.run.time.time", lambda: fake_now["value"])
    source = SimpleNamespace(platform=Platform.QQ_NAPCAT, chat_type="group")

    first = await asyncio.wait_for(
        runner._enrich_message_with_vision("第一张", ["/tmp/retry.jpg"], source=source),
        timeout=0.1,
    )
    await _wait_until(
        lambda: calls["count"] == 1
        and bool(getattr(runner, "_auto_vision_unhealthy_until", 0.0))
        and not bool(getattr(runner, "_auto_vision_tasks", {}))
    )
    assert "[Image attached; no verified image description is available yet.]" in first
    assert calls["count"] == 1

    fake_now["value"] += 1.5
    second = await asyncio.wait_for(
        runner._enrich_message_with_vision("第二张", ["/tmp/retry.jpg"], source=source),
        timeout=0.1,
    )

    assert calls["count"] == 2
    assert "同一张图第二次成功" in second


@pytest.mark.asyncio
async def test_auto_vision_provider_error_keeps_long_cooldown(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()
    runner._background_tasks = set()
    runner._auto_vision_unhealthy_until = 0.0
    runner._auto_vision_unhealthy_reason = ""
    runner._auto_vision_cache = {}
    runner._auto_vision_tasks = {}

    calls = {"count": 0}

    async def _provider_error(*, image_url, user_prompt, model=None):
        calls["count"] += 1
        return '{"success": false, "error": "payment required by upstream", "analysis": "payment required by upstream"}'

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _provider_error)
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_FAILURE_COOLDOWN_SECONDS",
        60.0,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_INLINE_WAIT_SECONDS",
        0.02,
        raising=False,
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    first = await runner._enrich_message_with_vision(
        "第一张",
        ["/tmp/first.jpg"],
        source=SimpleNamespace(platform=Platform.QQ_NAPCAT, chat_type="group"),
    )
    second = await runner._enrich_message_with_vision(
        "第二张",
        ["/tmp/second.jpg"],
        source=SimpleNamespace(platform=Platform.QQ_NAPCAT, chat_type="group"),
    )

    assert "[Image attached; no verified image description is available for this turn.]" in first
    assert "[Image attached; no verified image description is available for this turn.]" in second
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_auto_vision_enrichment_returns_immediately_while_background_cache_warms(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()
    runner._background_tasks = set()
    runner._auto_vision_unhealthy_until = 0.0
    runner._auto_vision_unhealthy_reason = ""
    runner._auto_vision_cache = {}
    runner._auto_vision_tasks = {}

    calls = {"count": 0}

    async def _slow_success(*, image_url, user_prompt, model=None):
        calls["count"] += 1
        await asyncio.sleep(0.05)
        return '{"success": true, "analysis": "屏幕截图里有一行测试文字"}'

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _slow_success)
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS",
        0.5,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_INLINE_WAIT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    source = SimpleNamespace(platform=Platform.QQ_NAPCAT, chat_type="group")
    first = await asyncio.wait_for(
        runner._enrich_message_with_vision(
            "看看这图",
            ["/tmp/demo-image.jpg"],
            source=source,
        ),
        timeout=0.1,
    )

    assert "[Image attached; no verified image description is available yet.]" in first
    assert first.endswith("看看这图")
    await _wait_until(lambda: calls["count"] == 1)

    await asyncio.sleep(0.06)

    second = await runner._enrich_message_with_vision(
        "再看一次",
        ["/tmp/demo-image.jpg"],
        source=source,
    )

    assert "屏幕截图里有一行测试文字" in second
    assert "vision_analyze" not in second
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_auto_vision_image_only_group_waits_for_result(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()
    runner._background_tasks = set()
    runner._auto_vision_unhealthy_until = 0.0
    runner._auto_vision_unhealthy_reason = ""
    runner._auto_vision_cache = {}
    runner._auto_vision_tasks = {}

    async def _slower_success(*, image_url, user_prompt, model=None):
        await asyncio.sleep(0.05)
        return '{"success": true, "analysis": "图里是一张带文字的界面截图"}'

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _slower_success)
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS",
        0.5,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_INLINE_WAIT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_GROUP_INLINE_WAIT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_IMAGE_ONLY_INLINE_WAIT_SECONDS",
        0.2,
        raising=False,
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    source = SimpleNamespace(platform=Platform.QQ_NAPCAT, chat_type="group")
    result = await asyncio.wait_for(
        runner._enrich_message_with_vision(
            "",
            ["/tmp/demo-image.jpg"],
            source=source,
        ),
        timeout=0.12,
    )

    assert "图里是一张带文字的界面截图" in result
    assert "[Image attached; no verified image description is available yet.]" not in result


@pytest.mark.asyncio
async def test_auto_vision_failed_background_analysis_enters_cooldown(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()
    runner._background_tasks = set()
    runner._auto_vision_unhealthy_until = 0.0
    runner._auto_vision_unhealthy_reason = ""
    runner._auto_vision_cache = {}
    runner._auto_vision_tasks = {}

    calls = {"count": 0}

    async def _slow_failure(*, image_url, user_prompt, model=None):
        calls["count"] += 1
        await asyncio.sleep(0.02)
        return (
            '{"success": false, "error": "Vision model returned empty content after retry", '
            '"analysis": "The vision model returned an empty response twice."}'
        )

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _slow_failure)
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS",
        0.5,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_INLINE_WAIT_SECONDS",
        0.005,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_FAILURE_COOLDOWN_SECONDS",
        60.0,
        raising=False,
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    source = SimpleNamespace(platform=Platform.QQ_NAPCAT, chat_type="group")
    first = await asyncio.wait_for(
        runner._enrich_message_with_vision(
            "第一张",
            ["/tmp/first.jpg"],
            source=source,
        ),
        timeout=0.1,
    )
    assert "[Image attached; no verified image description is available yet.]" in first

    await _wait_until(
        lambda: calls["count"] == 1
        and bool(getattr(runner, "_auto_vision_unhealthy_until", 0.0))
    )

    second = await asyncio.wait_for(
        runner._enrich_message_with_vision(
            "第二张",
            ["/tmp/second.jpg"],
            source=source,
        ),
        timeout=0.1,
    )

    assert "[Image attached; no verified image description is available for this turn.]" in second
    assert "vision_analyze" not in second
    assert "/tmp/second.jpg" not in second
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_auto_vision_low_value_sticker_media_fails_fast_without_llm(
    monkeypatch, tmp_path
):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()
    runner._background_tasks = set()
    runner._auto_vision_unhealthy_until = 0.0
    runner._auto_vision_unhealthy_reason = ""
    runner._auto_vision_cache = {}
    runner._auto_vision_tasks = {}

    sticker = tmp_path / "qq-sticker.webp"
    sticker.write_bytes(b"RIFF\x18\x00\x00\x00WEBPVP8 " + b"\x00" * 16)

    monkeypatch.setattr(vision_tools_module, "_RECENT_VISION_FAILURES", {})
    monkeypatch.setattr(vision_tools_module, "_VISION_PROVIDER_UNHEALTHY", {})
    mock_llm = AsyncMock()
    monkeypatch.setattr("tools.vision_tools.async_call_llm", mock_llm)
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS",
        0.5,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_INLINE_WAIT_SECONDS",
        0.02,
        raising=False,
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    result = await runner._enrich_message_with_vision(
        "看看这个表情",
        [str(sticker)],
        source=SimpleNamespace(platform=Platform.QQ_NAPCAT, chat_type="group"),
    )

    assert "vision_analyze" not in result
    assert str(sticker) not in result
    assert result.endswith("看看这个表情")
    mock_llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_vision_tool_provider_cooldown_avoids_second_llm_call(
    monkeypatch, tmp_path
):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()
    runner._background_tasks = set()
    runner._auto_vision_unhealthy_until = 0.0
    runner._auto_vision_unhealthy_reason = ""
    runner._auto_vision_cache = {}
    runner._auto_vision_tasks = {}

    first_image = tmp_path / "first.png"
    first_image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    second_image = tmp_path / "second.png"
    second_image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

    monkeypatch.setattr(vision_tools_module, "_RECENT_VISION_FAILURES", {})
    monkeypatch.setattr(vision_tools_module, "_VISION_PROVIDER_UNHEALTHY", {})
    mock_llm = AsyncMock(side_effect=RuntimeError("payment required by upstream"))
    monkeypatch.setattr("tools.vision_tools.async_call_llm", mock_llm)
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS",
        0.5,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_INLINE_WAIT_SECONDS",
        0.02,
        raising=False,
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    first = await runner._enrich_message_with_vision(
        "第一张",
        [str(first_image)],
        source=SimpleNamespace(platform=Platform.QQ_NAPCAT, chat_type="group"),
    )
    second = await runner._enrich_message_with_vision(
        "第二张",
        [str(second_image)],
        source=SimpleNamespace(platform=Platform.QQ_NAPCAT, chat_type="group"),
    )

    assert "vision_analyze" not in first
    assert "vision_analyze" not in second
    assert second.endswith("第二张")
    assert mock_llm.await_count == 1


@pytest.mark.asyncio
async def test_auto_vision_multi_image_group_path_uses_shared_inline_budget(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()
    runner._background_tasks = set()
    runner._auto_vision_unhealthy_until = 0.0
    runner._auto_vision_unhealthy_reason = ""
    runner._auto_vision_cache = {}
    runner._auto_vision_tasks = {}

    calls = {"count": 0}

    async def _slow_success(*, image_url, user_prompt, model=None):
        calls["count"] += 1
        await asyncio.sleep(0.05)
        return '{"success": true, "analysis": "late image analysis"}'

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _slow_success)
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS",
        0.5,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_INLINE_WAIT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_GROUP_INLINE_WAIT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    source = SimpleNamespace(platform=Platform.QQ_NAPCAT, chat_type="group")
    started = time.monotonic()
    result = await asyncio.wait_for(
        runner._enrich_message_with_vision(
            "三张图一起看",
            ["/tmp/1.jpg", "/tmp/2.jpg", "/tmp/3.jpg"],
            source=source,
        ),
        timeout=0.12,
    )
    elapsed = time.monotonic() - started
    await _wait_until(lambda: calls["count"] == 3)

    assert calls["count"] == 3
    assert result.count("[Image attached; no verified image description is available yet.]") == 3
    assert elapsed < 0.08


@pytest.mark.asyncio
async def test_auto_vision_skips_new_warmup_when_inflight_limit_reached(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace()
    runner._background_tasks = set()
    runner._auto_vision_unhealthy_until = 0.0
    runner._auto_vision_unhealthy_reason = ""
    runner._auto_vision_cache = {}
    runner._auto_vision_tasks = {}

    pending = asyncio.get_running_loop().create_future()
    runner._auto_vision_tasks["busy-key"] = pending

    calls = {"count": 0}

    async def _vision_should_not_run(*, image_url, user_prompt, model=None):
        calls["count"] += 1
        return '{"success": true, "analysis": "unexpected"}'

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _vision_should_not_run)
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_MAX_INFLIGHT_TASKS",
        1,
        raising=False,
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    result = await runner._enrich_message_with_vision(
        "看看这张",
        ["/tmp/limit.jpg"],
        source=SimpleNamespace(platform=Platform.QQ_NAPCAT, chat_type="group"),
    )

    assert "[Image attached; no verified image description is available for this turn.]" in result
    assert calls["count"] == 0
    pending.cancel()


def test_auto_vision_prunes_expired_and_old_cache_entries(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    runner._auto_vision_cache = {
        "expired": {
            "status": "error",
            "updated_at": 10.0,
            "expires_at": 20.0,
        },
        "oldest": {
            "status": "success",
            "analysis": "a",
            "updated_at": 30.0,
            "expires_at": 130.0,
        },
        "middle": {
            "status": "success",
            "analysis": "b",
            "updated_at": 40.0,
            "expires_at": 140.0,
        },
        "newest": {
            "status": "success",
            "analysis": "c",
            "updated_at": 50.0,
            "expires_at": 150.0,
        },
    }
    runner._auto_vision_tasks = {}

    monkeypatch.setattr("gateway.run.time.time", lambda: 100.0)
    monkeypatch.setattr(
        "gateway.run._AUTO_VISION_MAX_CACHE_ENTRIES",
        2,
        raising=False,
    )

    runner._prune_auto_vision_state()

    assert "expired" not in runner._auto_vision_cache
    assert "oldest" not in runner._auto_vision_cache
    assert set(runner._auto_vision_cache) == {"middle", "newest"}


@pytest.mark.asyncio
async def test_shared_thread_image_only_turn_uses_shared_auto_vision_enrichment():
    """Foreground prep should enrich image-only turns via auto-vision path."""
    from gateway.foreground_turn_runtime_service import prepare_gateway_foreground_message
    from gateway.run import _image_vision_inputs_from_event

    enrich = AsyncMock(
        return_value="[The user sent an image~ Here's what I can see:\n测试图片描述]"
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=SessionSource(
            platform=Platform.QQ_NAPCAT,
            chat_id="group-1",
            chat_type="group",
            thread_id="thread-1",
            user_name="Alice",
        ),
        attachments=[
            MessageAttachment(
                kind="image",
                mime_type="image/jpeg",
                local_path="/tmp/demo.jpg",
                analysis_ref="/tmp/demo.jpg",
            )
        ],
    )

    prepared = await prepare_gateway_foreground_message(
        event=event,
        source=event.source,
        session_id="sid-1",
        history=[],
        thread_sessions_per_user=False,
        hooks=SimpleNamespace(emit=AsyncMock()),
        adapters={},
        image_vision_inputs_from_event=_image_vision_inputs_from_event,
        enrich_message_with_vision=enrich,
        auto_vision_degraded_note=lambda _path, pending: "degraded",
        enrich_message_with_transcription=AsyncMock(side_effect=lambda text, paths: text),
        has_setup_skill=lambda: False,
        expand_context_references=AsyncMock(side_effect=lambda text: text),
    )

    enrich.assert_awaited_once()
    assert prepared.message_text == (
        "[Alice] [The user sent an image~ Here's what I can see:\n测试图片描述]"
    )
