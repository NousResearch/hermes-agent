import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _FatalAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="token"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._set_fatal_error(
            "telegram_token_lock",
            "Another local Hermes gateway is already using this Telegram bot token.",
            retryable=False,
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _RuntimeRetryableAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="token"), Platform.WHATSAPP)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _ReplacementDeliveryAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="token", typing_indicator=False),
            Platform.DISCORD,
        )
        self.sent: list[str] = []
        self.image_batches: list[list[str]] = []
        self.voices: list[str] = []
        self.videos: list[str] = []
        self.documents: list[str] = []
        self.native_attempts: list[tuple[str, str]] = []
        self.fail_kinds: set[str] = set()
        self.connected = True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self.connected = False

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if not self.connected:
            return SendResult(success=False, error="Not connected")
        self.sent.append(content)
        return SendResult(success=True, message_id=f"m-{len(self.sent)}")

    async def _record_native_send(
        self, kind: str, path: str, delivered: list[str]
    ) -> SendResult:
        self.native_attempts.append((kind, path))
        if not self.connected:
            return SendResult(success=False, error="Not connected")
        if kind in self.fail_kinds:
            return SendResult(success=False, error=f"{kind} failed")
        delivered.append(path)
        return SendResult(success=True, message_id=f"{kind}-{len(delivered)}")

    async def send_voice(
        self,
        chat_id,
        audio_path,
        caption=None,
        reply_to=None,
        metadata=None,
        **_kwargs,
    ):
        return await self._record_native_send("voice", audio_path, self.voices)

    async def send_video(
        self,
        chat_id,
        video_path,
        caption=None,
        reply_to=None,
        metadata=None,
        **_kwargs,
    ):
        return await self._record_native_send("video", video_path, self.videos)

    async def send_document(
        self,
        chat_id,
        file_path,
        caption=None,
        file_name=None,
        reply_to=None,
        metadata=None,
        **_kwargs,
    ):
        return await self._record_native_send(
            "document", file_path, self.documents
        )

    async def send_multiple_images(
        self, chat_id, images, metadata=None, human_delay=0.0
    ):
        paths = [url for url, _caption in images]
        self.native_attempts.extend(("image", path) for path in paths)
        if not self.connected:
            raise RuntimeError("Not connected")
        if "image" in self.fail_kinds:
            raise RuntimeError("image batch failed")
        self.image_batches.append(paths)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _PerImageReplacementDeliveryAdapter(_ReplacementDeliveryAdapter):
    """Replacement adapter that uses the base per-image batch loop."""

    send_multiple_images = BasePlatformAdapter.send_multiple_images

    def __init__(self):
        super().__init__()
        self.sent_images: list[str] = []

    async def send_image(
        self,
        chat_id,
        image_url,
        caption=None,
        reply_to=None,
        metadata=None,
    ):
        if not self.connected:
            return SendResult(success=False, error="Not connected")
        self.sent_images.append(image_url)
        return SendResult(success=True, message_id=f"image-{len(self.sent_images)}")


@pytest.mark.asyncio
async def test_runner_requests_clean_exit_for_nonretryable_startup_conflict(monkeypatch, tmp_path):
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: _FatalAdapter())

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is True
    assert "already using this Telegram bot token" in runner.exit_reason


@pytest.mark.asyncio
async def test_runner_queues_retryable_runtime_fatal_for_reconnection(monkeypatch, tmp_path):
    """Retryable runtime fatal errors queue the platform for reconnection
    AND keep the gateway alive — the background reconnect watcher recovers
    the platform when the underlying issue clears.  (Previously this
    exited-with-failure to trigger a systemd restart; that converted
    transient failures into infinite restart loops.)
    """
    config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error(
        "whatsapp_bridge_exited",
        "WhatsApp bridge process exited unexpectedly (code 1).",
        retryable=True,
    )

    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    await runner._handle_adapter_fatal_error(adapter)

    # Gateway stays alive — watcher will retry in background
    runner.stop.assert_not_awaited()
    assert runner._exit_with_failure is False
    assert Platform.WHATSAPP in runner._failed_platforms
    assert runner._failed_platforms[Platform.WHATSAPP]["attempts"] == 0


@pytest.mark.asyncio
async def test_retryable_fatal_queues_reconnect_after_cancellation_swallowing_disconnect(
    monkeypatch, tmp_path
):
    """A wedged old adapter cannot block runner-owned reconnect recovery."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error("transport_stale", "transport stale", retryable=True)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters

    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def swallow_cancellation():
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        finished.set()

    monkeypatch.setattr(adapter, "disconnect", swallow_cancellation)
    operation = asyncio.create_task(runner._handle_adapter_fatal_error(adapter))
    await started.wait()
    done, _pending = await asyncio.wait({operation}, timeout=0.2)
    try:
        assert operation in done
        assert runner.adapters == {}
        assert Platform.WHATSAPP in runner._failed_platforms
        assert runner._failed_platforms[Platform.WHATSAPP]["attempts"] == 0
    finally:
        release.set()
        await asyncio.wait({operation}, timeout=0.2)
        await asyncio.wait_for(finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_concurrent_fatal_notifications_disconnect_same_adapter_once(monkeypatch, tmp_path):
    """
    Two fatal-error notifications for the same still-installed adapter (e.g.
    from two concurrent recovery paths racing on the same underlying outage)
    must result in exactly one disconnect() call.

    Regression test for the TOCTOU race in _handle_adapter_fatal_error: the
    old code only removed the adapter from self.adapters in a `finally` block
    *after* awaiting disconnect(), so a second concurrent call could still see
    itself as "existing" and disconnect() the same object twice — the
    concrete origin of the "'NoneType' object has no attribute 'updater'"
    crash when the adapter's own teardown code re-reads self._app afterwards.
    """
    config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error(
        "whatsapp_bridge_exited",
        "WhatsApp bridge process exited unexpectedly (code 1).",
        retryable=True,
    )

    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    disconnect_calls = 0
    release_second_call = asyncio.Event()

    async def slow_disconnect():
        nonlocal disconnect_calls
        disconnect_calls += 1
        # Yield control so the second concurrent notification can run its
        # "existing is adapter" check before this call finishes tearing down.
        release_second_call.set()
        await asyncio.sleep(0)
        adapter._mark_disconnected()

    monkeypatch.setattr(adapter, "disconnect", slow_disconnect)

    await asyncio.gather(
        runner._handle_adapter_fatal_error(adapter),
        runner._handle_adapter_fatal_error(adapter),
    )

    assert disconnect_calls == 1


@pytest.mark.asyncio
async def test_stale_fatal_notification_from_superseded_adapter_is_ignored(monkeypatch, tmp_path):
    """
    A delayed fatal-error notification from an adapter instance that has
    since been replaced by a different, already-installed adapter (e.g. a
    background retry chain on the old instance finally giving up after a
    reconnect on a new instance already succeeded) must be ignored: it must
    not disconnect the new adapter, must not re-queue an already-healthy
    platform for reconnection, and must not shut the gateway down.
    """
    config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    old_adapter = _RuntimeRetryableAdapter()
    old_adapter._set_fatal_error(
        "whatsapp_bridge_exited",
        "stale failure from a superseded adapter instance",
        retryable=True,
    )

    new_adapter = _RuntimeRetryableAdapter()
    new_adapter.disconnect = AsyncMock()
    runner.adapters = {Platform.WHATSAPP: new_adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    await runner._handle_adapter_fatal_error(old_adapter)

    new_adapter.disconnect.assert_not_awaited()
    assert runner.adapters[Platform.WHATSAPP] is new_adapter
    assert Platform.WHATSAPP not in runner._failed_platforms
    runner.stop.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, "reviewer"], ids=["primary", "secondary"])
async def test_inflight_final_reply_uses_replacement_adapter_after_reconnect(
    tmp_path, profile
):
    config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    old_adapter = _ReplacementDeliveryAdapter()
    replacement = _ReplacementDeliveryAdapter()
    old_adapter.gateway_runner = runner
    replacement.gateway_runner = runner
    if profile:
        runner.adapters = {}
        runner._profile_adapters = {profile: {Platform.DISCORD: old_adapter}}
    else:
        runner.adapters = {Platform.DISCORD: old_adapter}
    runner.delivery_router.adapters = runner.adapters

    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(_event):
        await old_adapter.send("channel-1", "partial preview")
        handler_started.set()
        await release_handler.wait()
        return "complete final reply"

    old_adapter.set_message_handler(handler)
    event = MessageEvent(
        text="long-running request",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="channel-1",
            chat_type="dm",
            user_id="user-1",
            profile=profile,
        ),
        message_id="inbound-1",
    )
    task = asyncio.create_task(
        old_adapter._process_message_background(event, build_session_key(event.source))
    )
    await handler_started.wait()

    await old_adapter.disconnect()
    if profile:
        runner._profile_adapters[profile][Platform.DISCORD] = replacement
    else:
        runner.adapters = {Platform.DISCORD: replacement}
    runner.delivery_router.adapters = runner.adapters
    release_handler.set()
    await task

    assert old_adapter.sent == ["partial preview"]
    assert replacement.sent == ["complete final reply"]


def _install_adapter_for_profile(runner, adapter, profile):
    if profile:
        runner.adapters = {}
        runner._profile_adapters = {profile: {Platform.DISCORD: adapter}}
    else:
        runner.adapters = {Platform.DISCORD: adapter}
    runner.delivery_router.adapters = runner.adapters


def _replace_adapter_for_profile(runner, replacement, profile):
    if profile:
        runner._profile_adapters[profile][Platform.DISCORD] = replacement
    else:
        runner.adapters = {Platform.DISCORD: replacement}
    runner.delivery_router.adapters = runner.adapters


def _replacement_event(profile, *, message_type=MessageType.TEXT):
    return MessageEvent(
        text="long-running request",
        message_type=message_type,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="channel-1",
            chat_type="dm",
            user_id="user-1",
            profile=profile,
        ),
        message_id="inbound-1",
    )


def _write_media_file(root: Path, name: str) -> Path:
    path = root / name
    path.write_bytes(b"test media")
    return path.resolve()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, "reviewer"], ids=["primary", "secondary"])
async def test_inflight_final_native_attachments_use_replacement_adapter_after_reconnect(
    tmp_path, monkeypatch, profile
):
    media_root = tmp_path / "media-cache"
    media_root.mkdir()
    image = _write_media_file(media_root, "chart.png")
    audio = _write_media_file(media_root, "audio.mp3")
    video = _write_media_file(media_root, "clip.mp4")
    bare_video = _write_media_file(media_root, "bare-clip.mp4")
    report = _write_media_file(media_root, "report.pdf")
    bare_file = _write_media_file(media_root, "bare.txt")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (media_root,)
    )

    runner = GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, token="token")
            },
            sessions_dir=tmp_path / "sessions",
        )
    )
    old_adapter = _ReplacementDeliveryAdapter()
    replacement = _ReplacementDeliveryAdapter()
    old_adapter.gateway_runner = runner
    replacement.gateway_runner = runner
    _install_adapter_for_profile(runner, old_adapter, profile)

    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(_event):
        await old_adapter.send("channel-1", "partial preview")
        handler_started.set()
        await release_handler.wait()
        return "\n".join(
            (
                "complete final reply",
                "![remote chart](https://example.com/remote.png)",
                f"MEDIA:{image}",
                f"MEDIA:{audio}",
                f"MEDIA:{video}",
                f"MEDIA:{report}",
                str(bare_video),
                str(bare_file),
            )
        )

    old_adapter.set_message_handler(handler)
    event = _replacement_event(profile)
    task = asyncio.create_task(
        old_adapter._process_message_background(event, build_session_key(event.source))
    )
    await handler_started.wait()

    await old_adapter.disconnect()
    _replace_adapter_for_profile(runner, replacement, profile)
    release_handler.set()
    await task

    assert old_adapter.sent == ["partial preview"]
    assert old_adapter.native_attempts == []
    assert replacement.sent == ["complete final reply"]
    assert replacement.image_batches == [
        [("https://example.com/remote.png")],
        [f"file://{image}"],
    ]
    assert replacement.voices == [str(audio)]
    assert replacement.videos == [str(video), str(bare_video)]
    assert replacement.documents == [str(report), str(bare_file)]


@pytest.mark.asyncio
async def test_inflight_as_document_image_uses_replacement_document_sender(
    tmp_path, monkeypatch
):
    media_root = tmp_path / "media-cache"
    media_root.mkdir()
    lossless_image = _write_media_file(media_root, "infographic.png")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (media_root,)
    )

    runner = GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, token="token")
            },
            sessions_dir=tmp_path / "sessions",
        )
    )
    old_adapter = _ReplacementDeliveryAdapter()
    replacement = _ReplacementDeliveryAdapter()
    old_adapter.gateway_runner = runner
    replacement.gateway_runner = runner
    _install_adapter_for_profile(runner, old_adapter, None)

    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(_event):
        handler_started.set()
        await release_handler.wait()
        return f"complete final reply\n[[as_document]]\nMEDIA:{lossless_image}"

    old_adapter.set_message_handler(handler)
    event = _replacement_event(None)
    task = asyncio.create_task(
        old_adapter._process_message_background(event, build_session_key(event.source))
    )
    await handler_started.wait()

    await old_adapter.disconnect()
    _replace_adapter_for_profile(runner, replacement, None)
    release_handler.set()
    await task

    assert old_adapter.native_attempts == []
    assert replacement.sent == ["complete final reply"]
    assert replacement.image_batches == []
    assert replacement.documents == [str(lossless_image)]


@pytest.mark.asyncio
async def test_default_image_batch_reresolves_after_each_human_delay(
    tmp_path, monkeypatch
):
    runner = GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, token="token")
            },
            sessions_dir=tmp_path / "sessions",
        )
    )
    old_adapter = _ReplacementDeliveryAdapter()
    first_replacement = _PerImageReplacementDeliveryAdapter()
    second_replacement = _PerImageReplacementDeliveryAdapter()
    third_replacement = _PerImageReplacementDeliveryAdapter()
    for adapter in (
        old_adapter,
        first_replacement,
        second_replacement,
        third_replacement,
    ):
        adapter.gateway_runner = runner
    _install_adapter_for_profile(runner, old_adapter, None)
    monkeypatch.setattr(old_adapter, "_get_human_delay", lambda: 0.01)

    real_sleep = asyncio.sleep
    sleep_calls = 0

    async def replace_during_delay(delay):
        nonlocal sleep_calls
        if delay == 0.01:
            sleep_calls += 1
            replacement = second_replacement if sleep_calls == 1 else third_replacement
            _replace_adapter_for_profile(runner, replacement, None)
        await real_sleep(0)

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", replace_during_delay)
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(_event):
        handler_started.set()
        await release_handler.wait()
        return "\n".join(
            (
                "complete final reply",
                "![first](https://example.com/first.png)",
                "![second](https://example.com/second.png)",
            )
        )

    old_adapter.set_message_handler(handler)
    event = _replacement_event(None)
    task = asyncio.create_task(
        old_adapter._process_message_background(event, build_session_key(event.source))
    )
    await handler_started.wait()

    await old_adapter.disconnect()
    _replace_adapter_for_profile(runner, first_replacement, None)
    release_handler.set()
    await task

    assert sleep_calls == 2
    assert first_replacement.sent == ["complete final reply"]
    assert first_replacement.sent_images == []
    assert second_replacement.sent_images == ["https://example.com/first.png"]
    assert third_replacement.sent_images == ["https://example.com/second.png"]


@pytest.mark.asyncio
async def test_each_final_attachment_resolves_the_latest_replacement_adapter(
    tmp_path, monkeypatch
):
    media_root = tmp_path / "media-cache"
    media_root.mkdir()
    audio = _write_media_file(media_root, "audio.mp3")
    video = _write_media_file(media_root, "clip.mp4")
    report = _write_media_file(media_root, "report.pdf")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (media_root,)
    )

    runner = GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, token="token")
            },
            sessions_dir=tmp_path / "sessions",
        )
    )
    old_adapter = _ReplacementDeliveryAdapter()
    first_replacement = _ReplacementDeliveryAdapter()
    second_replacement = _ReplacementDeliveryAdapter()
    for adapter in (old_adapter, first_replacement, second_replacement):
        adapter.gateway_runner = runner
    _install_adapter_for_profile(runner, old_adapter, None)

    original_send_voice = first_replacement.send_voice

    async def send_voice_then_replace(*args, **kwargs):
        result = await original_send_voice(*args, **kwargs)
        _replace_adapter_for_profile(runner, second_replacement, None)
        return result

    monkeypatch.setattr(first_replacement, "send_voice", send_voice_then_replace)
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(_event):
        handler_started.set()
        await release_handler.wait()
        return "\n".join(
            (
                "complete final reply",
                f"MEDIA:{audio}",
                f"MEDIA:{video}",
                f"MEDIA:{report}",
            )
        )

    old_adapter.set_message_handler(handler)
    event = _replacement_event(None)
    task = asyncio.create_task(
        old_adapter._process_message_background(event, build_session_key(event.source))
    )
    await handler_started.wait()

    await old_adapter.disconnect()
    _replace_adapter_for_profile(runner, first_replacement, None)
    release_handler.set()
    await task

    assert old_adapter.native_attempts == []
    assert first_replacement.sent == ["complete final reply"]
    assert first_replacement.voices == [str(audio)]
    assert first_replacement.videos == []
    assert first_replacement.documents == []
    assert second_replacement.videos == [str(video)]
    assert second_replacement.documents == [str(report)]


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, "reviewer"], ids=["primary", "secondary"])
async def test_inflight_auto_tts_uses_replacement_adapter_after_reconnect(
    tmp_path, monkeypatch, profile
):
    tts_path = tmp_path / "tts-reply.mp3"
    tts_path.write_bytes(b"test audio")
    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool",
        lambda **_kwargs: json.dumps({"success": True, "file_path": str(tts_path)}),
    )

    runner = GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, token="token")
            },
            sessions_dir=tmp_path / "sessions",
        )
    )
    old_adapter = _ReplacementDeliveryAdapter()
    replacement = _ReplacementDeliveryAdapter()
    old_adapter.gateway_runner = runner
    replacement.gateway_runner = runner
    monkeypatch.setattr(old_adapter, "_should_auto_tts_for_chat", lambda _chat_id: True)
    _install_adapter_for_profile(runner, old_adapter, profile)

    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(_event):
        handler_started.set()
        await release_handler.wait()
        return "complete final reply"

    old_adapter.set_message_handler(handler)
    event = _replacement_event(profile, message_type=MessageType.VOICE)
    task = asyncio.create_task(
        old_adapter._process_message_background(event, build_session_key(event.source))
    )
    await handler_started.wait()

    await old_adapter.disconnect()
    _replace_adapter_for_profile(runner, replacement, profile)
    release_handler.set()
    await task

    assert old_adapter.native_attempts == []
    assert replacement.voices == [str(tts_path)]
    assert replacement.sent == ["complete final reply"]
    assert not tts_path.exists()


@pytest.mark.asyncio
async def test_inflight_auto_tts_failure_notifies_through_replacement(
    tmp_path, monkeypatch
):
    tts_path = tmp_path / "tts-reply.mp3"
    tts_path.write_bytes(b"test audio")
    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool",
        lambda **_kwargs: json.dumps({"success": True, "file_path": str(tts_path)}),
    )

    runner = GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, token="token")
            },
            sessions_dir=tmp_path / "sessions",
        )
    )
    old_adapter = _ReplacementDeliveryAdapter()
    replacement = _ReplacementDeliveryAdapter()
    replacement.fail_kinds.add("voice")
    old_adapter.gateway_runner = runner
    replacement.gateway_runner = runner
    monkeypatch.setattr(old_adapter, "_should_auto_tts_for_chat", lambda _chat_id: True)
    _install_adapter_for_profile(runner, old_adapter, None)

    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(_event):
        handler_started.set()
        await release_handler.wait()
        return "complete final reply"

    old_adapter.set_message_handler(handler)
    event = _replacement_event(None, message_type=MessageType.VOICE)
    task = asyncio.create_task(
        old_adapter._process_message_background(event, build_session_key(event.source))
    )
    await handler_started.wait()

    await old_adapter.disconnect()
    _replace_adapter_for_profile(runner, replacement, None)
    release_handler.set()
    await task

    notice = "⚠️ Couldn't deliver one or more attachments from the final response."
    assert old_adapter.native_attempts == []
    assert replacement.sent == ["complete final reply", notice]
    assert not tts_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_kind", "file_name"),
    [("document", "report.pdf"), ("image", "chart.png")],
    ids=["document", "image-batch"],
)
async def test_inflight_final_attachment_failure_notifies_through_replacement(
    tmp_path, monkeypatch, failure_kind, file_name
):
    media_root = tmp_path / "media-cache"
    media_root.mkdir()
    attachment = _write_media_file(media_root, file_name)
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (media_root,)
    )

    runner = GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, token="token")
            },
            sessions_dir=tmp_path / "sessions",
        )
    )
    old_adapter = _ReplacementDeliveryAdapter()
    replacement = _ReplacementDeliveryAdapter()
    replacement.fail_kinds.add(failure_kind)
    old_adapter.gateway_runner = runner
    replacement.gateway_runner = runner
    _install_adapter_for_profile(runner, old_adapter, None)

    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(_event):
        handler_started.set()
        await release_handler.wait()
        return f"complete final reply\nMEDIA:{attachment}"

    old_adapter.set_message_handler(handler)
    event = _replacement_event(None)
    task = asyncio.create_task(
        old_adapter._process_message_background(event, build_session_key(event.source))
    )
    await handler_started.wait()

    await old_adapter.disconnect()
    _replace_adapter_for_profile(runner, replacement, None)
    release_handler.set()
    await task

    notice = "⚠️ Couldn't deliver one or more attachments from the final response."
    assert old_adapter.native_attempts == []
    assert replacement.sent == ["complete final reply", notice]
    assert str(attachment) not in "\n".join(replacement.sent)
