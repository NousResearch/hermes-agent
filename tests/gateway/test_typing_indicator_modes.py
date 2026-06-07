"""Tests for the per-platform ``typing_indicator`` policy.

Three modes are supported on ``PlatformConfig.typing_indicator``:

  - ``always``     — legacy: refresh typing... continuously for the whole turn.
                     Default for every platform EXCEPT Telegram.
  - ``stream_only`` — only refresh typing... while assistant text is actively
                      streaming/about to be sent (gated by
                      ``_typing_streaming``). NEW default for Telegram.
  - ``off``        — never send a typing indicator.

These tests pin all three behaviors at the ``_keep_typing`` level plus the
default-resolution logic on ``typing_indicator_mode()``.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    Platform,
    PlatformConfig,
    SendResult,
)


class _StubAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform = Platform.TELEGRAM, typing_indicator=None):
        super().__init__(
            PlatformConfig(enabled=True, token="test", typing_indicator=typing_indicator),
            platform,
        )

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


class TestTypingIndicatorModeResolution:
    def test_telegram_default_is_stream_only(self):
        adapter = _StubAdapter(platform=Platform.TELEGRAM)
        assert adapter.typing_indicator_mode() == "stream_only"

    def test_discord_default_is_always(self):
        adapter = _StubAdapter(platform=Platform.DISCORD)
        assert adapter.typing_indicator_mode() == "always"

    def test_explicit_off_wins_over_default(self):
        adapter = _StubAdapter(platform=Platform.TELEGRAM, typing_indicator="off")
        assert adapter.typing_indicator_mode() == "off"

    def test_explicit_always_wins_over_telegram_default(self):
        adapter = _StubAdapter(platform=Platform.TELEGRAM, typing_indicator="always")
        assert adapter.typing_indicator_mode() == "always"

    def test_unknown_value_falls_through_to_default(self):
        # Invalid values are sanitized to None at config-load time, but
        # if one sneaks through ``typing_indicator_mode`` must still
        # return a valid mode (the platform default).
        adapter = _StubAdapter(platform=Platform.TELEGRAM, typing_indicator="garbage")
        assert adapter.typing_indicator_mode() == "stream_only"


class TestKeepTypingHonorsMode:
    @pytest.mark.asyncio
    async def test_always_mode_fires_send_typing(self, monkeypatch):
        adapter = _StubAdapter(platform=Platform.DISCORD)
        calls = []

        async def recording(chat_id, metadata=None):
            calls.append(chat_id)

        monkeypatch.setattr(adapter, "send_typing", recording)
        adapter.stop_typing = MagicMock(return_value=asyncio.sleep(0))

        stop = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing(chat_id="c1", interval=0.3, stop_event=stop, mode="always")
        )
        await asyncio.sleep(1.0)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)

        assert len(calls) >= 2, f"always mode should keep firing send_typing; got {calls}"

    @pytest.mark.asyncio
    async def test_stream_only_skips_when_not_streaming(self, monkeypatch):
        adapter = _StubAdapter(platform=Platform.TELEGRAM)
        calls = []

        async def recording(chat_id, metadata=None):
            calls.append(chat_id)

        monkeypatch.setattr(adapter, "send_typing", recording)
        adapter.stop_typing = MagicMock(return_value=asyncio.sleep(0))

        stop = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing(chat_id="c1", interval=0.3, stop_event=stop, mode="stream_only")
        )
        await asyncio.sleep(1.0)  # ~3 ticks elapsed; no streaming flag set
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)

        assert calls == [], (
            f"stream_only must not call send_typing while _typing_streaming "
            f"is empty; got {calls}"
        )

    @pytest.mark.asyncio
    async def test_stream_only_fires_when_streaming_flag_set(self, monkeypatch):
        adapter = _StubAdapter(platform=Platform.TELEGRAM)
        calls = []

        async def recording(chat_id, metadata=None):
            calls.append(chat_id)

        monkeypatch.setattr(adapter, "send_typing", recording)
        adapter.stop_typing = MagicMock(return_value=asyncio.sleep(0))
        adapter.start_streaming_typing("c1")

        stop = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing(chat_id="c1", interval=0.3, stop_event=stop, mode="stream_only")
        )
        await asyncio.sleep(1.0)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)

        assert len(calls) >= 2, (
            f"stream_only should fire send_typing once start_streaming_typing "
            f"was called; got {calls}"
        )

    @pytest.mark.asyncio
    async def test_stream_only_starts_quiet_then_wakes_up(self, monkeypatch):
        """Verify the live transition: no ticks while idle, ticks once the
        streaming flag flips on."""
        adapter = _StubAdapter(platform=Platform.TELEGRAM)
        calls = []

        async def recording(chat_id, metadata=None):
            calls.append((chat_id, asyncio.get_event_loop().time()))

        monkeypatch.setattr(adapter, "send_typing", recording)
        adapter.stop_typing = MagicMock(return_value=asyncio.sleep(0))

        stop = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing(chat_id="c1", interval=0.25, stop_event=stop, mode="stream_only")
        )
        # Phase 1: no streaming -> no calls
        await asyncio.sleep(0.7)
        assert calls == [], f"expected silence in phase 1, got {calls}"
        # Phase 2: turn streaming on
        adapter.start_streaming_typing("c1")
        await asyncio.sleep(0.8)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)
        assert len(calls) >= 1, (
            f"expected send_typing after start_streaming_typing, got {calls}"
        )


class TestStreamingHelpers:
    def test_start_and_stop_streaming_typing(self):
        adapter = _StubAdapter()
        adapter.start_streaming_typing("abc")
        assert "abc" in adapter._typing_streaming
        adapter.stop_streaming_typing("abc")
        assert "abc" not in adapter._typing_streaming

    def test_stop_streaming_typing_is_idempotent(self):
        adapter = _StubAdapter()
        # Should not raise when the chat was never marked streaming.
        adapter.stop_streaming_typing("never-streamed")


class TestKeepTypingDefaultInterval:
    def test_default_interval_is_four_seconds(self):
        """Refresh interval was bumped from 2.0s → 4.0s. Telegram's typing
        bubble expires at ~5s, so 4s keeps it alive with one HTTP call per
        tick instead of two."""
        import inspect

        sig = inspect.signature(BasePlatformAdapter._keep_typing)
        assert sig.parameters["interval"].default == 4.0
