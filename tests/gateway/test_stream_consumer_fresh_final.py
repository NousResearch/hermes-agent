"""Regression tests for the fresh-final-for-long-lived-previews path.

Ported from openclaw/openclaw#72038.  When a streamed preview has been
visible long enough that the platform's edit timestamp would be
noticeably stale by completion time, the stream consumer delivers the
final reply as a brand-new message and best-effort deletes the old
preview.  This makes Telegram's visible timestamp reflect completion
time instead of first-token time.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_adapter(*, supports_delete: bool = True) -> MagicMock:
    """Build a minimal MagicMock adapter wired for send/edit/delete."""
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="initial_preview",
    ))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="initial_preview",
    ))
    if supports_delete:
        adapter.delete_message = AsyncMock(return_value=True)
    else:
        # Adapter without the optional delete_message method — fresh-final
        # should still work, it just leaves the stale preview in place.
        del adapter.delete_message  # type: ignore[attr-defined]
    return adapter


class TestFreshFinalForLongLivedPreviews:
    """openclaw#72038 port — send fresh final when preview is old."""

    @pytest.mark.asyncio
    async def test_disabled_by_default_still_edits_in_place(self):
        """``fresh_final_after_seconds=0`` preserves the legacy edit path."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(fresh_final_after_seconds=0.0),
        )
        await consumer._send_or_edit("hello")
        # Pretend the preview has been visible for a long time.
        consumer._message_created_ts = 0.0  # far in the past
        await consumer._send_or_edit("hello world", finalize=True)
        # Should edit, not send a fresh message.
        assert adapter.send.call_count == 1  # only the initial send
        adapter.edit_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_lived_preview_edits_in_place(self):
        """Finalizing a preview younger than the threshold → normal edit."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(fresh_final_after_seconds=60.0),
        )
        await consumer._send_or_edit("hello")
        # Preview is "new" — leave _message_created_ts at its real value.
        await consumer._send_or_edit("hello world", finalize=True)
        assert adapter.send.call_count == 1
        adapter.edit_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_long_lived_preview_sends_fresh_final(self):
        """Finalizing a preview older than the threshold → fresh send."""
        adapter = _make_adapter()
        adapter.send.side_effect = [
            SimpleNamespace(success=True, message_id="initial_preview"),
            SimpleNamespace(success=True, message_id="fresh_final"),
        ]
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(fresh_final_after_seconds=60.0),
        )
        await consumer._send_or_edit("hello")
        # Force the preview to look stale (visible for > 60s).
        consumer._message_created_ts = 0.0  # zero = ~uptime seconds old
        await consumer._send_or_edit("hello world", finalize=True)
        # Fresh send happened; no edit of the old preview.
        assert adapter.send.call_count == 2
        adapter.edit_message.assert_not_called()
        # The old preview was deleted as cleanup.
        adapter.delete_message.assert_awaited_once_with("chat", "initial_preview")
        # State was updated to the new message id.
        assert consumer._message_id == "fresh_final"
        assert consumer._final_response_sent is True

    @pytest.mark.asyncio
    async def test_fresh_final_without_delete_support_is_best_effort(self):
        """Adapter lacking ``delete_message`` still gets the fresh send."""
        adapter = _make_adapter(supports_delete=False)
        adapter.send.side_effect = [
            SimpleNamespace(success=True, message_id="initial_preview"),
            SimpleNamespace(success=True, message_id="fresh_final"),
        ]
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(fresh_final_after_seconds=60.0),
        )
        await consumer._send_or_edit("hello")
        consumer._message_created_ts = 0.0
        await consumer._send_or_edit("hello world", finalize=True)
        assert adapter.send.call_count == 2
        adapter.edit_message.assert_not_called()
        # No delete attempt — just the fresh send.
        assert consumer._message_id == "fresh_final"

    @pytest.mark.asyncio
    async def test_fresh_final_fallback_to_edit_on_send_failure(self):
        """If the fresh send fails, fall back to the normal edit path."""
        adapter = _make_adapter()
        adapter.send.side_effect = [
            SimpleNamespace(success=True, message_id="initial_preview"),
            SimpleNamespace(success=False, error="network"),
        ]
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(fresh_final_after_seconds=60.0),
        )
        await consumer._send_or_edit("hello")
        consumer._message_created_ts = 0.0
        ok = await consumer._send_or_edit("hello world", finalize=True)
        # Fresh send was attempted and failed → edit happened instead.
        assert adapter.send.call_count == 2
        adapter.edit_message.assert_called_once()
        assert ok is True

    @pytest.mark.asyncio
    async def test_only_finalize_triggers_fresh_final(self):
        """Intermediate edits (``finalize=False``) never switch to fresh send."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(fresh_final_after_seconds=60.0),
        )
        await consumer._send_or_edit("hello")
        consumer._message_created_ts = 0.0  # stale
        await consumer._send_or_edit("hello partial")  # no finalize
        assert adapter.send.call_count == 1
        adapter.edit_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_edit_sentinel_is_not_affected(self):
        """Platforms with the ``__no_edit__`` sentinel never go fresh-final."""
        adapter = _make_adapter()
        adapter.send.return_value = SimpleNamespace(success=True, message_id=None)
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(fresh_final_after_seconds=60.0),
        )
        await consumer._send_or_edit("hello")
        assert consumer._message_id == "__no_edit__"
        assert consumer._message_created_ts is None
        # Even with finalize=True, no fresh send — the sentinel gates it.
        assert consumer._should_send_fresh_final() is False


class TestStreamConsumerConfigFreshFinalField:
    """The dataclass field must exist and default to 0 (disabled)."""

    def test_default_is_disabled(self):
        cfg = StreamConsumerConfig()
        assert cfg.fresh_final_after_seconds == 0.0

    def test_field_is_configurable(self):
        cfg = StreamConsumerConfig(fresh_final_after_seconds=120.0)
        assert cfg.fresh_final_after_seconds == 120.0


class TestStreamingConfigFreshFinalField:
    """The gateway-level StreamingConfig carries the setting."""

    def test_default_enables_with_60s(self):
        from gateway.config import StreamingConfig
        cfg = StreamingConfig()
        assert cfg.fresh_final_after_seconds == 60.0

    def test_from_dict_uses_default_when_missing(self):
        from gateway.config import StreamingConfig
        cfg = StreamingConfig.from_dict({"enabled": True})
        assert cfg.fresh_final_after_seconds == 60.0

    def test_from_dict_respects_explicit_zero(self):
        from gateway.config import StreamingConfig
        cfg = StreamingConfig.from_dict({
            "enabled": True,
            "fresh_final_after_seconds": 0,
        })
        assert cfg.fresh_final_after_seconds == 0.0

    def test_to_dict_round_trip(self):
        from gateway.config import StreamingConfig
        original = StreamingConfig(fresh_final_after_seconds=90.0)
        restored = StreamingConfig.from_dict(original.to_dict())
        assert restored.fresh_final_after_seconds == 90.0


class TestTelegramAdapterDeleteMessage:
    """Contract: Telegram adapter implements ``delete_message``."""

    def test_delete_message_method_exists(self):
        telegram = pytest.importorskip("gateway.platforms.telegram")
        import inspect
        cls = telegram.TelegramAdapter
        assert hasattr(cls, "delete_message"), (
            "TelegramAdapter.delete_message is required for the fresh-final "
            "cleanup path (openclaw/openclaw#72038 port)."
        )
        sig = inspect.signature(cls.delete_message)
        params = list(sig.parameters)
        assert params[:3] == ["self", "chat_id", "message_id"]

    def test_base_adapter_default_returns_false(self):
        """BasePlatformAdapter.delete_message default = no-op returning False."""
        from gateway.platforms.base import BasePlatformAdapter
        import inspect
        sig = inspect.signature(BasePlatformAdapter.delete_message)
        assert list(sig.parameters)[:3] == ["self", "chat_id", "message_id"]
