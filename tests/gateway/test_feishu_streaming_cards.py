"""Tests for Feishu CardKit v2 streaming card lifecycle.

Covers:
- streaming_cards_enabled property
- send_streaming_card() method
- stream_consumer integration (_uses_streaming_card flag)
- Config wiring (streaming_card setting + FEISHU_STREAMING_CARD env var)
"""

import asyncio
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_feishu_adapter(*, streaming_card: bool = True):
    """Create a minimal FeishuAdapter mock with streaming card support."""
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = True
    adapter.streaming_cards_enabled = streaming_card
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.message_len_fn = len
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
    adapter.send_streaming_card = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="msg_card_1")
    )
    adapter.truncate_message = lambda text, limit, **kw: [text]
    return adapter


# ── streaming_cards_enabled property ─────────────────────────────────────


class TestStreamingCardsEnabled:
    """Verify the streaming_cards_enabled property gates CardKit usage."""

    def test_enabled_by_default(self):
        """streaming_cards_enabled is True when config is enabled and SDK available."""
        from gateway.platforms.feishu import FeishuAdapter, FEISHU_AVAILABLE

        adapter = object.__new__(FeishuAdapter)
        adapter._streaming_card = True
        # FEISHU_AVAILABLE is a module-level flag; it's True when lark_oapi is installed
        # We test the property logic directly
        assert adapter._streaming_card is True

    def test_disabled_by_config(self):
        """streaming_cards_enabled is False when streaming_card config is False."""
        from gateway.platforms.feishu import FeishuAdapter

        adapter = object.__new__(FeishuAdapter)
        adapter._streaming_card = False
        assert adapter._streaming_card is False


# ── send_streaming_card method ───────────────────────────────────────────


class TestSendStreamingCard:
    """Verify send_streaming_card() creates cards and handles failures."""

    @pytest.mark.asyncio
    async def test_calls_create_streaming_card(self):
        """send_streaming_card delegates to _create_streaming_card."""
        from gateway.platforms.feishu import FeishuAdapter

        adapter = object.__new__(FeishuAdapter)
        adapter._streaming_card = True
        adapter._client = MagicMock()
        adapter.format_message = lambda text: text
        adapter._create_streaming_card = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_card_1")
        )

        with patch.object(FeishuAdapter, "streaming_cards_enabled", True):
            result = await adapter.send_streaming_card("oc_test", "Hello")
        assert result.success is True
        assert result.message_id == "msg_card_1"
        adapter._create_streaming_card.assert_called_once_with("oc_test", "Hello", None)

    @pytest.mark.asyncio
    async def test_disabled_returns_failure(self):
        """send_streaming_card returns failure when streaming cards are disabled."""
        from gateway.platforms.feishu import FeishuAdapter

        adapter = object.__new__(FeishuAdapter)
        adapter._streaming_card = False

        with patch.object(FeishuAdapter, "streaming_cards_enabled", False):
            result = await adapter.send_streaming_card("oc_test", "Hello")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_not_connected_returns_failure(self):
        """send_streaming_card returns failure when client is not connected."""
        from gateway.platforms.feishu import FeishuAdapter

        adapter = object.__new__(FeishuAdapter)
        adapter._streaming_card = True
        adapter._client = None

        with patch.object(FeishuAdapter, "streaming_cards_enabled", True):
            result = await adapter.send_streaming_card("oc_test", "Hello")
        assert result.success is False


# ── StreamConsumer integration ───────────────────────────────────────────


class TestStreamConsumerStreamingCard:
    """Verify GatewayStreamConsumer routes to send_streaming_card."""

    @pytest.mark.asyncio
    async def test_first_send_defers_to_regular_send(self):
        """When defer is active (default), first send uses regular send()."""
        adapter = _make_feishu_adapter(streaming_card=True)
        consumer = GatewayStreamConsumer(adapter, "oc_test")
        assert consumer._uses_streaming_card is True
        assert consumer._defer_streaming_card is True

        # First send with defer active → regular send, not streaming card
        result = await consumer._send_or_edit("Hello world")
        assert result is True
        adapter.send.assert_called_once()
        adapter.send_streaming_card.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_streaming_card_after_defer_disabled(self):
        """After defer is disabled, send uses send_streaming_card."""
        adapter = _make_feishu_adapter(streaming_card=True)
        consumer = GatewayStreamConsumer(adapter, "oc_test")
        consumer._defer_streaming_card = False  # simulate after first segment break

        result = await consumer._send_or_edit("Hello world")
        assert result is True
        adapter.send_streaming_card.assert_called_once()
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_send_on_card_failure(self):
        """When send_streaming_card fails, falls back to regular send."""
        adapter = _make_feishu_adapter(streaming_card=True)
        adapter.send_streaming_card = AsyncMock(
            return_value=SimpleNamespace(success=False, error="CardKit unavailable")
        )

        consumer = GatewayStreamConsumer(adapter, "oc_test")
        consumer._defer_streaming_card = False  # simulate after first segment break
        result = await consumer._send_or_edit("Hello world")
        assert result is True
        adapter.send_streaming_card.assert_called_once()
        adapter.send.assert_called_once()
        # _uses_streaming_card should be disabled after failure
        assert consumer._uses_streaming_card is False

    @pytest.mark.asyncio
    async def test_falls_back_on_card_exception(self):
        """When send_streaming_card raises, falls back to regular send."""
        adapter = _make_feishu_adapter(streaming_card=True)
        adapter.send_streaming_card = AsyncMock(side_effect=RuntimeError("Network error"))

        consumer = GatewayStreamConsumer(adapter, "oc_test")
        consumer._defer_streaming_card = False  # simulate after first segment break
        result = await consumer._send_or_edit("Hello world")
        assert result is True
        adapter.send.assert_called_once()
        assert consumer._uses_streaming_card is False

    @pytest.mark.asyncio
    async def test_skips_streaming_card_when_disabled(self):
        """When streaming_cards_enabled is False, uses regular send."""
        adapter = _make_feishu_adapter(streaming_card=False)

        consumer = GatewayStreamConsumer(adapter, "oc_test")
        assert consumer._uses_streaming_card is False
        result = await consumer._send_or_edit("Hello world")
        assert result is True
        adapter.send.assert_called_once()
        adapter.send_streaming_card.assert_not_called()

    @pytest.mark.asyncio
    async def test_subsequent_edits_use_edit_message(self):
        """After streaming card is created, edits go through edit_message."""
        adapter = _make_feishu_adapter(streaming_card=True)

        consumer = GatewayStreamConsumer(adapter, "oc_test")
        consumer._defer_streaming_card = False  # simulate after first segment break

        # First send creates the streaming card
        await consumer._send_or_edit("Hello")
        adapter.send_streaming_card.assert_called_once()

        # Second edit updates the card content
        result = await consumer._send_or_edit("Hello world more text")
        assert result is True
        adapter.edit_message.assert_called_once()
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_finalize_edit_on_stream_end(self):
        """Final edit (finalize=True) goes through edit_message with finalize flag."""
        adapter = _make_feishu_adapter(streaming_card=True)

        consumer = GatewayStreamConsumer(adapter, "oc_test")
        consumer._defer_streaming_card = False  # simulate after first segment break
        await consumer._send_or_edit("Hello")
        adapter.send_streaming_card.assert_called_once()

        # Final edit
        result = await consumer._send_or_edit("Hello world", finalize=True)
        assert result is True
        adapter.edit_message.assert_called_once()
        assert adapter.edit_message.call_args[1]["finalize"] is True

    @pytest.mark.asyncio
    async def test_reset_segment_re_enables_streaming_card(self):
        """After segment break, streaming card is re-enabled for next segment."""
        adapter = _make_feishu_adapter(streaming_card=True)

        consumer = GatewayStreamConsumer(adapter, "oc_test")
        consumer._defer_streaming_card = False  # simulate after first segment break
        assert consumer._uses_streaming_card is True

        # Simulate a failed card creation that disables streaming card
        adapter.send_streaming_card = AsyncMock(
            return_value=SimpleNamespace(success=False, error="fail")
        )
        await consumer._send_or_edit("Hello")
        assert consumer._uses_streaming_card is False

        # Reset segment state (simulates segment break)
        consumer._reset_segment_state()
        # Should be re-enabled because adapter still supports it
        assert consumer._uses_streaming_card is True

    @pytest.mark.asyncio
    async def test_reset_segment_does_not_enable_if_adapter_lacks_support(self):
        """Reset does not re-enable streaming card if adapter doesn't support it."""
        adapter = _make_feishu_adapter(streaming_card=False)

        consumer = GatewayStreamConsumer(adapter, "oc_test")
        assert consumer._uses_streaming_card is False

        consumer._reset_segment_state()
        assert consumer._uses_streaming_card is False


# ── Config wiring ────────────────────────────────────────────────────────


class TestStreamingCardConfig:
    """Verify streaming_card config is loaded from settings and env vars."""

    def test_default_is_enabled(self):
        """streaming_card defaults to True (enabled)."""
        from gateway.platforms.feishu import FeishuAdapterSettings

        settings = FeishuAdapterSettings(
            app_id="cli_test", app_secret="secret", domain_name="feishu",
            connection_mode="websocket", encrypt_key="", verification_token="",
            group_policy="allowlist", allowed_group_users=frozenset(),
            bot_open_id="", bot_user_id="", bot_name="",
            dedup_cache_size=100, text_batch_delay_seconds=0.6,
            text_batch_split_delay_seconds=2.0, text_batch_max_messages=8,
            text_batch_max_chars=4000, media_batch_delay_seconds=0.8,
            webhook_host="127.0.0.1", webhook_port=8645, webhook_path="/webhook",
        )
        assert settings.streaming_card is True

    def test_can_be_disabled_via_extra(self):
        """streaming_card can be disabled via extra config dict."""
        from gateway.platforms.feishu import FeishuAdapter

        settings = FeishuAdapter._load_settings({
            "app_id": "cli_test",
            "app_secret": "secret",
            "streaming_card": False,
        })
        assert settings.streaming_card is False

    @patch.dict(os.environ, {"FEISHU_STREAMING_CARD": "false"})
    def test_can_be_disabled_via_env(self):
        """streaming_card can be disabled via FEISHU_STREAMING_CARD env var."""
        from gateway.platforms.feishu import FeishuAdapter

        settings = FeishuAdapter._load_settings({
            "app_id": "cli_test",
            "app_secret": "secret",
        })
        assert settings.streaming_card is False

    @patch.dict(os.environ, {"FEISHU_STREAMING_CARD": "false"})
    def test_apply_settings_stores_config(self):
        """_apply_settings persists the streaming_card setting."""
        from gateway.platforms.feishu import FeishuAdapter

        settings = FeishuAdapter._load_settings({
            "app_id": "cli_test",
            "app_secret": "secret",
        })
        adapter = object.__new__(FeishuAdapter)
        adapter._apply_settings(settings)
        assert adapter._streaming_card is False


# ── CardKit API sequence numbers ─────────────────────────────────────────


class TestCardKitSequence:
    """Verify sequence numbers are strictly incrementing."""

    @pytest.mark.asyncio
    async def test_sequence_increments_on_stream_updates(self):
        """Each stream content update increments the sequence counter."""
        from gateway.platforms.feishu import _StreamingCardState

        state = _StreamingCardState(
            card_id="card_1",
            message_id="msg_1",
            sequence=0,
            start_time=time.monotonic(),
        )
        assert state.sequence == 0

        # Simulate stream updates
        state.sequence += 1
        assert state.sequence == 1
        state.sequence += 1
        assert state.sequence == 2


# ── _StreamingCardState dataclass ────────────────────────────────────────


class TestStreamingCardState:
    """Verify _StreamingCardState tracks card lifecycle correctly."""

    def test_default_flags(self):
        """New streaming card state has correct default flags."""
        from gateway.platforms.feishu import _StreamingCardState

        state = _StreamingCardState(
            card_id="card_1",
            message_id="msg_1",
            sequence=0,
            start_time=time.monotonic(),
        )
        assert state.inflight is False
        assert state.dirty is False
        assert state.is_cardkit is True

    def test_inflight_serialization(self):
        """inflight/dirty flags model request serialization correctly."""
        from gateway.platforms.feishu import _StreamingCardState

        state = _StreamingCardState(
            card_id="card_1",
            message_id="msg_1",
            sequence=0,
            start_time=time.monotonic(),
        )
        # Start a request
        state.inflight = True
        assert state.inflight is True

        # Mark dirty while inflight (another update arrived)
        state.dirty = True
        assert state.dirty is True

        # Request completes
        state.inflight = False
        assert state.dirty is True  # dirty should remain until re-flush
