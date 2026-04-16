"""Tests for Feishu CardKit streaming card support.

Covers:
- send_streaming_card(): card creation + message send + state tracking
- edit_message() routing to CardKit streaming API for active cards
- stop_streaming_card(): finalization + state cleanup
- _build_streaming_card_json(): card JSON structure
- GatewayStreamConsumer integration with streaming cards
"""

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# lark_oapi is an optional dependency (hermes-agent[feishu]) — skip the entire
# module when it is not installed so CI doesn't fail in minimal environments.
pytest.importorskip("lark_oapi", reason="lark_oapi not installed — skipping Feishu streaming card tests")


@patch.dict(os.environ, {}, clear=True)
class TestBuildStreamingCardJson(unittest.TestCase):
    """Test the static card JSON builder."""

    def test_card_json_has_streaming_mode_enabled(self):
        from gateway.platforms.feishu import FeishuAdapter

        card = FeishuAdapter._build_streaming_card_json("")
        assert card["schema"] == "2.0"
        assert card["config"]["streaming_mode"] is True

    def test_card_json_has_streaming_config(self):
        from gateway.platforms.feishu import FeishuAdapter

        card = FeishuAdapter._build_streaming_card_json("")
        sc = card["config"]["streaming_config"]
        assert "print_frequency_ms" in sc
        assert "print_step" in sc
        assert sc["print_strategy"] in ("fast", "delay")

    def test_card_json_has_markdown_element_with_id(self):
        from gateway.platforms.feishu import FeishuAdapter, _STREAMING_CARD_ELEMENT_ID

        card = FeishuAdapter._build_streaming_card_json("hello")
        elements = card["body"]["elements"]
        assert len(elements) == 1
        assert elements[0]["tag"] == "markdown"
        assert elements[0]["content"] == "hello"
        assert elements[0]["element_id"] == _STREAMING_CARD_ELEMENT_ID

    def test_card_json_default_empty_content(self):
        from gateway.platforms.feishu import FeishuAdapter

        card = FeishuAdapter._build_streaming_card_json()
        assert card["body"]["elements"][0]["content"] == ""


@patch.dict(os.environ, {}, clear=True)
class TestSendStreamingCard(unittest.TestCase):
    """Test send_streaming_card flow."""

    def _make_adapter(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        return adapter

    def _mock_cardkit_and_im(self, adapter, card_id="card_123", message_id="om_456"):
        """Wire up mock client with cardkit + im APIs."""
        captured = {"cardkit_create": None, "im_create": None}

        class _CardAPI:
            def create(self, request):
                captured["cardkit_create"] = request
                return SimpleNamespace(
                    code=0,
                    msg="success",
                    data=SimpleNamespace(card_id=card_id),
                )

            def settings(self, request):
                captured["cardkit_settings"] = request
                return SimpleNamespace(code=0, msg="success")

        class _CardElementAPI:
            def content(self, request):
                captured["cardkit_content"] = request
                return SimpleNamespace(code=0, msg="success")

        class _MessageAPI:
            def create(self, request):
                captured["im_create"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id=message_id),
                )

            def update(self, request):
                captured["im_update"] = request
                return SimpleNamespace(success=lambda: True)

        adapter._client = SimpleNamespace(
            cardkit=SimpleNamespace(
                v1=SimpleNamespace(
                    card=_CardAPI(),
                    card_element=_CardElementAPI(),
                )
            ),
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            ),
        )
        return captured

    def test_send_streaming_card_creates_card_and_sends_message(self):
        adapter = self._make_adapter()
        captured = self._mock_cardkit_and_im(adapter)

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send_streaming_card(
                    chat_id="oc_chat",
                    content="initial",
                )
            )

        assert result.success
        assert result.message_id == "om_456"
        # Card entity was created
        assert captured["cardkit_create"] is not None
        create_body = captured["cardkit_create"].request_body
        assert create_body.type == "card_json"
        card_data = json.loads(create_body.data)
        assert card_data["config"]["streaming_mode"] is True
        assert card_data["body"]["elements"][0]["content"] == "initial"
        # Message was sent with card_id reference
        assert captured["im_create"] is not None
        im_content = json.loads(captured["im_create"].request_body.content)
        assert im_content["type"] == "card"
        assert im_content["data"]["card_id"] == "card_123"
        # Streaming card state is tracked
        assert "om_456" in adapter._streaming_cards
        sc = adapter._streaming_cards["om_456"]
        assert sc.card_id == "card_123"
        assert sc.sequence == 1

    def test_send_streaming_card_fails_when_not_connected(self):
        adapter = self._make_adapter()
        adapter._client = None
        result = asyncio.run(
            adapter.send_streaming_card(chat_id="oc_chat", content="test")
        )
        assert not result.success
        assert "Not connected" in result.error

    def test_send_streaming_card_fails_on_card_create_error(self):
        adapter = self._make_adapter()

        class _CardAPI:
            def create(self, request):
                return SimpleNamespace(code=300001, msg="invalid card data", data=None)

        adapter._client = SimpleNamespace(
            cardkit=SimpleNamespace(
                v1=SimpleNamespace(card=_CardAPI())
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send_streaming_card(chat_id="oc_chat", content="test")
            )
        assert not result.success
        assert "CardKit create failed" in result.error

    def test_edit_message_routes_to_streaming_card_when_active(self):
        adapter = self._make_adapter()
        captured = self._mock_cardkit_and_im(adapter)

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            # First create the streaming card
            asyncio.run(adapter.send_streaming_card(chat_id="oc_chat"))
            # Now edit — should go through CardKit streaming API
            result = asyncio.run(
                adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_456",
                    content="streamed text update",
                )
            )

        assert result.success
        assert result.message_id == "om_456"
        # Verify CardKit content API was called
        content_req = captured["cardkit_content"]
        assert content_req.card_id == "card_123"
        assert content_req.request_body.content == "streamed text update"
        assert content_req.request_body.sequence == 2  # incremented from 1

    def test_edit_message_falls_through_to_regular_path_when_no_streaming_card(self):
        adapter = self._make_adapter()
        captured = self._mock_cardkit_and_im(adapter)

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_regular",
                    content="regular edit",
                )
            )

        assert result.success
        # Should have used IM update, not CardKit
        assert captured.get("cardkit_content") is None

    def test_stop_streaming_card_disables_streaming_mode(self):
        adapter = self._make_adapter()
        captured = self._mock_cardkit_and_im(adapter)

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            # Create streaming card first
            asyncio.run(adapter.send_streaming_card(chat_id="oc_chat"))
            assert "om_456" in adapter._streaming_cards

            # Stop streaming
            ok = asyncio.run(adapter.stop_streaming_card("om_456"))

        assert ok is True
        # State is cleaned up
        assert "om_456" not in adapter._streaming_cards
        # Settings API was called
        settings_req = captured["cardkit_settings"]
        assert settings_req.card_id == "card_123"
        settings_json = json.loads(settings_req.request_body.settings)
        assert settings_json["config"]["streaming_mode"] is False

    def test_stop_streaming_card_noop_for_unknown_message(self):
        adapter = self._make_adapter()
        adapter._client = SimpleNamespace()  # minimal mock
        ok = asyncio.run(adapter.stop_streaming_card("om_unknown"))
        assert ok is True

    def test_streaming_cards_enabled_property(self):
        from gateway.platforms.feishu import FEISHU_CARDKIT_AVAILABLE

        adapter = self._make_adapter()
        adapter._client = None
        assert adapter.streaming_cards_enabled is False

        adapter._client = SimpleNamespace()
        expected = bool(FEISHU_CARDKIT_AVAILABLE)
        assert adapter.streaming_cards_enabled == expected

    def test_sequence_increments_across_operations(self):
        adapter = self._make_adapter()
        captured = self._mock_cardkit_and_im(adapter)

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            asyncio.run(adapter.send_streaming_card(chat_id="oc_chat"))
            # Do multiple updates
            asyncio.run(adapter.edit_message("oc_chat", "om_456", "text 1"))
            asyncio.run(adapter.edit_message("oc_chat", "om_456", "text 2"))
            asyncio.run(adapter.edit_message("oc_chat", "om_456", "text 3"))
            # Stop
            asyncio.run(adapter.stop_streaming_card("om_456"))

        # sequence should have been: 1 (create) → 2,3,4 (updates) → 5 (stop)
        assert captured["cardkit_settings"].request_body.sequence == 5


@patch.dict(os.environ, {}, clear=True)
class TestStreamConsumerFeishuIntegration:
    """Test GatewayStreamConsumer with Feishu streaming cards."""

    @pytest.mark.asyncio
    async def test_consumer_uses_streaming_card_when_enabled(self):
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = AsyncMock()
        adapter.streaming_cards_enabled = True
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send_streaming_card.return_value = SimpleNamespace(
            success=True, message_id="om_stream"
        )
        adapter.edit_message.return_value = SimpleNamespace(
            success=True, message_id="om_stream"
        )
        adapter.stop_streaming_card.return_value = True

        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="oc_test",
            config=StreamConsumerConfig(edit_interval=0.01),
        )

        assert consumer._uses_streaming_card is True

        consumer.on_delta("Hello ")
        consumer.on_delta("World!")
        consumer.finish()
        await consumer.run()

        # Should have used send_streaming_card, not send
        adapter.send_streaming_card.assert_called()
        adapter.send.assert_not_called()
        # Should have called stop_streaming_card at the end
        adapter.stop_streaming_card.assert_called_once_with("om_stream")

    @pytest.mark.asyncio
    async def test_consumer_does_not_add_cursor_for_streaming_cards(self):
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = AsyncMock()
        adapter.streaming_cards_enabled = True
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send_streaming_card.return_value = SimpleNamespace(
            success=True, message_id="om_stream"
        )
        adapter.edit_message.return_value = SimpleNamespace(
            success=True, message_id="om_stream"
        )
        adapter.stop_streaming_card.return_value = True

        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="oc_test",
            config=StreamConsumerConfig(edit_interval=0.01, cursor=" ▌"),
        )

        consumer.on_delta("Hello")
        await asyncio.sleep(0.05)
        consumer.finish()
        await consumer.run()

        # Verify cursor was NOT added to any call
        for call in adapter.edit_message.call_args_list:
            content = call.kwargs.get("content", call.args[2] if len(call.args) > 2 else "")
            assert "▌" not in content

    @pytest.mark.asyncio
    async def test_consumer_falls_back_when_streaming_not_enabled(self):
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = AsyncMock()
        adapter.streaming_cards_enabled = False
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send.return_value = SimpleNamespace(
            success=True, message_id="om_regular"
        )
        adapter.edit_message.return_value = SimpleNamespace(
            success=True, message_id="om_regular"
        )

        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="oc_test",
            config=StreamConsumerConfig(edit_interval=0.01),
        )

        assert consumer._uses_streaming_card is False

        consumer.on_delta("Hello")
        consumer.finish()
        await consumer.run()

        # Should use regular send, not streaming card
        adapter.send.assert_called()
        adapter.send_streaming_card.assert_not_called()

    @pytest.mark.asyncio
    async def test_consumer_stops_streaming_card_on_segment_break(self):
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = AsyncMock()
        adapter.streaming_cards_enabled = True
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send_streaming_card.return_value = SimpleNamespace(
            success=True, message_id="om_stream"
        )
        adapter.edit_message.return_value = SimpleNamespace(
            success=True, message_id="om_stream"
        )
        adapter.stop_streaming_card.return_value = True

        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="oc_test",
            config=StreamConsumerConfig(edit_interval=0.01),
        )

        consumer.on_delta("First segment")
        await asyncio.sleep(0.05)
        consumer.on_delta(None)  # Tool boundary / segment break
        await asyncio.sleep(0.05)
        consumer.finish()
        await consumer.run()

        # Should have called stop at least once for the segment break
        assert adapter.stop_streaming_card.call_count >= 1
