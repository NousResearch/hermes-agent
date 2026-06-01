"""Tests for Feishu CardKit streaming card support.

Covers:
- _build_streaming_card_json(): card JSON structure
- send_streaming_card(): card creation + message send + state tracking
- edit_message() routing to CardKit streaming API for active cards
- _update_streaming_card_content(): incremental content push
- stop_streaming_card(): streaming mode disable + state cleanup
- _cardkit_finalize(): final-state card rendering
- Fallback chain: CardKit failure → IM update path
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


def _make_adapter():
    """Create a minimal FeishuAdapter for testing (no connection)."""
    from gateway.config import PlatformConfig
    from gateway.platforms.feishu import FeishuAdapter

    config = PlatformConfig(extra={"streaming_card": True})
    adapter = FeishuAdapter(config)
    return adapter


def _mock_cardkit_and_im(adapter, card_id="card_123", message_id="om_456"):
    """Wire up mock client with cardkit + im APIs.

    Returns a dict of captured request objects for assertion.
    """
    captured = {}

    class _CardAPI:
        def create(self, request):
            captured["cardkit_create"] = request
            return SimpleNamespace(
                success=lambda: True,
                data=SimpleNamespace(card_id=card_id),
            )

        def update(self, request):
            captured["cardkit_update"] = request
            return SimpleNamespace(success=lambda: True)

        def settings(self, request):
            captured["cardkit_settings"] = request
            return SimpleNamespace(success=lambda: True)

    class _CardElementAPI:
        def content(self, request):
            captured["cardkit_content"] = request
            return SimpleNamespace(success=lambda: True)

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


def _direct_thread(func, *args, **kwargs):
    """Replace asyncio.to_thread so it calls functions directly in tests."""
    return func(*args, **kwargs)


@patch.dict(os.environ, {}, clear=True)
class TestBuildStreamingCardJson(unittest.TestCase):
    """Test the static card JSON builder."""

    def test_card_json_has_streaming_mode_enabled(self):
        from gateway.platforms.feishu import FeishuAdapter

        card_str = FeishuAdapter._build_streaming_card_json("")
        card = json.loads(card_str)
        assert card["streaming_mode"] is True

    def test_card_json_has_streaming_config(self):
        from gateway.platforms.feishu import FeishuAdapter

        card_str = FeishuAdapter._build_streaming_card_json("")
        card = json.loads(card_str)
        sc = card["streaming_config"]
        assert "print_frequency_ms" in sc
        assert "print_step" in sc
        assert sc["print_strategy"] in ("fast", "delay")

    def test_card_json_has_markdown_element_with_id(self):
        from gateway.platforms.feishu import FeishuAdapter, _STREAMING_CARD_ELEMENT_ID

        card_str = FeishuAdapter._build_streaming_card_json("hello")
        card = json.loads(card_str)
        elements = card["body"]["elements"]
        assert len(elements) == 1
        assert elements[0]["tag"] == "markdown"
        assert elements[0]["content"] == "hello"
        assert elements[0]["element_id"] == _STREAMING_CARD_ELEMENT_ID

    def test_card_json_default_empty_content(self):
        from gateway.platforms.feishu import FeishuAdapter

        card_str = FeishuAdapter._build_streaming_card_json("")
        card = json.loads(card_str)
        assert card["body"]["elements"][0]["content"] == ""

    def test_card_json_schema_2(self):
        from gateway.platforms.feishu import FeishuAdapter

        card_str = FeishuAdapter._build_streaming_card_json("")
        card = json.loads(card_str)
        assert card["schema"] == "2.0"


@patch.dict(os.environ, {}, clear=True)
class TestSendStreamingCard(unittest.TestCase):
    """Test send_streaming_card flow."""

    def test_send_streaming_card_creates_card_and_sends_message(self):
        adapter = _make_adapter()
        captured = _mock_cardkit_and_im(adapter)

        async def _run():
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct_thread):
                result = await adapter.send_streaming_card(
                    chat_id="oc_chat",
                    content="initial",
                )
            return result

        result = asyncio.run(_run())

        assert result.success
        assert result.message_id == "om_456"
        # Card entity was created
        assert captured.get("cardkit_create") is not None
        # Message was sent
        assert captured.get("im_create") is not None
        # Streaming card state is tracked
        assert "om_456" in adapter._streaming_cards
        sc = adapter._streaming_cards["om_456"]
        assert sc.card_id == "card_123"
        assert sc.sequence == 1

    def test_send_streaming_card_fails_when_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None

        async def _run():
            return await adapter.send_streaming_card(chat_id="oc_chat", content="test")

        result = asyncio.run(_run())
        assert not result.success
        assert "Not connected" in result.error

    def test_send_streaming_card_fails_on_card_create_error(self):
        adapter = _make_adapter()

        class _CardAPI:
            def create(self, request):
                return SimpleNamespace(
                    success=lambda: False,
                    msg="invalid card data",
                    data=None,
                )

        adapter._client = SimpleNamespace(
            cardkit=SimpleNamespace(v1=SimpleNamespace(card=_CardAPI()))
        )

        async def _run():
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct_thread):
                return await adapter.send_streaming_card(chat_id="oc_chat", content="test")

        result = asyncio.run(_run())
        assert not result.success


@patch.dict(os.environ, {}, clear=True)
class TestStreamingCardEditing(unittest.TestCase):
    """Test edit_message routing to CardKit streaming API."""

    def test_edit_message_routes_to_cardkit_when_active(self):
        adapter = _make_adapter()
        captured = _mock_cardkit_and_im(adapter)

        async def _run():
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct_thread):
                # First create the streaming card
                await adapter.send_streaming_card(chat_id="oc_chat", content="")
                # Now edit — should go through CardKit streaming API
                result = await adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_456",
                    content="streamed text update",
                )
            return result

        result = asyncio.run(_run())

        assert result.success
        assert result.message_id == "om_456"
        # Verify CardKit content API was called
        content_req = captured.get("cardkit_content")
        assert content_req is not None
        assert content_req.card_id == "card_123"
        assert content_req.request_body.content == "streamed text update"
        # Sequence should have incremented from 1 → 2
        assert adapter._streaming_cards["om_456"].sequence == 2

    def test_edit_message_falls_through_when_no_streaming_card(self):
        adapter = _make_adapter()
        captured = _mock_cardkit_and_im(adapter)

        async def _run():
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct_thread):
                result = await adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_regular",
                    content="regular edit",
                )
            return result

        result = asyncio.run(_run())

        assert result.success
        # Should NOT have called CardKit content API
        assert captured.get("cardkit_content") is None

    def test_sequence_increments_across_operations(self):
        adapter = _make_adapter()
        _mock_cardkit_and_im(adapter)

        async def _run():
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct_thread):
                await adapter.send_streaming_card(chat_id="oc_chat", content="")
                # Do multiple updates
                await adapter.edit_message("oc_chat", "om_456", "text 1")
                await adapter.edit_message("oc_chat", "om_456", "text 2")
                await adapter.edit_message("oc_chat", "om_456", "text 3")
            # sequence: 1 (create) → 2, 3, 4 (updates)
            return adapter._streaming_cards["om_456"].sequence

        seq = asyncio.run(_run())
        assert seq == 4


@patch.dict(os.environ, {}, clear=True)
class TestStreamingCardFinalization(unittest.TestCase):
    """Test stop_streaming_card and _cardkit_finalize."""

    def test_cardkit_finalize_stops_streaming_and_renders_final(self):
        adapter = _make_adapter()
        captured = _mock_cardkit_and_im(adapter)

        async def _run():
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct_thread):
                await adapter.send_streaming_card(chat_id="oc_chat", content="")
                assert "om_456" in adapter._streaming_cards

                # Finalize via edit_message(finalize=True)
                result = await adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_456",
                    content="final content",
                    finalize=True,
                )
            return result

        result = asyncio.run(_run())

        assert result.success
        # State should be cleaned up
        assert "om_456" not in adapter._streaming_cards
        # Settings API was called to stop streaming
        assert captured.get("cardkit_settings") is not None
        # Card update was called for final layout
        assert captured.get("cardkit_update") is not None
        # Verify final card has green header
        update_req = captured["cardkit_update"]
        final_card = json.loads(update_req.request_body.card)
        assert final_card["header"]["template"] == "green"

    def test_stop_streaming_card_on_disconnect(self):
        adapter = _make_adapter()
        _mock_cardkit_and_im(adapter)

        async def _run():
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct_thread):
                await adapter.send_streaming_card(chat_id="oc_chat", content="test")
                assert "om_456" in adapter._streaming_cards

                # Simulate disconnect cleanup
                for _msg_id, _card_state in list(adapter._streaming_cards.items()):
                    await adapter.stop_streaming_card(_card_state)
                adapter._streaming_cards.clear()

        asyncio.run(_run())
        assert len(adapter._streaming_cards) == 0


@patch.dict(os.environ, {}, clear=True)
class TestStreamingCardFallback(unittest.TestCase):
    """Test that send() falls back to IM when CardKit fails."""

    def test_send_falls_back_to_im_when_cardkit_unavailable(self):
        adapter = _make_adapter()
        # streaming_card setting is True, but client has no cardkit attribute
        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=SimpleNamespace(
                        create=lambda req: SimpleNamespace(
                            success=lambda: True,
                            data=SimpleNamespace(message_id="om_fallback"),
                        )
                    )
                )
            )
        )

        async def _run():
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct_thread):
                result = await adapter.send(chat_id="oc_chat", content="hello")
            return result

        result = asyncio.run(_run())
        # Should have fallen back to regular IM send
        assert result.success

    def test_streaming_cards_enabled_false_when_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None
        # streaming_card is True in settings but client is None
        assert adapter.streaming_cards_enabled is False

    def test_streaming_cards_enabled_false_by_default(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        config = PlatformConfig()  # no streaming_card in extra
        adapter = FeishuAdapter(config)
        adapter._client = SimpleNamespace()
        assert adapter.streaming_cards_enabled is False


@patch.dict(os.environ, {}, clear=True)
class TestStreamingCardElementLimit(unittest.TestCase):
    """Test element count tracking and split logic."""

    def test_element_count_starts_at_1(self):
        adapter = _make_adapter()
        _mock_cardkit_and_im(adapter)

        async def _run():
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct_thread):
                await adapter.send_streaming_card(chat_id="oc_chat", content="")
            return adapter._streaming_cards

        cards = asyncio.run(_run())
        state = cards.get("om_456")
        assert state is not None
        assert state.element_count == 1

    def test_element_limit_constant(self):
        from gateway.platforms.feishu import _STREAMING_CARD_ELEMENT_LIMIT

        assert _STREAMING_CARD_ELEMENT_LIMIT == 180
