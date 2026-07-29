"""Tests for Feishu CardKit streaming card support.

Covers:
- _build_streaming_card_json(): card JSON structure
- send_streaming_card(): card creation + message send + state tracking
- send() gating: only streaming responses use CardKit path
- edit_message() routing to CardKit streaming API for active cards
- _update_streaming_card_content(): incremental content push
- stop_streaming_card(): streaming mode disable + state cleanup
- _cardkit_finalize(): final-state card rendering
- _cardkit_split_card(): element limit rollover
- REQUIRES_EDIT_FINALIZE: runtime-gated property
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
    from plugins.platforms.feishu.adapter import FeishuAdapter

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
            captured.setdefault("cardkit_settings", []).append(request)
            return SimpleNamespace(success=lambda: True)

    class _CardElementAPI:
        def content(self, request):
            captured["cardkit_content"] = request
            return SimpleNamespace(success=lambda: True)

    class _MessageAPI:
        def create(self, request):
            captured.setdefault("im_create", []).append(request)
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


def _patch_run_blocking():
    """Patch _run_blocking to call functions directly in tests."""
    return patch.object(
        type(_make_adapter()),
        "_run_blocking",
        new_callable=AsyncMock,
        side_effect=lambda func, *args: func(*args) if args else func(),
    )


@patch.dict(os.environ, {}, clear=True)
class TestBuildStreamingCardJson(unittest.TestCase):
    """Test the static card JSON builder."""

    def test_card_json_has_streaming_mode_enabled(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        card_str = FeishuAdapter._build_streaming_card_json("")
        card = json.loads(card_str)
        assert card["streaming_mode"] is True

    def test_card_json_has_streaming_config(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        card_str = FeishuAdapter._build_streaming_card_json("")
        card = json.loads(card_str)
        sc = card["streaming_config"]
        assert "print_frequency_ms" in sc
        assert "print_step" in sc
        assert sc["print_strategy"] in ("fast", "delay")

    def test_card_json_has_markdown_element_with_id(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter, _STREAMING_CARD_ELEMENT_ID

        card_str = FeishuAdapter._build_streaming_card_json("hello")
        card = json.loads(card_str)
        elements = card["body"]["elements"]
        assert len(elements) == 1
        assert elements[0]["tag"] == "markdown"
        assert elements[0]["content"] == "hello"
        assert elements[0]["element_id"] == _STREAMING_CARD_ELEMENT_ID

    def test_card_json_default_empty_content(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        card_str = FeishuAdapter._build_streaming_card_json("")
        card = json.loads(card_str)
        assert card["body"]["elements"][0]["content"] == ""

    def test_card_json_schema_2(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

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
            with _patch_run_blocking():
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

    def test_send_streaming_card_uses_card_json_type(self):
        """CardKit create must use type='card_json' and pass the full JSON."""
        adapter = _make_adapter()
        captured = _mock_cardkit_and_im(adapter)

        async def _run():
            with _patch_run_blocking():
                await adapter.send_streaming_card(chat_id="oc_chat", content="test")

        asyncio.run(_run())
        create_req = captured.get("cardkit_create")
        assert create_req is not None
        body = create_req.request_body
        assert body.type == "card_json"
        # The data field should be a JSON string containing streaming_mode
        card_data = json.loads(body.data)
        assert card_data["streaming_mode"] is True
        assert "streaming_config" in card_data

    def test_send_streaming_card_accepts_reply_to(self):
        adapter = _make_adapter()
        captured = _mock_cardkit_and_im(adapter)

        async def _run():
            with _patch_run_blocking():
                result = await adapter.send_streaming_card(
                    chat_id="oc_chat", content="test", reply_to="om_orig",
                )
            return result

        result = asyncio.run(_run())
        assert result.success

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
            with _patch_run_blocking():
                return await adapter.send_streaming_card(chat_id="oc_chat", content="test")

        result = asyncio.run(_run())
        assert not result.success


@patch.dict(os.environ, {}, clear=True)
class TestSendGating(unittest.TestCase):
    """Test that send() only uses CardKit for streaming responses."""

    def test_send_uses_cardkit_when_expect_edits(self):
        """send() should create a streaming card when metadata has expect_edits."""
        adapter = _make_adapter()
        captured = _mock_cardkit_and_im(adapter)

        async def _run():
            with _patch_run_blocking():
                result = await adapter.send(
                    chat_id="oc_chat",
                    content="streaming response",
                    metadata={"expect_edits": True},
                )
            return result

        result = asyncio.run(_run())
        assert result.success
        assert result.message_id == "om_456"
        # CardKit was used
        assert captured.get("cardkit_create") is not None

    def test_send_skips_cardkit_without_expect_edits(self):
        """send() should NOT use CardKit for non-streaming messages."""
        adapter = _make_adapter()
        captured = _mock_cardkit_and_im(adapter)

        async def _run():
            with _patch_run_blocking():
                result = await adapter.send(
                    chat_id="oc_chat",
                    content="ack message",
                    metadata={"notify": True},
                )
            return result

        result = asyncio.run(_run())
        # Should use regular IM path, not CardKit
        assert captured.get("cardkit_create") is None

    def test_send_skips_cardkit_without_metadata(self):
        """send() should NOT use CardKit when metadata is None."""
        adapter = _make_adapter()
        captured = _mock_cardkit_and_im(adapter)

        async def _run():
            with _patch_run_blocking():
                result = await adapter.send(
                    chat_id="oc_chat",
                    content="no metadata",
                )
            return result

        result = asyncio.run(_run())
        assert captured.get("cardkit_create") is None


@patch.dict(os.environ, {}, clear=True)
class TestRequiresEditFinalize(unittest.TestCase):
    """Test REQUIRES_EDIT_FINALIZE property gating."""

    def test_requires_edit_finalize_true_when_streaming_enabled(self):
        adapter = _make_adapter()
        _mock_cardkit_and_im(adapter)
        assert adapter.REQUIRES_EDIT_FINALIZE is True

    def test_requires_edit_finalize_false_when_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None
        assert adapter.REQUIRES_EDIT_FINALIZE is False

    def test_requires_edit_finalize_false_when_not_configured(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        config = PlatformConfig()  # no streaming_card in extra
        adapter = FeishuAdapter(config)
        adapter._client = SimpleNamespace()
        assert adapter.REQUIRES_EDIT_FINALIZE is False


@patch.dict(os.environ, {}, clear=True)
class TestStreamingCardEditing(unittest.TestCase):
    """Test edit_message routing to CardKit streaming API."""

    def test_edit_message_routes_to_cardkit_when_active(self):
        adapter = _make_adapter()
        captured = _mock_cardkit_and_im(adapter)

        async def _run():
            with _patch_run_blocking():
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
            with _patch_run_blocking():
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
            with _patch_run_blocking():
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
            with _patch_run_blocking():
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

    def test_stop_streaming_card_on_disconnect(self):
        adapter = _make_adapter()
        _mock_cardkit_and_im(adapter)

        async def _run():
            with _patch_run_blocking():
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
            with _patch_run_blocking():
                result = await adapter.send(
                    chat_id="oc_chat",
                    content="hello",
                    metadata={"expect_edits": True},
                )
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
        from plugins.platforms.feishu.adapter import FeishuAdapter

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
            with _patch_run_blocking():
                await adapter.send_streaming_card(chat_id="oc_chat", content="")
            return adapter._streaming_cards

        cards = asyncio.run(_run())
        state = cards.get("om_456")
        assert state is not None
        assert state.element_count == 1

    def test_element_limit_constant(self):
        from plugins.platforms.feishu.adapter import _STREAMING_CARD_ELEMENT_LIMIT

        assert _STREAMING_CARD_ELEMENT_LIMIT == 180

    def test_split_card_method_exists(self):
        """_cardkit_split_card method should exist and be callable."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        assert hasattr(FeishuAdapter, "_cardkit_split_card")

    def test_streaming_params_match_plugin_v11(self):
        """Streaming params should match hermes-lark-streaming plugin v0.11."""
        from plugins.platforms.feishu.adapter import (
            _STREAMING_CARD_PRINT_FREQUENCY_MS,
            _STREAMING_CARD_PRINT_STEP,
            _STREAMING_CARD_PRINT_STRATEGY,
        )

        assert _STREAMING_CARD_PRINT_FREQUENCY_MS == 15
        assert _STREAMING_CARD_PRINT_STEP == 1
        assert _STREAMING_CARD_PRINT_STRATEGY == "fast"

    def test_transient_error_codes_defined(self):
        """Transient error code constants should be defined for retry logic."""
        from plugins.platforms.feishu.adapter import (
            _CARDKIT_TRANSIENT_ERROR_CODES,
            _CARDKIT_GATEWAY_TIMEOUT,
            _CARDKIT_INTERNAL_ERROR,
            _CARDKIT_SERVER_INTERNAL_ERROR,
            _CARDKIT_RETRY_DELAYS_SEC,
            _CARDKIT_FINALIZE_MAX_ATTEMPTS,
        )

        # Core transient codes
        assert _CARDKIT_GATEWAY_TIMEOUT in _CARDKIT_TRANSIENT_ERROR_CODES
        assert _CARDKIT_INTERNAL_ERROR in _CARDKIT_TRANSIENT_ERROR_CODES
        assert _CARDKIT_SERVER_INTERNAL_ERROR in _CARDKIT_TRANSIENT_ERROR_CODES
        # Retry config should be reasonable
        assert len(_CARDKIT_RETRY_DELAYS_SEC) >= 2
        assert all(d > 0 for d in _CARDKIT_RETRY_DELAYS_SEC)
        assert _CARDKIT_FINALIZE_MAX_ATTEMPTS >= 2


@patch.dict(os.environ, {}, clear=True)
class TestCardkitApiCallRetry(unittest.TestCase):
    """Test _cardkit_api_call transient error retry behavior."""

    def test_retry_on_transient_error_then_succeed(self):
        """Should retry on transient error code and succeed on next attempt."""
        from plugins.platforms.feishu.adapter import (
            _CARDKIT_GATEWAY_TIMEOUT,
        )

        adapter = _make_adapter()
        call_count = 0

        def _failing_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(
                    success=lambda: False,
                    code=_CARDKIT_GATEWAY_TIMEOUT,
                    msg="gateway timeout",
                )
            return SimpleNamespace(success=lambda: True)

        async def _run():
            with _patch_run_blocking(), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                return await adapter._cardkit_api_call("test_op", _failing_then_ok)

        result = asyncio.run(_run())
        assert result.success()
        assert call_count == 2  # first failed, second succeeded

    def test_no_retry_on_non_transient_error(self):
        """Non-transient errors should NOT be retried."""
        adapter = _make_adapter()
        call_count = 0

        def _permanent_fail():
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(
                success=lambda: False,
                code=99999,  # not a transient code
                msg="permanent failure",
            )

        async def _run():
            with _patch_run_blocking():
                return await adapter._cardkit_api_call("test_op", _permanent_fail)

        result = asyncio.run(_run())
        assert not result.success()
        assert call_count == 1  # only called once, no retry

    def test_exhausted_retries_raises(self):
        """After exhausting all retries, should raise the last transient error."""
        from plugins.platforms.feishu.adapter import _CARDKIT_INTERNAL_ERROR

        adapter = _make_adapter()

        def _always_transient():
            return SimpleNamespace(
                success=lambda: False,
                code=_CARDKIT_INTERNAL_ERROR,
                msg="internal error",
            )

        async def _run():
            with _patch_run_blocking(), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RuntimeError, match="internal error"):
                    await adapter._cardkit_api_call("test_op", _always_transient)

        asyncio.run(_run())

    def test_success_first_call_no_retry(self):
        """Successful first call should not trigger any retry logic."""
        adapter = _make_adapter()
        call_count = 0

        def _immediate_success():
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(success=lambda: True)

        async def _run():
            with _patch_run_blocking():
                return await adapter._cardkit_api_call("test_op", _immediate_success)

        result = asyncio.run(_run())
        assert result.success()
        assert call_count == 1


@patch.dict(os.environ, {}, clear=True)
class TestCardkitFinalizeRetry(unittest.TestCase):
    """Test _cardkit_finalize retry on failure."""

    def test_finalize_retries_then_succeeds(self):
        """finalize should retry stop+update and eventually succeed."""
        adapter = _make_adapter()
        captured = _mock_cardkit_and_im(adapter)

        class _FlakyUpdateAPI:
            def __init__(self):
                self.attempts = 0

            def update(self, request):
                self.attempts += 1
                if self.attempts == 1:
                    return SimpleNamespace(
                        success=lambda: False,
                        code=2200,  # transient gateway timeout
                        msg="timeout",
                    )
                captured["cardkit_update"] = request
                return SimpleNamespace(success=lambda: True)

        adapter._client.cardkit.v1.card.update = _FlakyUpdateAPI().update

        async def _run():
            with _patch_run_blocking(), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                await adapter.send_streaming_card(chat_id="oc_chat", content="")
                await adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_456",
                    content="final",
                    finalize=True,
                )

        asyncio.run(_run())
        # State should be cleaned up after successful finalize
        assert "om_456" not in adapter._streaming_cards
