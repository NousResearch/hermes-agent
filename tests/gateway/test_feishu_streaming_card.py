"""Unit tests for Feishu CardKit v1 streaming card lifecycle.

Covers:
- Phase state machine transitions (happy + invalid)
- Card TTL rotation
- Fallback from streaming card to post/text
- Content truncation and card JSON structure
"""

from __future__ import annotations

import json
import time
import uuid
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.feishu import (
    _CardPhase,
    _FEISHU_CLOSED_CARD_CACHE_SIZE,
    _PHASE_TRANSITIONS,
    _STREAM_CARD_TTL_SECONDS,
    FeishuAdapter,
    SendResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(*, streaming_card_enabled: bool = True) -> FeishuAdapter:
    """Create a bare FeishuAdapter with streaming card state initialised."""
    adapter = object.__new__(FeishuAdapter)
    adapter._client = MagicMock()
    adapter._bot_name = "TestBot"
    adapter._streaming_card_enabled = streaming_card_enabled
    adapter._streaming_cards: Dict[str, Dict[str, Dict[str, Any]]] = {}
    adapter._closed_streaming_card_ids: "OrderedDict[str, None]" = OrderedDict()
    adapter._card_phases: Dict[str, int] = {}
    adapter._running = True
    adapter.MAX_MESSAGE_LENGTH = 8000
    return adapter


def _direct(func, *args, **kwargs):
    """Bypass asyncio.to_thread — call the sync function directly."""
    return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# 1. Phase state machine
# ---------------------------------------------------------------------------

class TestPhaseStateMachine:
    """Tests for _transition_card_phase / _card_phase."""

    def test_idle_to_creating(self):
        adapter = _make_adapter()
        assert adapter._transition_card_phase("chat_1", _CardPhase.CREATING) is True
        assert adapter._card_phase("chat_1") == _CardPhase.CREATING

    def test_creating_to_streaming(self):
        adapter = _make_adapter()
        adapter._card_phases["chat_1"] = _CardPhase.CREATING
        assert adapter._transition_card_phase("chat_1", _CardPhase.STREAMING) is True
        assert adapter._card_phase("chat_1") == _CardPhase.STREAMING

    def test_streaming_to_completed(self):
        adapter = _make_adapter()
        adapter._card_phases["chat_1"] = _CardPhase.STREAMING
        assert adapter._transition_card_phase("chat_1", _CardPhase.COMPLETED) is True

    def test_streaming_to_aborted(self):
        adapter = _make_adapter()
        adapter._card_phases["chat_1"] = _CardPhase.STREAMING
        assert adapter._transition_card_phase("chat_1", _CardPhase.ABORTED) is True

    def test_completed_to_idle(self):
        adapter = _make_adapter()
        adapter._card_phases["chat_1"] = _CardPhase.COMPLETED
        assert adapter._transition_card_phase("chat_1", _CardPhase.IDLE) is True
        assert adapter._card_phase("chat_1") == _CardPhase.IDLE

    def test_aborted_to_idle(self):
        adapter = _make_adapter()
        adapter._card_phases["chat_1"] = _CardPhase.ABORTED
        assert adapter._transition_card_phase("chat_1", _CardPhase.IDLE) is True

    def test_creation_failed_to_idle(self):
        adapter = _make_adapter()
        adapter._card_phases["chat_1"] = _CardPhase.CREATION_FAILED
        assert adapter._transition_card_phase("chat_1", _CardPhase.IDLE) is True

    def test_terminated_to_idle(self):
        adapter = _make_adapter()
        adapter._card_phases["chat_1"] = _CardPhase.TERMINATED
        assert adapter._transition_card_phase("chat_1", _CardPhase.IDLE) is True

    def test_invalid_idle_to_streaming(self):
        adapter = _make_adapter()
        assert adapter._transition_card_phase("chat_1", _CardPhase.STREAMING) is False
        assert adapter._card_phase("chat_1") == _CardPhase.IDLE

    def test_invalid_double_creating(self):
        adapter = _make_adapter()
        adapter._card_phases["chat_1"] = _CardPhase.CREATING
        assert adapter._transition_card_phase("chat_1", _CardPhase.CREATING) is False

    def test_invalid_streaming_to_creating(self):
        adapter = _make_adapter()
        adapter._card_phases["chat_1"] = _CardPhase.STREAMING
        assert adapter._transition_card_phase("chat_1", _CardPhase.CREATING) is False

    def test_invalid_completed_to_streaming(self):
        adapter = _make_adapter()
        adapter._card_phases["chat_1"] = _CardPhase.COMPLETED
        assert adapter._transition_card_phase("chat_1", _CardPhase.STREAMING) is False

    def test_default_phase_is_idle(self):
        adapter = _make_adapter()
        assert adapter._card_phase("unknown_chat") == _CardPhase.IDLE


# ---------------------------------------------------------------------------
# 2. Card TTL rotation
# ---------------------------------------------------------------------------

class TestCardTTLRotation:
    """Tests for _is_card_expired and TTL-based card rotation."""

    def test_card_not_expired(self):
        adapter = _make_adapter()
        adapter._streaming_cards["chat_1"] = {
            "card_abc": {"created_at": time.monotonic()}
        }
        assert adapter._is_card_expired("chat_1") is False

    def test_card_expired(self):
        adapter = _make_adapter()
        adapter._streaming_cards["chat_1"] = {
            "card_abc": {"created_at": time.monotonic() - _STREAM_CARD_TTL_SECONDS - 10}
        }
        assert adapter._is_card_expired("chat_1") is True

    def test_no_card(self):
        adapter = _make_adapter()
        assert adapter._is_card_expired("chat_1") is False

    def test_empty_card_state(self):
        adapter = _make_adapter()
        adapter._streaming_cards["chat_1"] = {
            "card_abc": {"created_at": 0}
        }
        assert adapter._is_card_expired("chat_1") is False

    @pytest.mark.asyncio
    async def test_ttl_rotation_closes_old_card(self):
        adapter = _make_adapter()
        old_msg_id = "om_old_msg"
        old_card_state = {
            "card_id": "card_old",
            "message_id": old_msg_id,
            "sequence": 5,
            "last_content": "old text",
            "created_at": time.monotonic() - _STREAM_CARD_TTL_SECONDS - 10,
        }
        adapter._streaming_cards["chat_1"] = {
            "card_old": old_card_state,
            old_msg_id: old_card_state,
        }

        closed_cards: list = []

        async def _fake_close_siblings(chat_id):
            closed_cards.append(chat_id)
            adapter._streaming_cards.pop(chat_id, None)

        adapter._close_streaming_siblings = _fake_close_siblings

        settings_resp = SimpleNamespace(success=lambda: True)
        create_resp = SimpleNamespace(success=lambda: True, data=SimpleNamespace(card_id="card_new"))
        msg_resp = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_new_msg"))

        def _fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("gateway.platforms.feishu.HAS_CARDKIT", True),
            patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_fake_to_thread),
            patch("gateway.platforms.feishu.Config") as mock_config,
            patch("gateway.platforms.feishu.Settings") as mock_settings_cls,
            patch("gateway.platforms.feishu.SettingsCardRequest") as mock_settings_req_cls,
            patch("gateway.platforms.feishu.SettingsCardRequestBody") as mock_settings_body_cls,
            patch("gateway.platforms.feishu.CreateCardRequest") as mock_create_req_cls,
            patch("gateway.platforms.feishu.CreateCardRequestBody") as mock_create_body_cls,
            patch("gateway.platforms.feishu.UpdateCardRequest"),
            patch("gateway.platforms.feishu.UpdateCardRequestBody"),
            patch("gateway.platforms.feishu.CardKitCard"),
        ):
            mock_settings_body_cls.builder.return_value \
                .settings.return_value \
                .sequence.return_value \
                .uuid.return_value \
                .build.return_value = MagicMock()
            mock_settings_req_cls.builder.return_value \
                .request_body.return_value \
                .card_id.return_value \
                .build.return_value = MagicMock()
            adapter._client.cardkit.v1.card.settings = MagicMock(return_value=settings_resp)
            adapter._client.cardkit.v1.card.update = MagicMock(return_value=SimpleNamespace(success=lambda: True))

            mock_create_body_cls.builder.return_value \
                .type.return_value \
                .data.return_value \
                .build.return_value = MagicMock()
            mock_create_req_cls.builder.return_value \
                .request_body.return_value \
                .build.return_value = MagicMock()
            adapter._client.cardkit.v1.card.create = MagicMock(return_value=create_resp)

            adapter._feishu_send_with_retry = AsyncMock(return_value=msg_resp)

            result = await adapter._send_streaming_card(
                chat_id="chat_1", content="new content"
            )

        # _close_streaming_siblings was called to close the expired card.
        assert "chat_1" in closed_cards
        assert result is not None
        assert result.success is True
        # New card tracked.
        assert "card_new" in adapter._streaming_cards.get("chat_1", {})


# ---------------------------------------------------------------------------
# 3. Streaming card fallback
# ---------------------------------------------------------------------------

class TestStreamingCardFallback:
    """Tests for graceful degradation when streaming card path fails."""

    @pytest.mark.asyncio
    async def test_send_returns_none_when_no_cardkit(self):
        adapter = _make_adapter()
        with patch("gateway.platforms.feishu.HAS_CARDKIT", False):
            result = await adapter._send_streaming_card(
                chat_id="chat_1", content="hello"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_send_returns_none_when_no_client(self):
        adapter = _make_adapter()
        adapter._client = None
        with patch("gateway.platforms.feishu.HAS_CARDKIT", True):
            result = await adapter._send_streaming_card(
                chat_id="chat_1", content="hello"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_send_returns_none_on_create_failure(self):
        adapter = _make_adapter()
        adapter._client.cardkit.v1.card.create = MagicMock(
            return_value=SimpleNamespace(success=lambda: False, msg="forbidden")
        )
        adapter._feishu_send_with_retry = AsyncMock()

        def _fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("gateway.platforms.feishu.HAS_CARDKIT", True),
            patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_fake_to_thread),
            patch("gateway.platforms.feishu.CreateCardRequest") as mock_req_cls,
            patch("gateway.platforms.feishu.CreateCardRequestBody") as mock_body_cls,
        ):
            mock_body_cls.builder.return_value.type.return_value.data.return_value.build.return_value = MagicMock()
            mock_req_cls.builder.return_value.request_body.return_value.build.return_value = MagicMock()
            result = await adapter._send_streaming_card(
                chat_id="chat_1", content="hello"
            )

        assert result is None
        # Phase should have been reset to IDLE.
        assert adapter._card_phase("chat_1") == _CardPhase.IDLE

    @pytest.mark.asyncio
    async def test_edit_card_already_closed_finalize_returns_success(self):
        adapter = _make_adapter()
        card_id = "card_closed_1"
        adapter._closed_streaming_card_ids[card_id] = None
        # No streaming card state — edit_message should return success for known closed cards.
        adapter.format_message = lambda c: c
        adapter.truncate_message = lambda c, _: [c]
        with (
            patch("gateway.platforms.feishu.HAS_CARDKIT", True),
        ):
            result = await adapter.edit_message(
                "chat_1", card_id, "final content", finalize=True
            )
        assert result.success is True
        assert result.message_id == card_id

    @pytest.mark.asyncio
    async def test_send_falls_through_to_post_text(self):
        adapter = _make_adapter()

        # _send_streaming_card returns None (simulate failure).
        with (
            patch("gateway.platforms.feishu.HAS_CARDKIT", False),
        ):
            result = await adapter._send_streaming_card(
                chat_id="chat_1", content="hello"
            )
        # Caller (send()) should interpret None as "fall back to post/text".
        assert result is None


# ---------------------------------------------------------------------------
# 4. Content truncation and card JSON structure
# ---------------------------------------------------------------------------

class TestCardJsonStructure:
    """Tests for _build_streaming_card_json and _build_finalized_header_json."""

    def test_streaming_card_json_schema(self):
        raw = FeishuAdapter._build_streaming_card_json(
            "Hello world", bot_name="MyBot"
        )
        card = json.loads(raw)

        assert card["schema"] == "2.0"
        assert card["config"]["streaming_mode"] is True
        assert card["header"]["title"]["content"] == "\U0001f916 MyBot"
        assert card["header"]["subtitle"]["content"] == "正在思考..."
        assert card["body"]["elements"][0]["content"] == "Hello world"

    def test_preview_strips_markdown(self):
        raw = FeishuAdapter._build_streaming_card_json(
            "# Title\n**bold** and `code` and [link](url)",
            bot_name="Bot",
        )
        card = json.loads(raw)
        summary = card["config"]["summary"]["content"]
        # No markdown chars should remain.
        assert "#" not in summary
        assert "*" not in summary
        assert "`" not in summary
        assert "[" not in summary
        # Should be a plain text preview.
        assert "Title" in summary
        assert "bold" in summary

    def test_preview_truncates_to_60_chars(self):
        long_text = "A" * 200
        raw = FeishuAdapter._build_streaming_card_json(long_text)
        card = json.loads(raw)
        assert len(card["config"]["summary"]["content"]) <= 60

    def test_finalized_header_completed(self):
        raw = FeishuAdapter._build_finalized_header_json(
            bot_name="Bot",
            status="completed",
            last_content="some content",
        )
        card = json.loads(raw)
        assert card["config"]["streaming_mode"] is False
        assert card["header"]["template"] == "blue"
        assert card["header"]["subtitle"]["content"] == ""
        assert card["body"]["elements"][0]["content"] == "some content"
        # Completed should have only [AI] tag, no status tag.
        assert len(card["header"]["text_tag_list"]) == 1

    def test_finalized_header_error(self):
        raw = FeishuAdapter._build_finalized_header_json(
            bot_name="Bot",
            status="error",
            error_summary="API timeout",
            last_content="partial text",
        )
        card = json.loads(raw)
        assert card["header"]["template"] == "grey"
        assert card["header"]["subtitle"]["content"] == "API timeout"
        assert len(card["header"]["text_tag_list"]) == 2
        assert card["body"]["elements"][0]["content"] == "partial text"

    def test_finalized_header_aborted(self):
        raw = FeishuAdapter._build_finalized_header_json(
            bot_name="Bot",
            status="aborted",
        )
        card = json.loads(raw)
        assert card["header"]["template"] == "grey"
        assert card["header"]["subtitle"]["content"] == ""
        tags = [t["text"]["tag"] == "plain_text" for t in card["header"]["text_tag_list"]]
        assert any(tags)

    def test_finalized_header_preserves_body_content(self):
        raw = FeishuAdapter._build_finalized_header_json(
            bot_name="Bot",
            status="completed",
            last_content="The actual streamed content",
        )
        card = json.loads(raw)
        assert card["body"]["elements"][0]["content"] == "The actual streamed content"

    def test_streaming_card_default_bot_name(self):
        raw = FeishuAdapter._build_streaming_card_json("hi", bot_name="")
        card = json.loads(raw)
        # Empty string falls back to "Hermes".
        assert "Hermes" in card["header"]["title"]["content"]

    def test_closed_card_cache_eviction(self):
        adapter = _make_adapter()
        # Simulate filling cache via the close path (which calls popitem).
        for i in range(_FEISHU_CLOSED_CARD_CACHE_SIZE + 50):
            adapter._closed_streaming_card_ids[f"card_{i}"] = None
            # Eviction logic from _close_streaming_card.
            while len(adapter._closed_streaming_card_ids) > _FEISHU_CLOSED_CARD_CACHE_SIZE:
                adapter._closed_streaming_card_ids.popitem(last=False)
        assert len(adapter._closed_streaming_card_ids) == _FEISHU_CLOSED_CARD_CACHE_SIZE
        assert f"card_{_FEISHU_CLOSED_CARD_CACHE_SIZE + 49}" in adapter._closed_streaming_card_ids


# ---------------------------------------------------------------------------
# 5. Final-state footer rendering
# ---------------------------------------------------------------------------

class TestFinalStateFooter:
    """Tests for elapsed-time footer in finalized card."""

    def test_footer_included_on_completed(self):
        raw = FeishuAdapter._build_finalized_header_json(
            bot_name="Bot",
            status="completed",
            last_content="result text",
            elapsed_seconds=12.3,
        )
        card = json.loads(raw)
        elements = card["body"]["elements"]
        assert len(elements) == 2
        footer = elements[1]["content"]
        assert "12.3s" in footer
        assert "<font color='grey'>" in footer
        assert "<text_align" not in footer
        assert elements[0]["content"] == "result text"

    def test_no_footer_on_zero_elapsed(self):
        raw = FeishuAdapter._build_finalized_header_json(
            bot_name="Bot",
            status="completed",
            last_content="result",
            elapsed_seconds=0,
        )
        card = json.loads(raw)
        assert len(card["body"]["elements"]) == 1

    def test_no_footer_on_error(self):
        raw = FeishuAdapter._build_finalized_header_json(
            bot_name="Bot",
            status="error",
            last_content="partial",
            elapsed_seconds=5.0,
        )
        card = json.loads(raw)
        assert len(card["body"]["elements"]) == 1

    def test_no_footer_on_aborted(self):
        raw = FeishuAdapter._build_finalized_header_json(
            bot_name="Bot",
            status="aborted",
            elapsed_seconds=3.0,
        )
        card = json.loads(raw)
        assert len(card["body"]["elements"]) == 1
