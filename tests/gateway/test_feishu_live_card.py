"""Tests for Feishu Live Progress Card."""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestCardElementValidator:
    """Task 1-3: CardElementValidator limits."""

    def test_short_markdown_unchanged(self):
        from gateway.platforms.feishu_card import CardElementValidator
        elements = [{"tag": "markdown", "content": "hello"}]
        result = CardElementValidator.validate(elements)
        assert result == elements

    def test_long_markdown_split_by_paragraphs(self):
        from gateway.platforms.feishu_card import CardElementValidator
        long_text = "\n\n".join([f"paragraph {i}" for i in range(50)])
        elements = [{"tag": "markdown", "content": long_text}]
        result = CardElementValidator.validate(elements)
        assert all(len(e["content"]) <= 4000 for e in result if e["tag"] == "markdown")
        rejoined = "\n\n".join(e["content"] for e in result if e["tag"] == "markdown")
        assert rejoined == long_text

    def test_unsplittable_long_block_hard_truncated(self):
        from gateway.platforms.feishu_card import CardElementValidator
        huge = "x" * 5000
        elements = [{"tag": "markdown", "content": huge}]
        result = CardElementValidator.validate(elements)
        assert len(result[0]["content"]) <= 4000
        assert result[0]["content"].endswith("...(内容过长已截断)")

    def test_tables_within_limit_unchanged(self):
        from gateway.platforms.feishu_card import CardElementValidator
        table = {"tag": "table", "columns": [{"name": "a", "display_name": "A"}], "rows": [{"a": "1"}]}
        elements = [dict(table) for _ in range(5)]
        result = CardElementValidator.validate(elements)
        assert sum(1 for e in result if e["tag"] == "table") == 5

    def test_excess_tables_converted_to_markdown(self):
        from gateway.platforms.feishu_card import CardElementValidator
        table = {"tag": "table", "columns": [{"name": "a", "display_name": "A"}], "rows": [{"a": "1"}]}
        elements = [dict(table) for _ in range(7)]
        result = CardElementValidator.validate(elements)
        assert sum(1 for e in result if e["tag"] == "table") <= 5

    def test_elements_merged_when_over_limit(self):
        from gateway.platforms.feishu_card import CardElementValidator
        elements = [{"tag": "markdown", "content": f"line {i}"} for i in range(35)]
        result = CardElementValidator.validate(elements)
        assert len(result) <= 30

    def test_total_bytes_truncated(self):
        from gateway.platforms.feishu_card import CardElementValidator
        big = "A" * 20000
        elements = [{"tag": "markdown", "content": big}, {"tag": "markdown", "content": big}]
        result = CardElementValidator.validate(elements)
        total = len(json.dumps(result, ensure_ascii=False).encode())
        assert total <= 24000

    def test_non_markdown_elements_preserved(self):
        from gateway.platforms.feishu_card import CardElementValidator
        elements = [
            {"tag": "hr"},
            {"tag": "markdown", "content": "text"},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "footer"}]},
        ]
        result = CardElementValidator.validate(elements)
        assert result[0]["tag"] == "hr"
        assert result[-1]["tag"] == "note"


class TestBuildProgressCardJson:
    """Task 4: build_progress_card_json."""

    def test_thinking_only(self):
        from gateway.platforms.feishu_card import build_progress_card_json
        card = build_progress_card_json(
            accumulated_text="",
            tool_lines=[],
            status_line="⏳ 已思考 5s",
        )
        assert card["config"]["update_multi"] is True
        md_elements = [e for e in card["elements"] if e["tag"] == "markdown"]
        assert any("已思考 5s" in e["content"] for e in md_elements)

    def test_with_text_and_tools(self):
        from gateway.platforms.feishu_card import build_progress_card_json
        card = build_progress_card_json(
            accumulated_text="Here is my analysis...",
            tool_lines=["📖 阅读文件", "💻 执行命令"],
            status_line="⏳ 已思考 12s · 执行命令",
        )
        md = "\n".join(e["content"] for e in card["elements"] if e["tag"] == "markdown")
        assert "Here is my analysis" in md
        assert "📖 阅读文件" in md
        assert "已思考 12s" in md

    def test_tool_chain_separator(self):
        from gateway.platforms.feishu_card import build_progress_card_json
        card = build_progress_card_json(
            accumulated_text="",
            tool_lines=["📖 阅读文件", "💻 执行命令", "🔍 搜索代码"],
            status_line="⏳ 已思考 8s",
        )
        md = "\n".join(e["content"] for e in card["elements"] if e["tag"] == "markdown")
        assert "→" in md

    def test_validation_applied(self):
        from gateway.platforms.feishu_card import build_progress_card_json
        huge_text = "x" * 5000
        card = build_progress_card_json(
            accumulated_text=huge_text,
            tool_lines=[],
            status_line="⏳ 已思考 3s",
        )
        for e in card["elements"]:
            if e["tag"] == "markdown":
                assert len(e["content"]) <= 4000


class TestLiveCardManager:
    """Tasks 5-7: LiveCardManager state machine."""

    def test_initial_state_is_idle(self):
        from gateway.platforms.feishu import LiveCardManager, LiveCardState
        mgr = LiveCardManager()
        assert mgr.state == LiveCardState.IDLE

    def test_start_sets_ack_sent(self):
        from gateway.platforms.feishu import LiveCardManager, LiveCardState
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        assert mgr.state == LiveCardState.ACK_SENT
        assert mgr.card_message_id == "msg_001"
        assert mgr.started_at == 100.0

    def test_update_text_transitions_to_live(self):
        from gateway.platforms.feishu import LiveCardManager, LiveCardState
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        mgr.update_text("hello")
        assert mgr.state == LiveCardState.LIVE
        assert mgr.accumulated_text == "hello"

    def test_append_tool_line(self):
        from gateway.platforms.feishu import LiveCardManager, LiveCardState
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        mgr.append_tool("Read")
        assert mgr.state == LiveCardState.LIVE
        assert len(mgr.tool_lines) == 1
        assert "阅读文件" in mgr.tool_lines[0]
        assert mgr.last_tool == "Read"

    def test_append_unknown_tool(self):
        from gateway.platforms.feishu import LiveCardManager
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        mgr.append_tool("CustomTool")
        assert mgr.last_tool == "CustomTool"
        assert len(mgr.tool_lines) == 1

    def test_reset_clears_all(self):
        from gateway.platforms.feishu import LiveCardManager, LiveCardState
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        mgr.update_text("text")
        mgr.append_tool("Bash")
        mgr.reset()
        assert mgr.state == LiveCardState.IDLE
        assert mgr.accumulated_text == ""
        assert mgr.tool_lines == []
        assert mgr.card_message_id is None

    def test_reset_cancels_heartbeat(self):
        from gateway.platforms.feishu import LiveCardManager
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        mock_task = Mock()
        mock_task.cancel = Mock()
        mgr.heartbeat_task = mock_task
        mgr.reset()
        mock_task.cancel.assert_called_once()
        assert mgr.heartbeat_task is None

    def test_build_card_ack_state(self):
        from gateway.platforms.feishu import LiveCardManager
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        card = mgr.build_card(now=105.0)
        md = "\n".join(e["content"] for e in card["elements"] if e["tag"] == "markdown")
        assert "已思考 5s" in md

    def test_build_card_live_state_with_text(self):
        from gateway.platforms.feishu import LiveCardManager
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        mgr.update_text("analysis result")
        mgr.append_tool("Read")
        card = mgr.build_card(now=112.0)
        md = "\n".join(e["content"] for e in card["elements"] if e["tag"] == "markdown")
        assert "analysis result" in md
        assert "阅读文件" in md
        assert "12s" in md

    def test_should_throttle_within_interval(self):
        from gateway.platforms.feishu import LiveCardManager, MIN_PATCH_INTERVAL
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=0.0)
        mgr.last_patch_ts = 10.0
        assert mgr.should_throttle(now=10.5) is True
        assert mgr.should_throttle(now=10.0 + MIN_PATCH_INTERVAL + 0.1) is False

    def test_should_throttle_never_patched(self):
        from gateway.platforms.feishu import LiveCardManager
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=0.0)
        assert mgr.should_throttle(now=0.1) is False

    def test_mark_degraded(self):
        from gateway.platforms.feishu import LiveCardManager
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=0.0)
        assert mgr.degraded is False
        mgr.mark_degraded()
        assert mgr.degraded is True

    def test_build_card_still_works_when_degraded(self):
        from gateway.platforms.feishu import LiveCardManager
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=0.0)
        mgr.update_text("some text")
        mgr.mark_degraded()
        card = mgr.build_card(now=5.0)
        assert card is not None
        assert card["config"]["update_multi"] is True


# ---------------------------------------------------------------------------
# Adapter integration tests — Tasks 8-14
# ---------------------------------------------------------------------------

from gateway.platforms.base import SendResult, ProcessingOutcome


def _make_live_card(*, state, msg_id="ack_001", started_at=0.0):
    from gateway.platforms.feishu import LiveCardManager, LiveCardState
    mgr = LiveCardManager()
    if state != LiveCardState.IDLE:
        mgr.start(msg_id, started_at=started_at)
    if state == LiveCardState.LIVE:
        mgr.update_text("")
    return mgr


def _make_event(*, chat_id="chat_001", message_id="msg_in_001"):
    source = Mock()
    source.chat_id = chat_id
    event = Mock()
    event.message_id = message_id
    event.source = source
    return event


def _make_adapter():
    """Create a minimal FeishuAdapter-like object for testing live card logic."""
    from gateway.platforms.feishu import LiveCardManager, LiveCardState

    adapter = Mock()
    adapter._live_cards = {}
    adapter._pending_ack_cards = {}
    adapter._card_mode_enabled = True
    adapter._pending_processing_reactions = {}
    adapter._client = Mock()
    adapter.format_message = Mock(side_effect=lambda c: c)
    adapter._patch_card = AsyncMock(
        return_value=SendResult(success=True, message_id="ack_001")
    )
    adapter._send_card = AsyncMock(
        return_value=SendResult(success=True, message_id="ack_001")
    )
    adapter._reactions_enabled = Mock(return_value=True)
    adapter._add_reaction = AsyncMock(return_value="reaction_001")
    adapter._remove_reaction = AsyncMock(return_value=True)
    adapter._remember_processing_reaction = Mock()
    adapter._pop_processing_reaction = Mock()
    return adapter


class TestFeishuLiveCardIntegration:
    """Tasks 8-11: Adapter lifecycle integration."""

    @pytest.mark.asyncio
    async def test_on_processing_start_creates_live_card(self):
        from gateway.platforms.feishu import (
            FeishuAdapter, LiveCardState, LiveCardManager,
        )
        adapter = _make_adapter()
        adapter._live_cards = {}

        event = _make_event(chat_id="chat_001", message_id="msg_in_001")
        await FeishuAdapter.on_processing_start(adapter, event)

        assert "chat_001" in adapter._live_cards
        live = adapter._live_cards["chat_001"]
        assert live.state == LiveCardState.ACK_SENT
        assert live.card_message_id == "ack_001"

    @pytest.mark.asyncio
    async def test_on_processing_start_without_card_mode(self):
        from gateway.platforms.feishu import FeishuAdapter
        adapter = _make_adapter()
        adapter._card_mode_enabled = False

        event = _make_event(chat_id="chat_001", message_id="msg_in_001")
        await FeishuAdapter.on_processing_start(adapter, event)

        assert "chat_001" not in adapter._live_cards

    @pytest.mark.asyncio
    async def test_send_final_patches_and_cleans_up(self):
        from gateway.platforms.feishu import (
            FeishuAdapter, LiveCardState, LiveCardManager,
        )
        adapter = _make_adapter()
        live = _make_live_card(state=LiveCardState.LIVE, msg_id="ack_001")
        adapter._live_cards["chat_001"] = live
        adapter._pending_ack_cards["chat_001"] = "ack_001"

        result = await FeishuAdapter.send(
            adapter, "chat_001", "Here is my full answer...",
            metadata={"footer_line": "📊 ↑1k | ↓2k"},
        )

        assert result.success
        assert "chat_001" not in adapter._pending_ack_cards
        assert "chat_001" not in adapter._live_cards

    @pytest.mark.asyncio
    async def test_edit_message_updates_accumulated_text(self):
        from gateway.platforms.feishu import (
            FeishuAdapter, LiveCardState, LiveCardManager,
        )
        adapter = _make_adapter()
        live = _make_live_card(state=LiveCardState.LIVE, msg_id="ack_001")
        adapter._live_cards["chat_001"] = live

        result = await FeishuAdapter.edit_message(
            adapter, "chat_001", "progress_001",
            "Updated streaming text...",
        )

        assert result.success
        assert adapter._live_cards["chat_001"].accumulated_text == "Updated streaming text..."

    @pytest.mark.asyncio
    async def test_edit_message_no_live_card_falls_through(self):
        from gateway.platforms.feishu import FeishuAdapter
        adapter = _make_adapter()

        result = await FeishuAdapter.edit_message(
            adapter, "chat_001", "msg_001", "edited text",
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_on_processing_complete_cancels_heartbeat(self):
        from gateway.platforms.feishu import (
            FeishuAdapter, LiveCardState, LiveCardManager,
        )
        adapter = _make_adapter()
        live = _make_live_card(state=LiveCardState.LIVE, msg_id="ack_001")
        mock_task = Mock()
        mock_task.cancel = Mock()
        live.heartbeat_task = mock_task
        adapter._live_cards["chat_001"] = live

        event = _make_event(chat_id="chat_001", message_id="msg_in_001")
        await FeishuAdapter.on_processing_complete(
            adapter, event, ProcessingOutcome.SUCCESS,
        )

        assert "chat_001" not in adapter._live_cards
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_processing_complete_without_live_card(self):
        from gateway.platforms.feishu import FeishuAdapter
        adapter = _make_adapter()

        event = _make_event(chat_id="chat_001", message_id="msg_in_001")
        await FeishuAdapter.on_processing_complete(
            adapter, event, ProcessingOutcome.SUCCESS,
        )

    @pytest.mark.asyncio
    async def test_send_progress_patches_card(self):
        from gateway.platforms.feishu import (
            FeishuAdapter, LiveCardState,
        )
        adapter = _make_adapter()
        live = _make_live_card(state=LiveCardState.ACK_SENT, msg_id="ack_001")
        adapter._live_cards["chat_001"] = live
        adapter._pending_ack_cards["chat_001"] = "ack_001"

        result = await FeishuAdapter.send(
            adapter, "chat_001", "Reading config.yaml...",
        )

        assert result.success
        # ACK card still in pending (not consumed — this was a progress message)
        assert "chat_001" in adapter._pending_ack_cards


class TestRecordPatchResult:
    """Consecutive-failure degradation via record_patch_result."""

    def test_success_resets_counter(self):
        from gateway.platforms.feishu import LiveCardManager
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=0.0)
        mgr.record_patch_result(False)
        mgr.record_patch_result(False)
        assert not mgr.degraded
        mgr.record_patch_result(True)
        assert mgr._consecutive_failures == 0
        assert not mgr.degraded

    def test_three_failures_triggers_degradation(self):
        from gateway.platforms.feishu import LiveCardManager
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=0.0)
        mgr.record_patch_result(False)
        mgr.record_patch_result(False)
        assert not mgr.degraded
        mgr.record_patch_result(False)
        assert mgr.degraded

    def test_intermittent_failure_does_not_degrade(self):
        from gateway.platforms.feishu import LiveCardManager
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=0.0)
        mgr.record_patch_result(False)
        mgr.record_patch_result(False)
        mgr.record_patch_result(True)
        mgr.record_patch_result(False)
        mgr.record_patch_result(False)
        assert not mgr.degraded

    @pytest.mark.asyncio
    async def test_send_progress_degrades_after_consecutive_failures(self):
        from gateway.platforms.feishu import FeishuAdapter, LiveCardState
        adapter = _make_adapter()
        adapter._patch_card = AsyncMock(
            return_value=SendResult(success=False, error="network error")
        )
        live = _make_live_card(state=LiveCardState.ACK_SENT, msg_id="ack_001")
        adapter._live_cards["chat_001"] = live
        adapter._pending_ack_cards["chat_001"] = "ack_001"

        for _ in range(3):
            await FeishuAdapter.send(adapter, "chat_001", "progress...")

        assert adapter._live_cards["chat_001"].degraded

    @pytest.mark.asyncio
    async def test_edit_message_degrades_after_consecutive_failures(self):
        from gateway.platforms.feishu import FeishuAdapter, LiveCardState
        adapter = _make_adapter()
        adapter._patch_card = AsyncMock(
            return_value=SendResult(success=False, error="network error")
        )
        live = _make_live_card(state=LiveCardState.LIVE, msg_id="ack_001")
        adapter._live_cards["chat_001"] = live

        for _ in range(3):
            await FeishuAdapter.edit_message(
                adapter, "chat_001", "msg_001", "streaming text..."
            )

        assert adapter._live_cards["chat_001"].degraded


class TestLiveCardDegradation:
    """Task 13: Three-tier degradation."""

    @pytest.mark.asyncio
    async def test_degraded_send_uses_original_path(self):
        from gateway.platforms.feishu import (
            FeishuAdapter, LiveCardState,
        )
        adapter = _make_adapter()
        live = _make_live_card(state=LiveCardState.LIVE, msg_id="ack_001")
        live.mark_degraded()
        adapter._live_cards["chat_001"] = live
        adapter._pending_ack_cards["chat_001"] = "ack_001"

        result = await FeishuAdapter.send(
            adapter, "chat_001", "final answer",
            metadata={"footer_line": "📊 stats"},
        )

        assert result.success
        # Should have consumed the ACK card via original path
        assert "chat_001" not in adapter._pending_ack_cards


class TestFullLifecycle:
    """Task 15: End-to-end lifecycle test."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        from gateway.platforms.feishu import (
            FeishuAdapter, LiveCardState,
        )
        adapter = _make_adapter()

        event = _make_event(chat_id="chat_001", message_id="msg_in_001")

        # 1. Processing start → ACK card sent, live card created
        await FeishuAdapter.on_processing_start(adapter, event)
        assert "chat_001" in adapter._live_cards
        assert adapter._live_cards["chat_001"].state == LiveCardState.ACK_SENT

        # 2. Streaming text → card patched with accumulated text
        await FeishuAdapter.edit_message(
            adapter, "chat_001", "progress_001", "Analyzing code...",
        )
        assert adapter._live_cards["chat_001"].state == LiveCardState.LIVE
        assert adapter._live_cards["chat_001"].accumulated_text == "Analyzing code..."

        # 3. Final answer → card patched with answer + footer, live card removed
        await FeishuAdapter.send(
            adapter, "chat_001", "Here is the answer.",
            metadata={"footer_line": "📊 stats"},
        )
        assert "chat_001" not in adapter._live_cards

        # 4. Processing complete → no crash (live card already gone)
        await FeishuAdapter.on_processing_complete(
            adapter, event, ProcessingOutcome.SUCCESS,
        )
