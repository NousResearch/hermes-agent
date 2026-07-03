"""Integration tests: SlackAdapter wiring of Block Kit into send paths.

Verifies the opt-in behaviour contract:
  * rich_blocks off (default)  => no ``blocks`` kwarg, plain ``text`` only
  * rich_blocks on             => ``blocks`` present AND ``text`` fallback set
  * edit_message: blocks only on finalize (streaming edits stay plain)
  * multi-chunk (>39k) messages fall back to plain text
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="xoxb-fake", extra=extra or {})
    a = SlackAdapter(config)
    a._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "111.222"})
    client.chat_update = AsyncMock(return_value={"ts": "111.222"})
    a._get_client = MagicMock(return_value=client)
    a.stop_typing = AsyncMock()
    a._running = True
    return a, client


RICH_MD = "# Title\n\n- a\n  - nested\n\n---\n\nbody text"
TOOL_PROGRESS = (
    "*AI Eng Forge* · Tool execution\n"
    "  ✓ Run skill_view · Completed in 0.0s\n"
    "    📚 Reading skill s2-github-pr-review\n"
    "  ✓ Run terminal · Completed in 6.3s\n"
    "    💻 terminal\n"
    "    ```\n"
    "    set -euo pipefail\n"
    "    pytest tests/gateway/test_slack_block_kit_adapter.py\n"
    "    ```"
)
TOOL_PROGRESS_FAILED = (
    "*AI Eng Forge* · Tool execution\n"
    "  ✓ Run skill_view · Completed in 0.0s\n"
    "    📚 Reading skill s2-github-pr-review\n"
    "  ✕ Run terminal · Failed in 1.2s · Permission denied\n"
    "    💻 terminal\n"
    "    ```\n"
    "    cat /secret\n"
    "    ```"
)


class TestSendMessageBlocks:
    @pytest.mark.asyncio
    async def test_disabled_by_default_no_blocks(self):
        adapter, client = _make_adapter()
        await adapter.send("C1", RICH_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]  # plain text still sent

    @pytest.mark.asyncio
    async def test_enabled_sends_blocks_with_text_fallback(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.send("C1", RICH_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" in kwargs and kwargs["blocks"]
        # text fallback is ALWAYS present alongside blocks (notifications/a11y)
        assert kwargs["text"]
        types = [b["type"] for b in kwargs["blocks"]]
        assert "header" in types
        assert "divider" in types

    @pytest.mark.asyncio
    async def test_enabled_but_unrenderable_falls_back_to_text(self):
        # 60 dividers -> renderer returns None -> no blocks kwarg, text stands
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.send("C1", "\n\n".join(["---"] * 60))
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]

    @pytest.mark.asyncio
    async def test_string_true_coerced(self):
        adapter, client = _make_adapter({"rich_blocks": "true"})
        await adapter.send("C1", RICH_MD)
        assert "blocks" in client.chat_postMessage.await_args.kwargs

    @pytest.mark.asyncio
    async def test_multichunk_message_no_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        huge = "word " * 20000  # well over MAX_MESSAGE_LENGTH -> chunked
        await adapter.send("C1", huge)
        # every posted chunk is plain text, none carry blocks
        for c in client.chat_postMessage.await_args_list:
            assert "blocks" not in c.kwargs
            assert c.kwargs["text"]

    @pytest.mark.asyncio
    async def test_tool_progress_uses_collapsible_blocks_by_default(self):
        adapter, client = _make_adapter()
        await adapter.send("C1", TOOL_PROGRESS)

        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" in kwargs and kwargs["blocks"]
        assert "set -euo pipefail" in kwargs["text"]
        assert kwargs["blocks"][0]["type"] == "section"
        assert kwargs["blocks"][0]["accessory"]["action_id"] == "hermes_tool_progress_toggle"
        assert kwargs["blocks"][0]["accessory"]["text"]["text"] == "Show details"
        visible = "\n".join(
            block.get("text", {}).get("text", "")
            for block in kwargs["blocks"]
            if block.get("type") == "section"
        )
        assert "2 steps" in visible
        assert "2 completed" in visible
        assert "Run skill_view" not in visible
        assert "Run terminal" not in visible
        assert "set -euo pipefail" not in visible


class TestEditMessageBlocks:
    @pytest.mark.asyncio
    async def test_intermediate_edit_no_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=False)
        kwargs = client.chat_update.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]

    @pytest.mark.asyncio
    async def test_finalize_edit_gets_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=True)
        kwargs = client.chat_update.await_args.kwargs
        assert "blocks" in kwargs and kwargs["blocks"]
        assert kwargs["text"]

    @pytest.mark.asyncio
    async def test_finalize_edit_disabled_no_blocks(self):
        adapter, client = _make_adapter()  # rich_blocks off
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=True)
        assert "blocks" not in client.chat_update.await_args.kwargs

    @pytest.mark.asyncio
    async def test_tool_progress_intermediate_edit_keeps_blocks(self):
        adapter, client = _make_adapter()
        await adapter.edit_message("C1", "111.222", TOOL_PROGRESS, finalize=False)

        kwargs = client.chat_update.await_args.kwargs
        assert "blocks" in kwargs and kwargs["blocks"]
        assert kwargs["blocks"][0]["accessory"]["action_id"] == "hermes_tool_progress_toggle"
        assert "set -euo pipefail" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_tool_progress_toggle_expands_cached_card(self):
        adapter, client = _make_adapter()
        await adapter.send("C1", TOOL_PROGRESS)

        ack = AsyncMock()
        await adapter._handle_tool_progress_toggle(
            ack,
            {"channel": {"id": "C1"}, "message": {"ts": "111.222", "text": TOOL_PROGRESS}},
            {"value": "collapsed"},
        )

        ack.assert_awaited_once()
        kwargs = client.chat_update.await_args.kwargs
        assert kwargs["blocks"][0]["accessory"]["text"]["text"] == "Hide details"
        visible = "\n".join(
            block.get("text", {}).get("text", "")
            for block in kwargs["blocks"]
            if block.get("type") == "section"
        )
        assert "set -euo pipefail" in visible
        assert "Reading skill s2-github-pr-review" in visible
        assert "Run skill_view" not in visible
        assert "Run terminal" not in visible

    @pytest.mark.asyncio
    async def test_tool_progress_expanded_failed_step_shows_error_without_summary_row(self):
        adapter, client = _make_adapter()
        await adapter.edit_message("C1", "111.222", TOOL_PROGRESS_FAILED, finalize=False)

        await adapter._handle_tool_progress_toggle(
            AsyncMock(),
            {
                "channel": {"id": "C1"},
                "message": {"ts": "111.222", "text": TOOL_PROGRESS_FAILED},
            },
            {"value": "collapsed"},
        )

        kwargs = client.chat_update.await_args.kwargs
        visible = "\n".join(
            block.get("text", {}).get("text", "")
            for block in kwargs["blocks"]
            if block.get("type") == "section"
        )
        assert "2 steps" in visible
        assert "1 completed" in visible
        assert "1 failed" in visible
        assert "Permission denied" in visible
        assert "cat /secret" in visible
        assert "Run terminal · Failed" not in visible
