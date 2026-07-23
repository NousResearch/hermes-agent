"""Integration tests: SlackAdapter wiring of Block Kit into send paths.

Verifies the opt-in behaviour contract:
  * rich_blocks off (default)  => no ``blocks`` kwarg, plain ``text`` only
  * rich_blocks on             => ``blocks`` present AND ``text`` fallback set
  * edit_message: blocks only on finalize (streaming edits stay plain)
  * multi-chunk (>39k) messages fall back to plain text
"""

from unittest.mock import AsyncMock, MagicMock, call

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
RICH_TABLE_MD = (
    "| Item | Status | Note |\n"
    "|---|---:|---|\n"
    "| Hermes | ok | table |"
)


class SlackRejectedBlocks(Exception):
    def __init__(self, error="invalid_blocks"):
        super().__init__(f"Slack API rejected blocks: {error}")
        self.response = {"error": error}


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
    async def test_feedback_buttons_opt_in_appended_to_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True, "feedback_buttons": True})

        await adapter.send("C1", "final answer")

        blocks = client.chat_postMessage.await_args.kwargs["blocks"]
        feedback = blocks[-1]
        assert feedback["type"] == "context_actions"
        assert feedback["elements"][0]["type"] == "feedback_buttons"
        assert feedback["elements"][0]["action_id"] == "hermes_feedback"

    @pytest.mark.asyncio
    async def test_feedback_buttons_require_rich_blocks(self):
        """feedback_buttons alone must not implicitly enable Block Kit rendering."""
        adapter, client = _make_adapter({"feedback_buttons": True})

        await adapter.send("C1", "final answer")

        assert "blocks" not in client.chat_postMessage.await_args.kwargs

    @pytest.mark.asyncio
    async def test_block_rejection_retries_send_without_blocks_using_workspace_client(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        client.chat_postMessage = AsyncMock(
            side_effect=[SlackRejectedBlocks("invalid_blocks"), {"ts": "111.333"}]
        )

        result = await adapter.send(
            "C1", RICH_TABLE_MD, metadata={"team_id": "T_SECONDARY"}
        )

        assert result.success is True
        assert adapter._get_client.call_args_list == [
            call("C1", team_id="T_SECONDARY"),
            call("C1", team_id="T_SECONDARY"),
        ]
        assert client.chat_postMessage.await_count == 2
        first = client.chat_postMessage.await_args_list[0].kwargs
        second = client.chat_postMessage.await_args_list[1].kwargs
        assert "blocks" in first and first["blocks"]
        assert "blocks" not in second
        assert second["text"]


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
    async def test_finalize_edit_gets_feedback_buttons_when_enabled(self):
        adapter, client = _make_adapter({"rich_blocks": True, "feedback_buttons": True})
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=True)
        blocks = client.chat_update.await_args.kwargs["blocks"]
        assert blocks[-1]["elements"][0]["type"] == "feedback_buttons"

    @pytest.mark.asyncio
    async def test_finalize_edit_disabled_no_blocks(self):
        adapter, client = _make_adapter()  # rich_blocks off
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=True)
        assert "blocks" not in client.chat_update.await_args.kwargs

    @pytest.mark.asyncio
    async def test_block_rejection_retries_edit_without_blocks_using_workspace_client(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        client.chat_update = AsyncMock(
            side_effect=[SlackRejectedBlocks("invalid_blocks"), {"ts": "111.222"}]
        )

        result = await adapter.edit_message(
            "C1",
            "111.222",
            RICH_TABLE_MD,
            finalize=True,
            metadata={"team_id": "T_SECONDARY"},
        )

        assert result.success is True
        assert adapter._get_client.call_args_list == [
            call("C1", team_id="T_SECONDARY"),
            call("C1", team_id="T_SECONDARY"),
        ]
        assert client.chat_update.await_count == 2
        first = client.chat_update.await_args_list[0].kwargs
        second = client.chat_update.await_args_list[1].kwargs
        assert "blocks" in first and first["blocks"]
        assert second["blocks"] == []
        assert second["text"]
