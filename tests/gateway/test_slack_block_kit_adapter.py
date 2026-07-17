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
    async def test_over_block_cap_sends_multiple_block_messages(self):
        # 60 headed sections -> >50 blocks -> split into several messages,
        # each <= 50 blocks with its own text fallback (never flat mrkdwn)
        adapter, client = _make_adapter({"rich_blocks": True})
        md = "\n\n".join(f"# t{i}\n\npara {i}" for i in range(35))  # 70 blocks
        await adapter.send("C1", md)
        calls = client.chat_postMessage.await_args_list
        assert len(calls) >= 2
        for c in calls:
            assert c.kwargs["blocks"]
            assert len(c.kwargs["blocks"]) <= 50
            assert c.kwargs["text"].strip()

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


class TestStreamingBlocks:
    @pytest.mark.asyncio
    async def test_intermediate_edit_renders_blocks_when_enabled(self):
        adapter, client = _make_adapter(
            {"rich_blocks": True, "rich_blocks_streaming": True}
        )
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=False)
        kwargs = client.chat_update.await_args.kwargs
        assert "blocks" in kwargs and kwargs["blocks"]
        assert kwargs["text"]

    @pytest.mark.asyncio
    async def test_intermediate_frames_never_carry_feedback_buttons(self):
        adapter, client = _make_adapter(
            {
                "rich_blocks": True,
                "rich_blocks_streaming": True,
                "feedback_buttons": True,
            }
        )
        await adapter.edit_message("C1", "1.2", RICH_MD, finalize=False)
        blocks = client.chat_update.await_args.kwargs["blocks"]
        assert all(b["type"] != "context_actions" for b in blocks)
        await adapter.edit_message("C1", "1.2", RICH_MD, finalize=True)
        blocks = client.chat_update.await_args.kwargs["blocks"]
        assert blocks[-1]["type"] == "context_actions"

    @pytest.mark.asyncio
    async def test_streaming_flag_requires_rich_blocks(self):
        adapter, client = _make_adapter({"rich_blocks_streaming": True})
        await adapter.edit_message("C1", "1.2", RICH_MD, finalize=False)
        assert "blocks" not in client.chat_update.await_args.kwargs


class TestBlocksRejectionFallback:
    @pytest.mark.asyncio
    async def test_send_retries_without_blocks_on_invalid_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        client.chat_postMessage = AsyncMock(
            side_effect=[
                Exception("The request to the Slack API failed. (error: invalid_blocks)"),
                {"ts": "1.2"},
            ]
        )
        res = await adapter.send("C1", RICH_MD)
        assert res.success
        calls = client.chat_postMessage.await_args_list
        assert "blocks" in calls[0].kwargs
        assert "blocks" not in calls[1].kwargs
        assert calls[1].kwargs["text"]  # message content survives as plain text

    @pytest.mark.asyncio
    async def test_send_non_block_error_propagates(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        client.chat_postMessage = AsyncMock(side_effect=Exception("ratelimited"))
        res = await adapter.send("C1", RICH_MD)
        assert not res.success
        assert len(client.chat_postMessage.await_args_list) == 1

    @pytest.mark.asyncio
    async def test_finalize_edit_retries_without_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        client.chat_update = AsyncMock(
            side_effect=[Exception("invalid_blocks"), {"ok": True}]
        )
        res = await adapter.edit_message("C1", "111.222", RICH_MD, finalize=True)
        assert res.success
        calls = client.chat_update.await_args_list
        assert "blocks" in calls[0].kwargs
        # text-only chat.update removes the rejected blocks and renders text
        assert "blocks" not in calls[1].kwargs


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
