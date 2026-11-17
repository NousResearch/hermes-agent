"""Send-path contract for ``slack.extra.agent_blocks``: what the final Slack call carries.

  * a valid ``slack-blocks`` reply is posted as those blocks, with the fence-free text as ``text``
  * a streamed answer whose final holds the fence is rewritten in place (chat.update) to the
    blocks, so the raw JSON never stays visible
  * with the flag off the fence is ordinary text (legacy behaviour)
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter

HEADER = {"type": "header", "text": {"type": "plain_text", "text": "Weekly metrics"}}
REPLY = "Here are the numbers.\n\n```slack-blocks\n" + json.dumps([HEADER, {"type": "divider"}]) + "\n```"
META = {"thread_id": "111.000", "user_id": "U123"}


def _make_adapter(extra=None):
    a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra=extra or {}))
    a._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "999.111"})
    client.chat_update = AsyncMock(return_value={"ts": "999.111"})
    client.chat_startStream = AsyncMock(return_value={"ok": True, "ts": "123.456"})
    client.chat_appendStream = AsyncMock(return_value={"ok": True})
    client.chat_stopStream = AsyncMock(return_value={"ok": True})
    a._get_client = MagicMock(return_value=client)
    a.stop_typing = AsyncMock()
    a._running = True
    return a, client


@pytest.mark.asyncio
async def test_fenced_reply_is_posted_as_blocks_with_fence_free_text():
    adapter, client = _make_adapter({"agent_blocks": True})
    assert (await adapter.send("C1", REPLY)).success
    kwargs = client.chat_postMessage.await_args.kwargs
    assert [b["type"] for b in kwargs["blocks"]] == ["header", "divider"]
    assert kwargs["text"] == "Here are the numbers."

    off, off_client = _make_adapter()
    await off.send("C1", REPLY)
    off_kwargs = off_client.chat_postMessage.await_args.kwargs
    assert "blocks" not in off_kwargs and '"Weekly metrics"' in off_kwargs["text"]   # raw JSON, as before


@pytest.mark.asyncio
async def test_streamed_fence_is_replaced_by_blocks_on_seal():
    adapter, client = _make_adapter({"agent_blocks": True})
    await adapter.send_draft("D1", 7, REPLY[:30], metadata=META)
    assert (await adapter.send("D1", REPLY, metadata=META)).success
    client.chat_postMessage.assert_not_awaited()   # the stream IS the final message
    update = client.chat_update.await_args.kwargs
    assert update["ts"] == "123.456"
    assert [b["type"] for b in update["blocks"]] == ["header", "divider"]
    assert "slack-blocks" not in update["text"]
