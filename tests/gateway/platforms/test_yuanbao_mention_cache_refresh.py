"""Yuanbao group sends refresh a cold/expired member cache before resolving @mentions.

Nothing on the send path filled ``_member_cache`` (only the yb_query_group_members tool did),
so an ``@nickname`` in a reply went out as plain text and never notified the member.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.yuanbao.adapter import MessageSender

_MEMBERS = [{"user_id": "user_abc123", "nickname": "元宝"}]


def _sender(cache):
    adapter = SimpleNamespace(name="yuanbao", MEMBER_CACHE_TTL_S=300.0, _member_cache=cache)

    async def _fetch(group_code, offset=0, limit=200):
        cache[group_code] = (time.time(), _MEMBERS)
        return {"members": _MEMBERS}

    adapter.get_group_member_list = AsyncMock(side_effect=_fetch)
    sent = []

    async def _send_msg_body(chat_id, msg_body, reply_to, group_code):
        sent.append(msg_body)
        return {"success": True, "msg_key": "k"}

    sender = MessageSender(adapter)
    sender._send_msg_body = _send_msg_body  # type: ignore[method-assign]
    return sender, adapter, sent


@pytest.mark.asyncio
@pytest.mark.parametrize("cache", [{}, {"g1": (time.time() - 600, _MEMBERS)}])
async def test_cold_or_expired_cache_is_refreshed_and_mention_resolved(cache):
    sender, adapter, sent = _sender(cache)
    result = await sender.send_text_chunk("group:g1", "@元宝 你好")
    assert result.success
    adapter.get_group_member_list.assert_awaited_once_with("g1")
    custom = [e for e in sent[0] if e["msg_type"] == "TIMCustomElem"]
    assert json.loads(custom[0]["msg_content"]["data"])["user_id"] == "user_abc123"


@pytest.mark.asyncio
async def test_no_mention_syntax_skips_refresh():
    sender, adapter, _sent = _sender({})
    await sender.send_text_chunk("group:g1", "mail me at a@b.com")
    adapter.get_group_member_list.assert_not_awaited()


@pytest.mark.asyncio
async def test_fresh_cache_skips_refresh():
    sender, adapter, _sent = _sender({"g1": (time.time(), _MEMBERS)})
    await sender.send_text_chunk("group:g1", "@元宝 hi")
    adapter.get_group_member_list.assert_not_awaited()
