"""Merge-forward messages: the event payload carries no children, so the adapter must expand
them from GET im/v1/messages/{id} (items[0] = container, items[1:] = forwarded children)."""

import asyncio
import json
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


def _item(msg_type, content, *, upper=None, sender_id="ou_a", sender_type="user"):
    return SimpleNamespace(
        msg_type=msg_type, body=SimpleNamespace(content=json.dumps(content)), mentions=None,
        upper_message_id=upper, sender=SimpleNamespace(id=sender_id, id_type="open_id", sender_type=sender_type),
    )


def _adapter(items):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._bot_open_id, adapter._bot_user_id, adapter._bot_name = "ou_bot", "", "Hermes"
    adapter._message_text_cache = OrderedDict()
    adapter._client = Mock()
    adapter._build_get_message_request = Mock(return_value=object())
    response = Mock()
    response.success = Mock(return_value=True)
    response.data = SimpleNamespace(items=items)
    adapter._client.im.v1.message.get = Mock(return_value=response)
    adapter._resolve_sender_name_from_api = AsyncMock(side_effect=lambda sid, is_bot=False: {"ou_a": "Felix"}.get(sid))
    return adapter


ITEMS = [
    _item("merge_forward", "Merged and Forwarded Message"),
    _item("text", {"text": "is this a CM test hotel?"}, upper="om_fwd"),
    _item("text", {"text": "please parse trip links"}, upper="om_fwd", sender_id="ou_b"),
]


def test_fetch_message_text_expands_merge_forward_children():
    text = asyncio.run(_adapter(ITEMS)._fetch_message_text("om_fwd"))
    assert "[Merged forward message]" not in text
    assert "Felix: is this a CM test hotel?" in text
    assert "please parse trip links" in text


def test_children_are_capped_grandchildren_skipped_and_each_sender_looked_up_once():
    items = [_item("merge_forward", "Merged and Forwarded Message")]
    items += [_item("text", {"text": f"line {i}"}, upper="om_fwd", sender_id="ou_x") for i in range(52)]
    items += [_item("text", {"text": "grandchild"}, upper="om_child")]
    adapter = _adapter(items)
    text = asyncio.run(adapter._fetch_message_text("om_fwd"))
    assert "line 49" in text and "line 50" not in text and "(2 more)" in text
    assert "grandchild" not in text
    assert adapter._resolve_sender_name_from_api.await_count == 1


def test_inbound_merge_forward_keeps_placeholder_when_fetch_fails():
    adapter = _adapter([])
    adapter._client.im.v1.message.get.return_value.success.return_value = False
    adapter._download_feishu_message_resources = AsyncMock(return_value=([], []))
    message = SimpleNamespace(message_type="merge_forward", message_id="om_fwd", mentions=None,
                              content=json.dumps("Merged and Forwarded Message"))
    assert asyncio.run(adapter._extract_message_content(message))[0] == "[Merged forward message]"


def test_inbound_merge_forward_uses_expanded_children():
    adapter = _adapter(ITEMS)
    adapter._download_feishu_message_resources = AsyncMock(return_value=([], []))
    message = SimpleNamespace(message_type="merge_forward", message_id="om_fwd", mentions=None,
                              content=json.dumps("Merged and Forwarded Message"))
    text = asyncio.run(adapter._extract_message_content(message))[0]
    assert "Felix: is this a CM test hotel?" in text
