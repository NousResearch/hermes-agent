"""Mattermost inbound post ``type``: only ``system_*`` events are dropped (openclaw#193836)."""
import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _adapter():
    from plugins.platforms.mattermost.adapter import MattermostAdapter
    adapter = MattermostAdapter(PlatformConfig(enabled=True, token="t", extra={"url": "https://mm.example.com"}))
    adapter._bot_user_id, adapter._bot_username = "bot_user_id", "hermes-bot"
    adapter.handle_message = AsyncMock()
    return adapter


def _event(post_type, post_id):
    post = {"id": post_id, "user_id": "user_123", "channel_id": "dm_chan", "message": "https://example.com/article"}
    if post_type is not None:
        post["type"] = post_type
    return {"event": "posted", "data": {"post": json.dumps(post), "channel_type": "D", "sender_name": "@alice"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(("post_type", "delivered"), [
    ("slack_attachment", True),   # a person's URL post decorated with a link-preview card
    ("", True),
    (None, True),
    ("system_join_channel", False),
    ("system_header_change", False),
])
async def test_only_system_post_types_are_dropped(post_type, delivered):
    adapter = _adapter()
    await adapter._handle_ws_event(_event(post_type, f"p_{post_type}"))
    assert adapter.handle_message.called is delivered
    if delivered:
        assert adapter.handle_message.call_args[0][0].text == "https://example.com/article"
