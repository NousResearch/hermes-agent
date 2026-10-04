"""QQ mention identity must survive both event modes and quoted context."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.qqbot.adapter import QQAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["Please rename Alice", "@Alice Please rename me", ""])
async def test_mentions_reach_model_without_removing_member_text(content):
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={
        "app_id": "test", "client_secret": "test", "group_policy": "open",
    }))
    adapter.handle_message = AsyncMock()
    member = {"id": "member-target", "username": "Alice😽", "bot": False}
    payload = {
        "id": "original", "group_openid": "test-group",
        "author": {"member_openid": "sender"}, "content": content,
        "mentions": [member, member, {"id": "bot", "username": "Hermes", "bot": True}, None],
    }
    await adapter._on_message("GROUP_AT_MESSAGE_CREATE", payload)
    event = adapter.handle_message.call_args.args[0]
    assert event.text.startswith(content)
    assert event.text.count("@Alice😽") == 1
    assert "member-target" in event.text
    assert "@Hermes" not in event.text
    assert event.raw_message is payload

    adapter.handle_message.reset_mock()
    quote = {
        **payload, "id": "quote", "content": "Who was mentioned?", "mentions": [],
        "message_type": 103,
        "msg_elements": [{"content": "Rename <@member-target>", "mentions": [member]}],
    }
    await adapter._on_message("GROUP_AT_MESSAGE_CREATE", quote)
    text = adapter.handle_message.call_args.args[0].text
    assert text.startswith("[Quoted message]:\nRename @Alice😽")
    assert "member-target" in text.split("Who was mentioned?")[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("bot_id,target,policy", [
    ("our-bot", "", "open"),
    ("our-bot", "other-bot", "open"),
    ("", "our-bot", "open"),
    ("our-bot", "our-bot", "disabled"),
    ("our-bot", "our-bot", "open"),
])
async def test_full_mode_requires_this_bot_and_keeps_member_positions(bot_id, target, policy):
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={
        "app_id": "test", "client_secret": "test", "group_policy": policy,
        "bot_member_openid": bot_id,
    }))
    adapter.handle_message = AsyncMock()
    prefix = f"<@{target}> " if target else ""
    payload = {
        "id": "full", "group_openid": "test-group", "author": {"member_openid": "sender"},
        "content": prefix + "Rename <@member-target> and <@unknown-member>",
        "mentions": [{"id": "member-target", "username": "Alice😽", "bot": False}],
    }
    await adapter._on_message("GROUP_MESSAGE_CREATE", payload)
    if not bot_id or target != bot_id or policy == "disabled":
        adapter.handle_message.assert_not_called()
        return
    event = adapter.handle_message.call_args.args[0]
    assert event.text.startswith("Rename @Alice😽 and <@unknown-member>")
    assert "<@our-bot>" not in event.text
    assert event.source.user_id == "sender"
    # Both QQ event modes can deliver the same message; do not invoke twice.
    await adapter._on_message("GROUP_AT_MESSAGE_CREATE", payload)
    assert adapter.handle_message.call_count == 1
