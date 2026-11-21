"""WhatsApp group members, end to end (#7269).

``whatsapp.group_allowed_chats`` (group JIDs, or ``"*"``) lets every member of a listed group talk to
the bot even with a ``WHATSAPP_ALLOWED_USERS`` sender allowlist, as ``group_allowed_chats`` does on
Telegram. Group admission is separate: ``group_policy`` / ``group_allow_from`` decide which groups
the bridge forwards (that gate is covered by ``scripts/whatsapp-bridge/allowlist.test.mjs``). Here a
bridge-shaped message goes through the real adapter intake into the gateway's authorization.
"""

import json
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, load_gateway_config

OWNER, MEMBER = "15550001111", "15550002222@s.whatsapp.net"
GROUP, OTHER_GROUP = "120363001234567890@g.us", "120363009999999999@g.us"
BOT = "15551230000@s.whatsapp.net"


def _gateway(monkeypatch, tmp_path, group_allowed_chats=None):
    """A runner and its WhatsApp adapter built from a config.yaml admitting both groups, mention-gated."""
    from gateway.run import GatewayRunner
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    home = tmp_path / ".hermes"
    home.mkdir()
    lines = ["whatsapp:", "  enabled: true", "  require_mention: true", "  group_policy: allowlist",
             f"  group_allow_from: {json.dumps([GROUP, OTHER_GROUP])}"]
    if group_allowed_chats is not None:
        lines.append(f"  group_allowed_chats: {json.dumps(group_allowed_chats)}")
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", OWNER)
    for name in ("WHATSAPP_REQUIRE_MENTION", "WHATSAPP_GROUP_POLICY", "WHATSAPP_GROUP_ALLOWED_USERS"):
        monkeypatch.setenv(name, "")  # restored after the test; the config.yaml bridge replaces them
    config = load_gateway_config()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    adapter = WhatsAppAdapter(config.platforms[Platform.WHATSAPP])
    adapter.gateway_runner = runner
    runner.adapters = {Platform.WHATSAPP: adapter}
    return runner, adapter


def _member_mentions_the_bot(chat_id):
    """A bridge event (``bridge_helpers.js`` ``extractBridgeEvent``): a member @mentions the bot."""
    return {"messageId": "M1", "chatId": chat_id, "senderId": MEMBER, "senderName": "Member", "chatName": "group",
            "isGroup": True, "body": "@bot what's new?", "hasMedia": False, "mediaType": "", "mediaUrls": [],
            "mentionedIds": [BOT], "botIds": [BOT], "hasQuotedMessage": False}


async def _member_is_authorized(runner, adapter, chat_id=GROUP):
    event = await adapter._build_message_event(_member_mentions_the_bot(chat_id))
    assert event is not None, "the adapter dropped an admitted group message that mentions the bot"
    return runner._is_user_authorized(event.source)


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed_chats", [[GROUP], [OTHER_GROUP, GROUP], ["*"]])
async def test_an_unlisted_member_mentioning_the_bot_in_a_listed_group_is_authorized(
        monkeypatch, tmp_path, allowed_chats):
    assert await _member_is_authorized(*_gateway(monkeypatch, tmp_path, allowed_chats))


@pytest.mark.asyncio
async def test_a_member_of_an_unlisted_group_is_refused(monkeypatch, tmp_path):
    assert not await _member_is_authorized(*_gateway(monkeypatch, tmp_path, [GROUP]), chat_id=OTHER_GROUP)


@pytest.mark.asyncio
async def test_admitting_a_group_does_not_authorize_its_members(monkeypatch, tmp_path):
    """With ``WHATSAPP_ALLOWED_USERS`` set, an admitted group's participant must still be on it."""
    assert not await _member_is_authorized(*_gateway(monkeypatch, tmp_path))
