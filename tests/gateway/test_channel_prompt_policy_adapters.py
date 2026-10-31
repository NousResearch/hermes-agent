"""Wildcard channel policy survives adapter resolution and Slack slash-command ingress."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from gateway.config import PlatformConfig


@pytest.mark.parametrize("exact, expected", [
    ("Thread context", "Policy\n\nThread context"),
    (" ", "Policy\n\nParent context"),
])
def test_discord_thread_combines_policy_with_selected_context(exact, expected):
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter.config = PlatformConfig(extra={"channel_prompts": {
        "*": "Policy", "thread": exact, "parent": "Parent context"}})
    assert adapter._resolve_channel_prompt("thread", "parent") == expected


@pytest.mark.asyncio
async def test_feishu_inbound_event_carries_policy_and_channel_context():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    adapter.config = PlatformConfig(extra={"channel_prompts": {
        "*": "Policy", "oc_chat": "Feishu context"}})
    adapter._bot_open_id, adapter._bot_user_id, adapter._bot_name = "ou_bot", "", "Hermes"
    adapter._download_feishu_message_resources = AsyncMock(return_value=([], []))
    adapter._fetch_message_text = AsyncMock(return_value=None)
    adapter.get_chat_info = AsyncMock(return_value={"name": "Test Chat"})
    adapter._resolve_sender_profile = AsyncMock(return_value={
        "user_id": "u1", "user_name": "Alice", "user_id_alt": None})
    adapter._resolve_source_chat_type = Mock(return_value="group")
    adapter.build_source = Mock(return_value=SimpleNamespace(thread_id=None))
    adapter._dispatch_inbound_event = AsyncMock()
    message = SimpleNamespace(
        content=json.dumps({"text": "plain message"}), message_type="text", message_id="m",
        mentions=[], chat_id="oc_chat", parent_id=None, upper_message_id=None, thread_id=None)

    await adapter._process_inbound_message(
        data=message, message=message, sender_id=None, chat_type="group", message_id="m")

    event = adapter._dispatch_inbound_event.call_args.args[0]
    assert event.channel_prompt == "Policy\n\nFeishu context"


def _slack_adapter(channel_id):
    from plugins.platforms.slack.adapter import SlackAdapter

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra={
        "channel_prompts": {"*": "Policy", channel_id: "Slack context"}}))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._app.client.conversations_replies = AsyncMock(return_value={"messages": []})
    adapter._app.client.users_info = AsyncMock(return_value={
        "user": {"profile": {"display_name": "Alice"}, "real_name": "Alice"}})
    adapter._app.client.conversations_info = AsyncMock(return_value={
        "ok": True, "channel": {"id": channel_id, "name": "ops"}})
    adapter._bot_user_id, adapter._bot_display_name = "U_BOT", "HermesBot"
    adapter._running = True
    adapter.handle_message = AsyncMock()
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("slash_text", ["what broke?", "queue what broke?"])
async def test_slack_message_and_slash_preserve_identity_then_policy(slash_text):
    adapter = _slack_adapter("C_OPS")
    message = {
        "text": "<@U_BOT> what broke?", "user": "U_ALICE", "channel": "C_OPS",
        "channel_type": "channel", "ts": "1700.000100", "thread_ts": "1700.000050"}
    await adapter._handle_slack_message(message, {"team_id": "T1"})
    await adapter._handle_slash_command({
        "command": "/hermes", "text": slash_text, "user_id": "U_ALICE",
        "channel_id": "C_OPS", "team_id": "T1", "thread_ts": "1700.000050"})

    assert adapter.handle_message.await_count == 2
    ordinary, slash = (call.args[0] for call in adapter.handle_message.await_args_list)
    identity = adapter._build_identity_prompt("T1")
    assert identity and "HermesBot" in identity
    assert ordinary.channel_prompt == f"{identity}\n\nPolicy\n\nSlack context"
    assert slash.channel_prompt == ordinary.channel_prompt
