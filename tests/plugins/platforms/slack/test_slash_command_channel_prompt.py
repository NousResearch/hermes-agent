"""Slack slash-command turns carry the same channel prompt, skill binding and source names as messages.

A ``/hermes <question>`` turn is a human turn, so the gateway re-pins ``channel_pin`` and the
session-context key from it. Built without ``channel_prompt``, ``chat_name`` and ``user_name``,
it ran without the configured channel prompt and flipped both pins, and the next ordinary
message flipped them back: two agent rebuilds and two prompt-cache misses per slash turn. Without
``auto_skill``, a session opened by ``/hermes <question>`` never loaded the channel's bound skill.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


def _adapter(channel_id: str) -> SlackAdapter:
    a = SlackAdapter(PlatformConfig(
        enabled=True, token="xoxb-fake", extra={
            "channel_prompts": {channel_id: "Answer in haiku."},
            "channel_skill_bindings": [{"id": channel_id, "skill": "triage"}]}))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(
        return_value={"user": {"profile": {"display_name": "Alice"}, "real_name": "Alice"}})
    a._app.client.conversations_info = AsyncMock(
        return_value={"ok": True, "channel": (
            {"id": channel_id, "is_im": True, "user": "U_ALICE"} if channel_id.startswith("D")
            else {"id": channel_id, "name": "ops"})})
    a._bot_user_id = "U_BOT"
    a._bot_display_name = "HermesBot"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.mark.asyncio
@pytest.mark.parametrize("channel_id, channel_type, chat_name", [
    ("C_OPS", "channel", "ops"), ("D_ALICE", "im", "Alice")])
async def test_slash_turn_matches_message_turn_prompt_and_names(channel_id, channel_type, chat_name):
    adapter = _adapter(channel_id)
    await adapter._handle_slash_command(
        {"command": "/hermes", "text": "what broke?", "user_id": "U_ALICE",
         "channel_id": channel_id, "team_id": "T1"})
    await adapter._handle_slack_message(
        {"text": "<@U_BOT> what broke?", "user": "U_ALICE", "channel": channel_id,
         "channel_type": channel_type, "ts": "1700.000100"},
        {"team_id": "T1"})
    assert adapter.handle_message.await_count == 2
    slash, message = (c.args[0] for c in adapter.handle_message.await_args_list)
    assert message.channel_prompt and "Answer in haiku." in message.channel_prompt
    assert slash.channel_prompt == message.channel_prompt
    assert slash.auto_skill == message.auto_skill == ["triage"]
    assert (message.source.chat_name, message.source.user_name) == (chat_name, "Alice")
    assert (slash.source.chat_name, slash.source.user_name) == (
        message.source.chat_name, message.source.user_name)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel_id, reaches_runner", [("C_OPS", 0), ("D_ALICE", 1)], ids=["channel", "dm"])
async def test_rejected_slash_sender_costs_no_slack_lookup(channel_id, reaches_runner):
    """The message path rejects an unauthorized sender before any Slack lookup, and the names the
    slash path now resolves must not cost one either. In a DM the runner still gets the event,
    without names: it answers an unauthorized DM per ``unauthorized_dm_behavior`` (pairing code
    or decline), and a slash command there is how an unpaired user gets that answer."""
    adapter = _adapter(channel_id)
    adapter.set_authorization_check(lambda *_args, **_kwargs: False)
    await adapter._handle_slash_command(
        {"command": "/hermes", "text": "what broke?", "user_id": "U_MALLORY",
         "channel_id": channel_id, "team_id": "T1"})
    adapter._app.client.users_info.assert_not_awaited()
    adapter._app.client.conversations_info.assert_not_awaited()
    assert adapter.handle_message.await_count == reaches_runner
