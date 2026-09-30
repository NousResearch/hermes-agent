"""An unauthorized Slack DM reaches the gateway when the gateway is going to answer it.

``unauthorized_dm_behavior`` (a pairing code by default, or a decline) is the gateway's decision,
made on a ``MessageEvent``. The adapter's early reject returned before one existed, so an unpaired
user's DM got silence and the documented pairing prompt never fired. Regression for #129107.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("channel_id, channel_type, allowlist, answered", [
    ("D_NEW", "im", None, 1), ("D_NEW", "im", "U_OWNER", 0),
    ("G_GROUP", "mpim", None, 0), ("C_OPS", "channel", None, 0),
], ids=["dm-pair", "dm-ignore", "group-dm", "channel"])
async def test_unpaired_sender_gets_the_gateway_answer_only_in_a_dm(
        monkeypatch, channel_id, channel_type, allowlist, answered):
    """The sender is rejected before any Slack lookup in every row. A 1:1 DM still goes on to the
    real admission path, as a bare event, when that path answers it (no allowlist: pairing code).
    With an allowlist the behaviour resolves to ``ignore`` and the DM stays a silent drop."""
    for key in ("SLACK_ALLOWED_USERS", "SLACK_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    if allowlist:
        monkeypatch.setenv("SLACK_ALLOWED_USERS", allowlist)
    platform_config = PlatformConfig(enabled=True, token="xoxb-fake")
    adapter = SlackAdapter(platform_config)
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = "U_BOT"
    adapter._running = True
    adapter.send = AsyncMock()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.SLACK: platform_config})
    runner.adapters = {Platform.SLACK: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.pairing_store.generate_code.return_value = "ABC12DEF"
    runner.hooks = SimpleNamespace(dispatch=AsyncMock(return_value=None))
    runner._running_agents, runner._running_agents_ts = {}, {}
    runner._update_prompts, runner._sessions = {}, {}
    adapter.set_message_handler(runner._handle_message)
    adapter.handle_message = AsyncMock(wraps=adapter.handle_message)

    await adapter._handle_slack_message(
        {"text": "<@U_BOT> hello", "user": "U_NEW", "channel": channel_id, "channel_type": channel_type,
         "ts": "1700.000100", "client_msg_id": "typed-by-a-human"},
        {"team_id": "T1"})
    await asyncio.gather(*adapter._background_tasks)

    adapter._app.client.users_info.assert_not_awaited()
    adapter._app.client.conversations_info.assert_not_awaited()
    assert adapter.handle_message.await_count == answered
    assert adapter.send.await_count == answered
    if answered:
        chat_id, reply = adapter.send.await_args.args[:2]
        assert chat_id == channel_id and "ABC12DEF" in reply
