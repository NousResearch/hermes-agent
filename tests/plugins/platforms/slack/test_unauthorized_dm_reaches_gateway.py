"""An unauthorized Slack DM reaches the gateway when the gateway is going to answer it.

``unauthorized_dm_behavior`` (a pairing code by default, or a decline) is the gateway's decision,
made on a ``MessageEvent``. The adapter's early reject returned before one existed, so an unpaired
user's DM got silence and the documented pairing prompt never fired. Regression for #129107.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import DEFAULT_UNAUTHORIZED_DM_DECLINE_MESSAGE, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from plugins.platforms.slack.adapter import SlackAdapter


def _pairing_store(code):
    store = MagicMock()
    store.is_approved.return_value = False
    store._is_rate_limited.return_value = False
    store.has_recent_decline.return_value = False
    store.generate_code.return_value = code
    return store


@pytest.mark.asyncio
@pytest.mark.parametrize("channel_id, channel_type, allowlist, behavior, multiplexed, reply", [
    ("D_NEW", "im", None, None, False, "ABC12DEF"),
    ("D_NEW", "im", "U_OWNER", None, False, None),
    ("D_NEW", "im", "U_OWNER", "decline", False, DEFAULT_UNAUTHORIZED_DM_DECLINE_MESSAGE),
    ("D_NEW", "im", None, None, True, "WORK7777"),
    ("G_GROUP", "mpim", None, None, False, None),
    ("C_OPS", "channel", None, None, False, None),
], ids=["dm-pair", "dm-ignore", "dm-decline", "dm-pair-multiplexed-owner", "group-dm", "channel"])
async def test_unpaired_sender_gets_the_gateway_answer_only_in_a_dm(
        monkeypatch, channel_id, channel_type, allowlist, behavior, multiplexed, reply):
    """The sender is rejected before any Slack lookup in every row. A 1:1 DM still goes on to the
    real admission path, as a bare event, when that path answers it: a pairing code with no
    allowlist, the decline text when configured over an allowlist. ``ignore`` (allowlist, no
    override) stays a silent drop. Under multiplex the handler is a closure, so the runner is found
    through ``gateway_runner`` and the answer comes from the owning profile's pairing store."""
    for key in ("SLACK_ALLOWED_USERS", "SLACK_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    if allowlist:
        monkeypatch.setenv("SLACK_ALLOWED_USERS", allowlist)
    platform_config = PlatformConfig(enabled=True, token="xoxb-fake")
    if behavior:
        platform_config.extra["unauthorized_dm_behavior"] = behavior
    adapter = SlackAdapter(platform_config)
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = "U_BOT"
    adapter._running = True
    adapter.send = AsyncMock()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.SLACK: platform_config})
    runner.pairing_store = _pairing_store("ABC12DEF")
    runner.hooks = SimpleNamespace(dispatch=AsyncMock(return_value=None))
    runner._running_agents, runner._running_agents_ts = {}, {}
    runner._update_prompts, runner._sessions = {}, {}
    if multiplexed:
        # The secondary-bot wiring: a profile-scoped closure handler and auth check, owner "work".
        runner.adapters, runner._profile_adapters = {}, {"work": {Platform.SLACK: adapter}}
        runner.pairing_stores = {"work": _pairing_store("WORK7777")}
        adapter.set_owner_profile("work")
        adapter.gateway_runner = runner
        adapter.set_message_handler(runner._make_profile_message_handler("work"))
        adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.SLACK, profile_name="work"))
    else:
        runner.adapters = {Platform.SLACK: adapter}
        adapter.set_message_handler(runner._handle_message)
    adapter.handle_message = AsyncMock(wraps=adapter.handle_message)

    await adapter._handle_slack_message(
        {"text": "<@U_BOT> hello", "user": "U_NEW", "channel": channel_id, "channel_type": channel_type,
         "ts": "1700.000100", "client_msg_id": "typed-by-a-human"},
        {"team_id": "T1"})
    await asyncio.gather(*adapter._background_tasks)

    adapter._app.client.users_info.assert_not_awaited()
    adapter._app.client.conversations_info.assert_not_awaited()
    adapter._app.client.conversations_replies.assert_not_awaited()
    answered = int(reply is not None)
    assert adapter.handle_message.await_count == answered
    assert adapter.send.await_count == answered
    if answered:
        chat_id, sent = adapter.send.await_args.args[:2]
        assert chat_id == channel_id and reply in sent
