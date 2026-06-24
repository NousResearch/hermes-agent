"""
Repeatable smoke tests for Slack peer-agent routing invariants.

Run this file directly after Slack gateway config changes or deploys.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_slack_mock():
    """Install mock slack modules so SlackAdapter can be imported."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)

    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


BOT_USER_ID = "U_TARGET_BOT"
PEER_USER_ID = "U_PEER_BOT"
TEAM_ID = "T_SMOKE"
CHANNEL_ID = "C_SMOKE"
THREAD_TS = "1700000000.100000"
REPLY_TS = "1700000000.200000"


def _make_event(
    *,
    text: str,
    user: str,
    ts: str,
    thread_ts: str | None = None,
    bot_id: str | None = None,
) -> dict:
    event = {
        "text": text,
        "user": user,
        "channel": CHANNEL_ID,
        "channel_type": "channel",
        "team": TEAM_ID,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    if bot_id is not None:
        event["bot_id"] = bot_id
    return event


def _assert_peer_agent_preflight(adapter: SlackAdapter) -> None:
    assert adapter._slack_require_mention() is True, (
        "config: slack.require_mention must be true for the peer-agent smoke target"
    )
    assert adapter._slack_strict_mention() is True, (
        "config: slack.strict_mention must be true so stale thread state cannot wake peers"
    )
    assert str(adapter.config.extra.get("allow_bots", "")).lower().strip() == "mentions", (
        "config: slack.allow_bots must be 'mentions' for peer-agent routing smoke"
    )
    assert adapter._slack_allowed_channels() == set(), (
        "config: slack.allowed_channels must stay empty for the documented smoke profile"
    )
    assert adapter._app is not None and getattr(adapter._app, "client", None) is not None, (
        "platform_connectivity: Slack client must be initialized before routing smoke runs"
    )
    assert adapter._bot_user_id, (
        "bot_identity: Slack auth_test must resolve a bot user id before routing smoke runs"
    )


@pytest.fixture()
def smoke_adapter():
    config = PlatformConfig(
        enabled=True,
        token="xoxb-fake-token",
        extra={
            "require_mention": True,
            "strict_mention": True,
            "allow_bots": "mentions",
            "allowed_channels": "",
        },
    )
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = BOT_USER_ID
    adapter._team_bot_user_ids = {TEAM_ID: BOT_USER_ID}
    adapter._channel_team = {}
    adapter._running = True
    adapter.handle_message = AsyncMock()
    adapter._resolve_user_name = AsyncMock(return_value="Smoke User")
    adapter._fetch_thread_context = AsyncMock(return_value="")
    adapter._fetch_thread_parent_text = AsyncMock(return_value="Parent mention")
    return adapter


class TestSlackPeerAgentSmoke:
    def test_peer_agent_smoke_preflight_contract(self, smoke_adapter):
        _assert_peer_agent_preflight(smoke_adapter)

    @pytest.mark.asyncio
    async def test_human_message_with_current_mention_routes(self, smoke_adapter):
        event = _make_event(
            text=f"<@{BOT_USER_ID}> summarize the deploy status",
            user="U_HUMAN",
            ts=REPLY_TS,
        )

        await smoke_adapter._handle_slack_message(event)

        smoke_adapter.handle_message.assert_awaited_once()
        msg_event = smoke_adapter.handle_message.await_args.args[0]
        assert msg_event.text == "summarize the deploy status", (
            "routing_logic: human @mentions must route and strip the current target mention"
        )
        assert msg_event.source.thread_id == REPLY_TS
        smoke_adapter._fetch_thread_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_peer_bot_without_current_mention_is_ignored_despite_thread_state(
        self, smoke_adapter
    ):
        smoke_adapter._bot_message_ts.add(THREAD_TS)
        smoke_adapter._mentioned_threads.add(THREAD_TS)
        smoke_adapter._has_active_session_for_thread = MagicMock(return_value=True)

        event = _make_event(
            text="status: work finished",
            user=PEER_USER_ID,
            bot_id="B_PEER",
            ts=REPLY_TS,
            thread_ts=THREAD_TS,
        )

        await smoke_adapter._handle_slack_message(event)

        smoke_adapter.handle_message.assert_not_called()
        smoke_adapter._fetch_thread_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_peer_bot_with_current_explicit_mention_routes(self, smoke_adapter):
        smoke_adapter._bot_message_ts.add(THREAD_TS)
        smoke_adapter._has_active_session_for_thread = MagicMock(return_value=False)

        event = _make_event(
            text=f"<@{BOT_USER_ID}> please take over this thread",
            user=PEER_USER_ID,
            bot_id="B_PEER",
            ts=REPLY_TS,
            thread_ts=THREAD_TS,
        )

        await smoke_adapter._handle_slack_message(event)

        smoke_adapter.handle_message.assert_awaited_once()
        msg_event = smoke_adapter.handle_message.await_args.args[0]
        assert msg_event.text == "please take over this thread", (
            "routing_logic: explicit peer-bot @mentions must still route in allow_bots=mentions mode"
        )
        assert smoke_adapter._mentioned_threads == set(), (
            "routing_logic: strict peer-agent mode must not persist thread mentions after routing"
        )
        smoke_adapter._fetch_thread_context.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("text", "case_id"),
        [
            ("ack: sent the summary", "ack"),
            ("status: waiting for approval", "status"),
            ("error: tool call failed", "error"),
        ],
        ids=["ack", "status", "error"],
    )
    async def test_passive_peer_bot_messages_do_not_route(
        self, smoke_adapter, text, case_id
    ):
        smoke_adapter._bot_message_ts.add(THREAD_TS)
        smoke_adapter._mentioned_threads.add(THREAD_TS)
        smoke_adapter._has_active_session_for_thread = MagicMock(return_value=True)

        event = _make_event(
            text=text,
            user=PEER_USER_ID,
            bot_id="B_PEER",
            ts=REPLY_TS,
            thread_ts=THREAD_TS,
        )

        await smoke_adapter._handle_slack_message(event)

        assert smoke_adapter.handle_message.await_count == 0, (
            f"routing_logic: passive peer bot {case_id} messages must never create bot-to-bot loops"
        )
