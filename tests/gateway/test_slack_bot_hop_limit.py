"""A bot answering a bot must stop on its own.

The cap counts one direction of one thread. A person mentioning the bot
resets it. A different thread is a different count.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_slack_mock():
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


BOT = "U_TARGET_BOT"
PEER = "U_PEER_BOT"
TEAM = "T1"
CHANNEL = "C1"
THREAD = "1700000000.100000"


def _event(*, text, user, ts, thread=THREAD, bot_id=None):
    event = {
        "text": text,
        "user": user,
        "channel": CHANNEL,
        "channel_type": "channel",
        "team": TEAM,
        "ts": ts,
        "thread_ts": thread,
    }
    if bot_id:
        event["bot_id"] = bot_id
    else:
        event["client_msg_id"] = f"cm-{ts}"
    return event


@pytest.fixture()
def adapter():
    config = PlatformConfig(
        enabled=True,
        token="xoxb-test",
        extra={"allow_bots": "all", "require_mention": False},
    )
    built = SlackAdapter(config)
    built._app = MagicMock()
    built._app.client = AsyncMock()
    built._bot_user_id = BOT
    built._team_bot_user_ids = {TEAM: BOT}
    built._running = True
    built.handle_message = AsyncMock()
    built._resolve_user_name = AsyncMock(return_value="Peer")
    built._fetch_thread_context = AsyncMock(return_value="")
    built._resolve_user_is_bot = AsyncMock(return_value=False)
    return built


def _bot(n):
    return _event(text=f"reply {n}", user=PEER, ts=f"1700000001.{n:06d}", bot_id="B_PEER")


class TestSlackBotHopLimit:

    @pytest.mark.asyncio
    async def test_seventh_bot_reply_in_one_thread_is_dropped(self, adapter):
        for n in range(1, 7):
            await adapter._handle_slack_message(_bot(n))
        assert adapter.handle_message.await_count == 6

        await adapter._handle_slack_message(_bot(7))

        assert adapter.handle_message.await_count == 6

    @pytest.mark.asyncio
    async def test_a_human_mention_restarts_the_count(self, adapter):
        for n in range(1, 7):
            await adapter._handle_slack_message(_bot(n))

        human = _event(text=f"<@{BOT}> go on", user="U_HUMAN", ts="1700000002.000001")
        await adapter._handle_slack_message(human)
        await adapter._handle_slack_message(_bot(8))

        assert adapter.handle_message.await_count == 8

    @pytest.mark.asyncio
    async def test_a_different_thread_has_its_own_count(self, adapter):
        for n in range(1, 7):
            await adapter._handle_slack_message(_bot(n))

        other = _event(
            text="reply elsewhere", user=PEER, ts="1700000003.000001",
            thread="1700000099.100000", bot_id="B_PEER")
        await adapter._handle_slack_message(other)

        assert adapter.handle_message.await_count == 7
