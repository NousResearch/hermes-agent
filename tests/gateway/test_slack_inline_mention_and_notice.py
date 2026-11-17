"""Slack: an inline self-mention reads as ``@<bot> (you)`` (opt-in ``inline_mention_label``), and an
unauthorized channel @mention gets one ephemeral notice (opt-in ``unauthorized_mention_notice``).

Both drive the real ``SlackAdapter._handle_slack_message`` with a fake web client.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

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


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter

BOT = "U_BOT_123"
CHANNEL = "C0AQWDLHY9M"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache")
    monkeypatch.setattr("gateway.pairing.PAIRING_DIR", tmp_path / "pairing")
    for var in ("SLACK_REQUIRE_MENTION", "SLACK_FREE_RESPONSE_CHANNELS", "SLACK_STRICT_MENTION"):
        monkeypatch.delenv(var, raising=False)


def _adapter(**extra):
    a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra=dict(extra)))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = BOT
    a._bot_display_name = "hermes"
    a._running = True
    a.handle_message = AsyncMock()
    return a


async def _run(adapter, text, ts, user="U_HUMAN", channel=CHANNEL, channel_type="channel"):
    event = {"channel": channel, "channel_type": channel_type, "user": user, "text": text, "ts": ts,
             "client_msg_id": f"m{ts}"}
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="human")), \
            patch.object(adapter, "_fetch_thread_context", new=AsyncMock(return_value=None)), \
            patch.object(adapter, "_fetch_thread_parent_text", new=AsyncMock(return_value="")), \
            patch.object(adapter, "_has_active_session_for_thread", return_value=False):
        await adapter._handle_slack_message(event)


@pytest.mark.asyncio
@pytest.mark.parametrize("label, text, expected", [
    (False, f"<@{BOT}> ask <@{BOT}> later about the deploy", "ask  later about the deploy"),  # legacy
    (True, f"<@{BOT}> ask <@{BOT}> later about the deploy", "ask @hermes (you) later about the deploy"),
    (True, f"<@{BOT}>, /stop", ", /stop"),  # the label never turns plain text into a command
])
async def test_inline_self_mention_label(label, text, expected):
    adapter = _adapter(inline_mention_label=label)
    await _run(adapter, text, ts="1700000000.000001")
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == expected
    assert msg.get_command() is None
    assert msg.reply_expected is True


@pytest.mark.asyncio
async def test_unauthorized_channel_mention_gets_one_ephemeral_notice():
    adapter = _adapter(unauthorized_mention_notice=True)
    adapter.set_authorization_check(lambda *_a, **_kw: False)
    await _run(adapter, f"<@{BOT}> help me", ts="1700000000.000002")
    await _run(adapter, f"<@{BOT}> help me again", ts="1700000000.000003")
    await _run(adapter, "no mention here", ts="1700000000.000004", user="U_OTHER")
    await _run(adapter, f"<@{BOT}> hi", ts="1700000000.000005", user="U_GROUP", channel="G0MPIM",
               channel_type="mpim")
    adapter.config.extra["allowed_channels"] = [CHANNEL, "G0MPIM"]
    await _run(adapter, f"<@{BOT}> hi", ts="1700000000.000006", user="U_ELSEWHERE", channel="C0OTHER")
    adapter.handle_message.assert_not_awaited()
    calls = adapter._app.client.chat_postEphemeral.await_args_list
    assert [(c.kwargs["channel"], c.kwargs["user"]) for c in calls] == [
        (CHANNEL, "U_HUMAN"), ("G0MPIM", "U_GROUP")]
    assert "not enabled for you" in calls[0].kwargs["text"]
