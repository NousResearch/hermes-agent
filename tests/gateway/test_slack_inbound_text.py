"""Slack message text reaches the agent as typed.

Slack escapes ``&``/``<``/``>`` as ``&amp;``/``&lt;``/``&gt;`` in a message's flat ``text`` and wraps
links as ``<url>``/``<url|label>``. Passed on raw, a URL with a query string becomes a different URL
(``?a=1&amp;b=2``), pasted code is entity-encoded, and command arguments are wrong.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


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
        ("slack_bolt.adapter.socket_mode.async_handler",
         slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod
_slack_mod.SLACK_AVAILABLE = True

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture()
def adapter():
    a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(
        return_value={
            "user": {
                "is_bot": False,
                "profile": {"display_name": "Test User"},
                "real_name": "Test User",
            }
        }
    )
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


def _event(text, channel_type="im", channel="D123", **extra):
    return {
        "text": text,
        "user": "U_USER",
        "channel": channel,
        "channel_type": channel_type,
        "ts": "1234567890.000001",
        **extra,
    }


def _rich_text_section(*elements):
    return [{"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": list(elements)}]}]


async def _agent_text(adapter, event):
    await adapter._handle_slack_message(event)
    return adapter.handle_message.call_args[0][0]


@pytest.mark.asyncio
async def test_dm_text_reaches_agent_as_typed(adapter):
    wire = (
        'run: grep -c "a &lt; b" log.txt &gt; out.txt &amp;&amp; open '
        "<https://example.com/search?q=hermes&amp;page=2>"
    )

    msg_event = await _agent_text(adapter, _event(wire))

    assert msg_event.text == (
        'run: grep -c "a < b" log.txt > out.txt && open '
        "https://example.com/search?q=hermes&page=2"
    )


@pytest.mark.asyncio
async def test_command_arguments_are_unescaped(adapter):
    msg_event = await _agent_text(adapter, _event("!queue fix R&amp;D &lt;draft&gt;"))

    assert msg_event.message_type == MessageType.COMMAND
    assert msg_event.get_command_args() == "fix R&D <draft>"


@pytest.mark.asyncio
async def test_channel_mention_still_routes_and_text_is_unescaped(adapter):
    msg_event = await _agent_text(adapter, _event(
        "<@U_BOT> fetch <https://example.com/s?a=1&amp;b=2|the results>",
        channel_type="channel",
        channel="C123",
    ))

    assert msg_event.text == "fetch the results (https://example.com/s?a=1&b=2)"


@pytest.mark.asyncio
async def test_typed_brackets_and_entities_stay_as_typed(adapter):
    # Typed ``<https://a.example>`` and ``&lt;`` arrive escaped once more and stay literal.
    msg_event = await _agent_text(
        adapter, _event("keep &lt;https://a.example&gt; and &amp;lt; as typed"))

    assert msg_event.text == "keep <https://a.example> and &lt; as typed"


@pytest.mark.asyncio
async def test_link_also_in_rich_text_is_unwrapped_once(adapter):
    msg_event = await _agent_text(adapter, _event(
        "read <https://example.com/docs|the docs>",
        blocks=_rich_text_section(
            {"type": "text", "text": "read "},
            {"type": "link", "url": "https://example.com/docs", "text": "the docs"},
        ),
    ))

    assert msg_event.text == "read the docs (https://example.com/docs)"


def test_thread_context_text_is_unescaped_without_relisting_its_url():
    msg = {
        "text": "see <https://example.com/a?b=1&amp;c=2> for R&amp;D",
        "blocks": _rich_text_section(
            {"type": "text", "text": "see "},
            {"type": "link", "url": "https://example.com/a?b=1&c=2"},
            {"type": "text", "text": " for R&D"},
        ),
    }

    assert SlackAdapter._render_message_text(msg) == "see https://example.com/a?b=1&c=2 for R&D"


def test_thread_context_section_matching_the_text_is_not_repeated():
    msg = {
        "text": "Deploy A &amp; B finished",
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": "Deploy A &amp; B finished"}}
        ],
    }

    assert SlackAdapter._render_message_text(msg) == "Deploy A & B finished"
