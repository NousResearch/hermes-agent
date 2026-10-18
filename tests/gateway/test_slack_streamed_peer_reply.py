"""A peer bot's streamed Slack reply must reach the agent once, with its text.

Reproduction (#138676, 2026-10-17): a bot that answers through Slack's streaming API
(``chat.startStream``/``appendStream``/``stopStream`` — what Hermes itself uses) produces two
inbound events for one reply:

    message          ts=T  text=""   streaming_state="in_progress"
    message_changed  ts=T  text=...  streaming_state="completed"

The empty opener built a turn whose body was just the sender label, and claimed ``T`` in
``_processed_message_ts``; the completed edit then fell to ``_normalize_changed_message``'s
already-routed check and was dropped. The agent saw an empty message and answered it — in a
channel with two streaming bots, a loop.

These tests drive the real ``_handle_slack_message`` path with only ``handle_message`` captured.
"""

import asyncio
import importlib
import sys
from importlib.machinery import PathFinder
from types import ModuleType
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig


def _load_installed_package(name):
    if PathFinder.find_spec(name) is None:
        return None
    prefix = f"{name}."
    displaced = {
        m: sys.modules.pop(m)
        for m in tuple(sys.modules)
        if (m == name or m.startswith(prefix)) and not isinstance(sys.modules[m], ModuleType)
    }
    try:
        return importlib.import_module(name)
    except ImportError:
        sys.modules.update(displaced)
        return None


_load_installed_package("slack_bolt")
_load_installed_package("slack_sdk")

SlackAdapter = importlib.import_module("plugins.platforms.slack.adapter").SlackAdapter

CHANNEL = "C0AKA6F2D60"
TEAM = "T0AJR51R2ET"
THREAD_TS = "1792014768.201629"
REPLY_TS = "1792247652.967939"
PEER_USER = "U0AJR5VC4H5"
PEER_BOT_ID = "B0AL0T1QX6U"
BOT_UID = "U0BCLP7DB7B"
REPLY_TEXT = "My reply went through fine on this end."


def _make_adapter(delivered):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra={"allow_bots": "all"}))
    adapter._bot_user_id = BOT_UID
    adapter._resolve_user_name = AsyncMock(return_value="OpenClaw")
    # Hermes was @-mentioned in this thread earlier, so an unmentioned reply in it wakes the
    # bot — the state in which the reporter's gateway built the empty turn.
    adapter._register_mentioned_thread(THREAD_TS, team_id=TEAM)
    adapter._hydrate_thread_context = AsyncMock(return_value=(None, [], []))

    async def _capture(event):
        delivered.append(event)

    adapter.handle_message = _capture
    return adapter


def _stream_opener():
    return {
        "type": "message", "user": PEER_USER, "bot_id": PEER_BOT_ID, "app_id": "A0AL0SCC600",
        "bot_profile": {"id": PEER_BOT_ID, "name": "OpenClaw"},
        "text": "", "streaming_state": "in_progress",
        "channel": CHANNEL, "channel_type": "channel", "team": TEAM,
        "ts": REPLY_TS, "thread_ts": THREAD_TS,
    }


def _stream_completed():
    return {
        "type": "message", "subtype": "message_changed", "hidden": True,
        "channel": CHANNEL, "channel_type": "channel", "team": TEAM,
        "ts": "1792247653.000100", "event_ts": "1792247653.000100",
        "message": {
            "type": "message", "user": PEER_USER, "bot_id": PEER_BOT_ID,
            "bot_profile": {"id": PEER_BOT_ID, "name": "OpenClaw"},
            "text": REPLY_TEXT, "streaming_state": "completed",
            "ts": REPLY_TS, "thread_ts": THREAD_TS,
        },
    }


def _body():
    return {"team_id": TEAM, "event_id": "Ev0STREAM01"}


def test_streamed_peer_reply_yields_one_turn_carrying_the_completed_text():
    delivered = []
    adapter = _make_adapter(delivered)

    async def scenario():
        await adapter._handle_slack_message(_stream_opener(), _body())
        await adapter._handle_slack_message(_stream_completed(), _body())

    asyncio.run(scenario())
    assert len(delivered) == 1, [e.text for e in delivered]
    assert "My reply went through fine" in delivered[0].text


def test_in_progress_opener_alone_builds_no_turn():
    """The opener carries no content; a turn from it is the empty body the agent answered."""
    delivered = []
    adapter = _make_adapter(delivered)
    asyncio.run(adapter._handle_slack_message(_stream_opener(), _body()))
    assert delivered == []
