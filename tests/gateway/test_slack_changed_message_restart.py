"""Regression for #118349: a restart must not let ``message_changed`` replay an old message.

``_processed_message_ts`` lives in memory, so a new adapter cannot know which messages the
previous process already answered. Slack keeps emitting ``message_changed`` for those messages
on its own (thread reply metadata, native agent streams, unfurls, file state). Before the fix
each one became a new user turn with the original text, and the bot answered it again.
"""

import asyncio
import importlib
import sys
import time
from importlib.machinery import PathFinder
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

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

BOT = "U0BOT00001"
USER = "U0USER0001"
TEXT = f"<@{BOT}> summarize the release notes"


def _slack_ts(seconds: float) -> str:
    return f"{seconds:.6f}"


def _adapter(delivered):
    """A freshly started adapter: what the gateway has right after a restart."""
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._bot_user_id = BOT
    adapter._resolve_user_name = AsyncMock(return_value="user")

    async def _capture(event):
        delivered.append(event)

    adapter.handle_message = _capture
    return adapter


def _changed_event(message_ts, *, edited_ts=None, previous_text=None, text=TEXT):
    """``message_changed`` for ``message_ts``, emitted now."""
    change_ts = _slack_ts(time.time())
    message = {
        "type": "message", "user": USER, "text": text, "ts": message_ts,
        "client_msg_id": "cmid-1", "reply_count": 1, "latest_reply": change_ts,
    }
    if edited_ts:
        message["edited"] = {"user": USER, "ts": edited_ts}
    event = {
        "type": "message", "subtype": "message_changed", "hidden": True,
        "channel": "C0CHAN0001", "channel_type": "channel", "team": "T0TEAM0001",
        "ts": change_ts, "event_ts": change_ts, "message": message,
    }
    if previous_text is not None:
        event["previous_message"] = {
            "type": "message", "user": USER, "text": previous_text, "ts": message_ts}
    return event


def _run(adapter, event):
    asyncio.run(adapter._handle_slack_message(event, {"team_id": "T0TEAM0001"}))


@pytest.mark.parametrize("shape", ["no_previous_message", "same_text", "edited_before_restart"])
def test_change_slack_made_to_a_message_from_before_the_restart_is_not_a_new_turn(shape):
    delivered = []
    adapter = _adapter(delivered)
    posted = time.time() - 600  # answered by the previous process, before this adapter started
    message_ts = _slack_ts(posted)
    event = {
        "no_previous_message": lambda: _changed_event(message_ts),
        "same_text": lambda: _changed_event(message_ts, previous_text=TEXT),
        "edited_before_restart": lambda: _changed_event(
            message_ts, edited_ts=_slack_ts(posted + 30), previous_text=TEXT),
    }[shape]()

    _run(adapter, event)

    assert delivered == []


def test_edit_made_after_the_restart_still_reaches_the_agent():
    """An @mention edited into an older, unanswered message still wakes the bot once."""
    delivered = []
    adapter = _adapter(delivered)
    message_ts = _slack_ts(time.time() - 600)
    event = _changed_event(message_ts, previous_text="summarize the release notes")
    event["message"]["edited"] = {"user": USER, "ts": event["event_ts"]}

    _run(adapter, event)

    assert len(delivered) == 1
    assert delivered[0].message_id == message_ts
