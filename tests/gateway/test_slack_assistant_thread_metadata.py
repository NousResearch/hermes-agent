"""Regression test for #29921 — _extract_assistant_thread_metadata
must not fall back to event.get("message_ts") as a thread_ts source.

On a top-level channel message, `message_ts` is the message's own ts.
Using it as thread_ts injects a synthetic thread identifier into
session metadata, which propagates downstream and defeats
`reply_in_thread: false` (the resolver sees a non-empty thread_id and
replies in-thread anyway).
"""

import sys
from unittest.mock import MagicMock

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
        ("slack_bolt.adapter.socket_mode.async_handler",
         slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import gateway.platforms.slack as _slack_mod  # noqa: E402
_slack_mod.SLACK_AVAILABLE = True
from gateway.platforms.slack import SlackAdapter  # noqa: E402


@pytest.fixture()
def adapter():
    return SlackAdapter(PlatformConfig(enabled=True, token="***"))


def test_extract_thread_metadata_does_not_use_message_ts_as_thread_ts(adapter):
    """Top-level message with only message_ts must yield empty thread_ts."""
    event = {
        "channel": "C123",
        "user": "U_USER",
        "team": "T1",
        "message_ts": "1700000000.000100",  # message's own ts, NOT a thread
        # no assistant_thread, no event.thread_ts
    }
    md = adapter._extract_assistant_thread_metadata(event)
    assert md["channel_id"] == "C123"
    assert md["user_id"] == "U_USER"
    assert md["thread_ts"] == "", (
        "message_ts must not be used as a thread_ts fallback "
        "(#29921: this synthetic value defeats reply_in_thread=false)"
    )


def test_extract_thread_metadata_uses_assistant_thread_thread_ts(adapter):
    event = {
        "assistant_thread": {
            "channel_id": "C_AT",
            "thread_ts": "1700000000.000200",
            "user_id": "U1",
        },
    }
    md = adapter._extract_assistant_thread_metadata(event)
    assert md["channel_id"] == "C_AT"
    assert md["thread_ts"] == "1700000000.000200"
    assert md["user_id"] == "U1"


def test_extract_thread_metadata_uses_event_thread_ts(adapter):
    event = {
        "channel": "C2",
        "user": "U2",
        "thread_ts": "1700000000.000300",  # legitimate in-thread reply
        "message_ts": "1700000000.000400",
    }
    md = adapter._extract_assistant_thread_metadata(event)
    # event.thread_ts wins over message_ts; message_ts is never a fallback
    assert md["thread_ts"] == "1700000000.000300"
