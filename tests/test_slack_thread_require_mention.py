import asyncio
import os

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter, _apply_yaml_config


def run(coro):
    return asyncio.run(coro)


def make_adapter(extra=None):
    config = PlatformConfig(extra=extra or {})
    adapter = SlackAdapter(config)
    adapter._bot_user_id = "UBOT"
    adapter._team_bot_user_ids["T1"] = "UBOT"
    adapter._has_active_session_for_thread = lambda **_: False

    async def no_thread_context(**_):
        return ""

    async def no_parent_text(**_):
        return ""

    async def user_name(*_, **__):
        return "Sebastian"

    adapter._fetch_thread_context = no_thread_context
    adapter._fetch_thread_parent_text = no_parent_text
    adapter._resolve_user_name = user_name
    return adapter


def slack_event(text, ts="100.000", thread_ts=None):
    event = {
        "type": "message",
        "channel": "C123",
        "channel_type": "channel",
        "team": "T1",
        "user": "U123",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


def test_thread_require_mention_env_bridge(monkeypatch):
    monkeypatch.delenv("SLACK_THREAD_REQUIRE_MENTION", raising=False)

    _apply_yaml_config(
        {},
        {
            "thread_require_mention": True,
        },
    )

    assert os.environ["SLACK_THREAD_REQUIRE_MENTION"] == "true"


def test_thread_require_mention_parses_yaml_and_env(monkeypatch):
    monkeypatch.setenv("SLACK_THREAD_REQUIRE_MENTION", "true")

    assert make_adapter()._slack_thread_require_mention() is True
    assert (
        make_adapter({"thread_require_mention": "false"})._slack_thread_require_mention()
        is False
    )
    assert make_adapter({"thread_require_mention": True})._slack_thread_require_mention() is True


def test_thread_require_mention_allows_top_level_free_response():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention": False,
            "thread_require_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture

    run(adapter._handle_slack_message(slack_event("vpn is broken", ts="100.000")))

    assert len(handled) == 1
    assert handled[0].text == "vpn is broken"
    assert handled[0].source.thread_id == "100.000"


def test_thread_require_mention_blocks_unmentioned_thread_reply():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention": False,
            "thread_require_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture

    run(
        adapter._handle_slack_message(
            slack_event("we found another 403", ts="101.000", thread_ts="100.000")
        )
    )

    assert handled == []


def test_thread_require_mention_allows_mentioned_thread_reply_without_sticky_thread():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention": False,
            "thread_require_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture

    run(
        adapter._handle_slack_message(
            slack_event("<@UBOT> update this", ts="101.000", thread_ts="100.000")
        )
    )

    assert len(handled) == 1
    assert handled[0].text == "update this"
    assert "100.000" not in adapter._mentioned_threads

    run(
        adapter._handle_slack_message(
            slack_event("follow-up without mention", ts="102.000", thread_ts="100.000")
        )
    )

    assert len(handled) == 1
