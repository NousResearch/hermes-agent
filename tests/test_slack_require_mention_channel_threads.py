import asyncio

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


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


def capture_messages(adapter):
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture
    return handled


def test_require_mention_channel_threads_parses_env_yaml_list_and_csv(monkeypatch):
    monkeypatch.delenv("SLACK_REQUIRE_MENTION_CHANNEL_THREADS", raising=False)
    assert make_adapter()._slack_require_mention_channel_threads() == set()

    monkeypatch.setenv(
        "SLACK_REQUIRE_MENTION_CHANNEL_THREADS", " CENV1, CENV2, , CENV1 "
    )
    assert make_adapter()._slack_require_mention_channel_threads() == {
        "CENV1",
        "CENV2",
    }

    assert make_adapter(
        {"require_mention_channel_threads": [" C123 ", "", "C456"]}
    )._slack_require_mention_channel_threads() == {"C123", "C456"}
    assert make_adapter(
        {"require_mention_channel_threads": " C789, C012, , C789 "}
    )._slack_require_mention_channel_threads() == {"C789", "C012"}


def test_unmentioned_thread_reply_in_listed_channel_is_ignored_before_wake_checks():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention_channel_threads": ["C123"],
            "reply_in_thread": True,
        }
    )
    adapter._bot_message_ts.add("101.000")
    adapter._mentioned_threads.add("101.000")
    adapter._has_active_session_for_thread = lambda **_: True

    async def mentioned_parent(**_):
        return "<@UBOT> please follow up"

    adapter._fetch_thread_parent_text = mentioned_parent
    handled = capture_messages(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("follow-up", ts="102.000", thread_ts="101.000")
        )
    )

    assert handled == []


def test_mentioned_thread_reply_in_listed_channel_proceeds():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention_channel_threads": ["C123"],
            "reply_in_thread": True,
        }
    )
    handled = capture_messages(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("<@UBOT> follow up", ts="102.000", thread_ts="101.000")
        )
    )

    assert len(handled) == 1
    assert handled[0].text == "follow up"


def test_unmentioned_top_level_message_in_listed_channel_is_unchanged():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention": False,
            "require_mention_channel_threads": ["C123"],
            "reply_in_thread": True,
        }
    )
    handled = capture_messages(adapter)

    run(adapter._handle_slack_message(slack_event("top-level update")))

    assert len(handled) == 1
    assert handled[0].text == "top-level update"


def test_unmentioned_thread_reply_in_non_listed_channel_is_unchanged():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention_channel_threads": ["C456"],
            "reply_in_thread": True,
        }
    )
    adapter._has_active_session_for_thread = lambda **_: True
    handled = capture_messages(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("follow-up", ts="102.000", thread_ts="101.000")
        )
    )

    assert len(handled) == 1
    assert handled[0].text == "follow-up"


def test_listed_channel_gate_wins_over_free_response_for_thread_replies():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "free_response_channels": ["C123"],
            "require_mention_channel_threads": ["C123"],
            "reply_in_thread": True,
        }
    )
    handled = capture_messages(adapter)

    run(
        adapter._handle_slack_message(
            slack_event("follow-up", ts="102.000", thread_ts="101.000")
        )
    )

    assert handled == []
