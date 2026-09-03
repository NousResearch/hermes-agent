import asyncio
import os

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter, _apply_yaml_config


NATURAL_CHANNEL = "C_NATURAL"
STRICT_CHANNEL = "C_STRICT"
BOT_USER_ID = "U_BOT"


def run(coro):
    return asyncio.run(coro)


def make_adapter(extra=None):
    config = PlatformConfig(
        extra={
            "allowed_channels": [NATURAL_CHANNEL, STRICT_CHANNEL],
            "require_mention": True,
            "strict_mention": True,
            "thread_require_mention": True,
            "natural_thread_channels": [NATURAL_CHANNEL],
            "reply_in_thread": True,
            **(extra or {}),
        }
    )
    adapter = SlackAdapter(config)
    adapter._bot_user_id = BOT_USER_ID
    adapter._team_bot_user_ids["T1"] = BOT_USER_ID
    adapter._has_active_session_for_thread = lambda **_: False

    async def no_thread_context(**_):
        return ""

    async def no_parent_text(**_):
        return ""

    async def user_name(*_, **__):
        return "Test User"

    adapter._fetch_thread_context = no_thread_context
    adapter._fetch_thread_parent_text = no_parent_text
    adapter._resolve_user_name = user_name
    return adapter


def slack_event(text, *, channel, ts="100.000", thread_ts=None):
    event = {
        "type": "message",
        "channel": channel,
        "channel_type": "channel",
        "team": "T1",
        "user": "U123",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


def test_natural_thread_channels_parse_list_csv_and_env(monkeypatch):
    monkeypatch.delenv("SLACK_NATURAL_THREAD_CHANNELS", raising=False)
    assert make_adapter()._slack_natural_thread_channels() == {NATURAL_CHANNEL}
    assert make_adapter({"natural_thread_channels": "C1, C2"})._slack_natural_thread_channels() == {
        "C1",
        "C2",
    }

    monkeypatch.setenv("SLACK_NATURAL_THREAD_CHANNELS", "C9,C10")
    adapter = make_adapter({"natural_thread_channels": None})
    assert adapter._slack_natural_thread_channels() == {"C9", "C10"}


def test_natural_thread_channels_yaml_bridge_is_profile_local(monkeypatch):
    monkeypatch.delenv("SLACK_NATURAL_THREAD_CHANNELS", raising=False)
    first_extra = _apply_yaml_config(
        {}, {"natural_thread_channels": ["C1", "C2"]}
    )
    assert os.environ["SLACK_NATURAL_THREAD_CHANNELS"] == "C1,C2"
    assert first_extra == {"natural_thread_channels": ["C1", "C2"]}

    second_extra = _apply_yaml_config(
        {}, {"natural_thread_channels": ["C9"]}
    )
    assert second_extra == {"natural_thread_channels": ["C9"]}
    assert SlackAdapter(PlatformConfig(extra=first_extra))._slack_natural_thread_channels() == {
        "C1",
        "C2",
    }
    assert SlackAdapter(PlatformConfig(extra=second_extra))._slack_natural_thread_channels() == {
        "C9",
    }


def test_natural_thread_channels_do_not_borrow_process_env_in_multiplex(monkeypatch):
    monkeypatch.setenv("SLACK_NATURAL_THREAD_CHANNELS", "C_PRIMARY")
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)

    secondary = SlackAdapter(PlatformConfig(extra={}))

    assert secondary._slack_natural_thread_channels() == set()


def test_scoped_multiplex_yaml_bridge_does_not_mutate_process_env(monkeypatch):
    from agent.secret_scope import (
        is_multiplex_active,
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )

    monkeypatch.delenv("SLACK_NATURAL_THREAD_CHANNELS", raising=False)
    monkeypatch.delenv("SLACK_ALLOW_BOTS", raising=False)
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    token = set_secret_scope({})
    try:
        extra = _apply_yaml_config(
            {},
            {
                "natural_thread_channels": ["C_SECONDARY"],
                "allow_bots": "mentions",
            },
        )
    finally:
        reset_secret_scope(token)
        set_multiplex_active(previous_multiplex)

    assert "SLACK_NATURAL_THREAD_CHANNELS" not in os.environ
    assert "SLACK_ALLOW_BOTS" not in os.environ
    assert extra == {
        "natural_thread_channels": ["C_SECONDARY"],
        "allow_bots": "mentions",
    }


def test_scoped_multiplex_allowed_channels_stay_adapter_local(monkeypatch):
    from agent.secret_scope import (
        is_multiplex_active,
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )

    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", "C_PRIMARY")
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    token = set_secret_scope({})
    try:
        extra = _apply_yaml_config({}, {"allowed_channels": ["C_SECONDARY"]})
        configured = SlackAdapter(PlatformConfig(extra=extra or {}))
        unconfigured = SlackAdapter(PlatformConfig(extra={}))
        configured_channels = configured._slack_allowed_channels()
        unconfigured_channels = unconfigured._slack_allowed_channels()
    finally:
        reset_secret_scope(token)
        set_multiplex_active(previous_multiplex)

    assert os.environ["SLACK_ALLOWED_CHANNELS"] == "C_PRIMARY"
    assert extra == {"allowed_channels": ["C_SECONDARY"]}
    assert configured_channels == {"C_SECONDARY"}
    assert unconfigured_channels == set()


def test_natural_channel_requires_top_level_mention_then_allows_plain_thread_reply():
    adapter = make_adapter()
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture

    run(
        adapter._handle_slack_message(
            slack_event("ambient chatter", channel=NATURAL_CHANNEL, ts="100.000")
        )
    )
    assert handled == []

    run(
        adapter._handle_slack_message(
            slack_event(
                f"<@{BOT_USER_ID}> start here",
                channel=NATURAL_CHANNEL,
                ts="101.000",
            )
        )
    )
    assert [event.text for event in handled] == ["start here"]
    assert ("T1", "101.000") in adapter._mentioned_threads

    run(
        adapter._handle_slack_message(
            slack_event(
                "plain follow-up",
                channel=NATURAL_CHANNEL,
                ts="102.000",
                thread_ts="101.000",
            )
        )
    )
    assert [event.text for event in handled] == ["start here", "plain follow-up"]


def test_natural_channel_top_level_gate_wins_over_free_response():
    adapter = make_adapter(
        {
            "require_mention": False,
            "free_response_channels": [NATURAL_CHANNEL],
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture

    run(
        adapter._handle_slack_message(
            slack_event("ambient chatter", channel=NATURAL_CHANNEL, ts="150.000")
        )
    )
    assert handled == []

    run(
        adapter._handle_slack_message(
            slack_event(
                f"<@{BOT_USER_ID}> explicit start",
                channel=NATURAL_CHANNEL,
                ts="151.000",
            )
        )
    )
    assert [event.text for event in handled] == ["explicit start"]


def test_unlisted_channel_stays_strict_on_every_turn():
    adapter = make_adapter()
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture

    run(
        adapter._handle_slack_message(
            slack_event(
                f"<@{BOT_USER_ID}> strict start",
                channel=STRICT_CHANNEL,
                ts="200.000",
            )
        )
    )
    assert [event.text for event in handled] == ["strict start"]
    assert "200.000" not in adapter._mentioned_threads

    run(
        adapter._handle_slack_message(
            slack_event(
                "plain strict follow-up",
                channel=STRICT_CHANNEL,
                ts="201.000",
                thread_ts="200.000",
            )
        )
    )
    assert [event.text for event in handled] == ["strict start"]
