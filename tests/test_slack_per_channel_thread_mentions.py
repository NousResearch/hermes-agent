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


def slack_event(text, ts="100.000", thread_ts=None, channel="C123"):
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


def run_handled(adapter, event):
    handled = []

    async def capture(ev):
        handled.append(ev)

    adapter.handle_message = capture
    run(adapter._handle_slack_message(event))
    return handled


def base_extra(**overrides):
    extra = {
        "allowed_channels": ["C123"],
        "require_mention": False,
        "thread_require_mention": True,
        "reply_in_thread": True,
    }
    extra.update(overrides)
    return extra


# --- Accessor resolution -----------------------------------------------------


def test_accessor_falls_back_to_global_when_channel_unset():
    # Global true, channel unlisted -> inherits global (True).
    assert make_adapter(base_extra())._slack_thread_mentions_enabled("C123") is True
    # Global false, channel unlisted -> inherits global (False).
    assert (
        make_adapter(base_extra(thread_require_mention=False))._slack_thread_mentions_enabled("C123")
        is False
    )


def test_accessor_reads_per_channel_override():
    adapter = make_adapter(
        base_extra(
            channels={"C123": {"thread_mentions_enabled": False}},
        )
    )
    assert adapter._slack_thread_mentions_enabled("C123") is False

    adapter2 = make_adapter(
        base_extra(
            thread_require_mention=False,
            channels={"C123": {"thread_mentions_enabled": True}},
        )
    )
    assert adapter2._slack_thread_mentions_enabled("C123") is True


def test_accessor_ignores_malformed_channel_map():
    extra = base_extra()
    extra["channels"] = {"C123": "not-a-dict"}  # noqa: E501
    assert make_adapter(extra)._slack_thread_mentions_enabled("C123") is True
    extra["channels"] = {"C123": {"other_flag": True}}
    assert make_adapter(extra)._slack_thread_mentions_enabled("C123") is True


# --- Gating behavior ---------------------------------------------------------


def test_enabled_channel_blocks_unmentioned_thread_reply_even_when_global_off():
    # Global thread_require_mention=false, but this channel overrides to true.
    adapter = make_adapter(
        base_extra(
            thread_require_mention=False,
            channels={"C123": {"thread_mentions_enabled": True}},
        )
    )
    handled = run_handled(
        adapter,
        slack_event("follow-up without mention", ts="101.000", thread_ts="100.000"),
    )
    assert handled == []


def test_disabled_channel_allows_unmentioned_thread_reply_even_when_global_on():
    # Global thread_require_mention=true, this channel overrides to false -> auto-follow.
    adapter = make_adapter(
        base_extra(
            channels={"C123": {"thread_mentions_enabled": False}},
        )
    )
    handled = run_handled(
        adapter,
        slack_event("follow-up without mention", ts="101.000", thread_ts="100.000"),
    )
    assert len(handled) == 1
    assert handled[0].text == "follow-up without mention"


def test_unset_channel_inherits_global_true_blocks():
    adapter = make_adapter(base_extra())  # no channels map, global true
    handled = run_handled(
        adapter,
        slack_event("follow-up without mention", ts="101.000", thread_ts="100.000"),
    )
    assert handled == []


def test_unset_channel_inherits_global_false_allows():
    adapter = make_adapter(base_extra(thread_require_mention=False))
    handled = run_handled(
        adapter,
        slack_event("follow-up without mention", ts="101.000", thread_ts="100.000"),
    )
    assert len(handled) == 1
    assert handled[0].text == "follow-up without mention"


def test_mentioned_reply_allowed_in_enabled_channel():
    adapter = make_adapter(
        base_extra(
            channels={"C123": {"thread_mentions_enabled": True}},
        )
    )
    handled = run_handled(
        adapter,
        slack_event("<@UBOT> update this", ts="101.000", thread_ts="100.000"),
    )
    assert len(handled) == 1
    assert handled[0].text == "update this"
    # thread gating is on for this channel -> thread NOT sticky-registered.
    assert "100.000" not in adapter._mentioned_threads


def test_disabled_channel_registers_mentioned_thread():
    # With per-channel thread_mentions_enabled=false, the thread becomes sticky
    # so later unmentioned replies keep flowing (auto-follow).
    adapter = make_adapter(
        base_extra(
            channels={"C123": {"thread_mentions_enabled": False}},
        )
    )
    run_handled(adapter, slack_event("<@UBOT> update this", ts="101.000", thread_ts="100.000"))
    assert ("T1", "100.000") in adapter._mentioned_threads


def test_group_dm_channel_id_honored():
    # Per-channel override keyed by a group-DM (MPIM) channel id.
    adapter = make_adapter(
        base_extra(
            allowed_channels=["C123", "G7890"],
            channels={"G7890": {"thread_mentions_enabled": False}},
        )
    )
    handled = run_handled(
        adapter,
        slack_event(
            "follow-up without mention",
            ts="101.000",
            thread_ts="100.000",
            channel="G7890",
        ),
    )
    assert len(handled) == 1
    assert handled[0].text == "follow-up without mention"
