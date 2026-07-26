import asyncio
import os
from types import SimpleNamespace

from gateway.config import PlatformConfig
from gateway.session import build_session_key
from plugins.platforms.slack.adapter import SlackAdapter, _apply_yaml_config
from tools import clarify_gateway


def run(coro):
    return asyncio.run(coro)


def make_adapter(extra=None):
    config = PlatformConfig(extra=extra or {})
    adapter = SlackAdapter(config)
    adapter._bot_user_id = "UBOT"
    adapter._team_bot_user_ids["T1"] = "UBOT"
    adapter._user_is_bot_cache[("T1", "U123")] = False
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


def slack_event(text, ts="100.000", thread_ts=None, **overrides):
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
    event.update(overrides)
    return event


def arm_thread_clarify(
    adapter,
    clarify_id,
    *,
    awaiting_text=True,
    profile=None,
    stamp_adapter_profile=True,
):
    config = SimpleNamespace(
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
        multiplex_profiles=bool(profile),
    )

    def generate_session_key(source):
        return build_session_key(
            source,
            group_sessions_per_user=config.group_sessions_per_user,
            thread_sessions_per_user=config.thread_sessions_per_user,
            profile=(source.profile or profile) if config.multiplex_profiles else None,
        )

    adapter._session_store = SimpleNamespace(
        config=config,
        _generate_session_key=generate_session_key,
    )
    if profile and stamp_adapter_profile:
        adapter._gateway_profile_name = profile
    session_key = (
        f"agent:{profile}:slack:group:T1:C123:100.000"
        if profile
        else "agent:main:slack:group:T1:C123:100.000"
    )
    clarify_gateway.register(
        clarify_id,
        session_key,
        "How should this be handled?",
        ["Skip", "Process"],
    )
    if awaiting_text:
        assert clarify_gateway.mark_awaiting_text(clarify_id) is True
    return session_key


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


def test_pending_clarify_text_reply_bypasses_strict_mention_gate():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture
    session_key = arm_thread_clarify(adapter, "strict-mention-other-answer")

    try:
        run(
            adapter._handle_slack_message(
                slack_event(
                    "convert each job and apply the payment",
                    ts="101.000",
                    thread_ts="100.000",
                )
            )
        )
    finally:
        clarify_gateway.clear_session(session_key)

    assert len(handled) == 1
    assert handled[0].text == "convert each job and apply the payment"
    assert (
        handled[0].metadata["_hermes_clarify_response_only"]
        == "strict-mention-other-answer"
    )


def test_pending_clarify_bypass_respects_registered_authorization_check():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []
    authorization_calls = []
    unauthorized_side_effects = []

    async def capture(event):
        handled.append(event)

    async def forbidden_fetch(**_):
        unauthorized_side_effects.append("thread-fetch")
        return ""

    async def forbidden_name(*_, **__):
        unauthorized_side_effects.append("name-lookup")
        return "unauthorized"

    def remember_channel(*_):
        unauthorized_side_effects.append("channel-watermark")

    def deny(user_id, chat_type, chat_id):
        authorization_calls.append((user_id, chat_type, chat_id))
        return False

    adapter.handle_message = capture
    adapter._fetch_thread_context = forbidden_fetch
    adapter._fetch_thread_parent_text = forbidden_fetch
    adapter._resolve_user_name = forbidden_name
    adapter._remember_channel_team = remember_channel
    adapter.set_authorization_check(deny)
    session_key = arm_thread_clarify(adapter, "strict-mention-unauthorized-answer")

    try:
        run(
            adapter._handle_slack_message(
                slack_event(
                    "unauthorized typed answer",
                    ts="101.000",
                    thread_ts="100.000",
                    client_msg_id="cmid-unauthorized-human",
                )
            )
        )
    finally:
        clarify_gateway.clear_session(session_key)

    assert authorization_calls == [("U123", "group", "C123")]
    assert unauthorized_side_effects == []
    assert handled == []


def test_pending_clarify_authorization_exception_fails_closed():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    def explode(*_):
        raise RuntimeError("authorization unavailable")

    adapter.handle_message = capture
    adapter.set_authorization_check(explode)
    session_key = arm_thread_clarify(adapter, "strict-mention-auth-error-answer")

    try:
        run(
            adapter._handle_slack_message(
                slack_event(
                    "typed answer during auth failure",
                    ts="101.000",
                    thread_ts="100.000",
                    client_msg_id="cmid-auth-error-human",
                )
            )
        )
    finally:
        clarify_gateway.clear_session(session_key)

    assert handled == []


def test_pending_clarify_uses_non_default_multiplexed_profile_session_key():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.set_message_handler(capture)
    session_key = arm_thread_clarify(
        adapter,
        "strict-mention-multiplexed-answer",
        profile="aya",
    )
    adapter._active_sessions[session_key] = asyncio.Event()

    def fail_new_session(*_):
        raise AssertionError("multiplexed clarify reply started a new agent turn")

    adapter._start_session_processing = fail_new_session

    try:
        run(
            adapter._handle_slack_message(
                slack_event(
                    "use the non-default profile",
                    ts="101.000",
                    thread_ts="100.000",
                )
            )
        )
    finally:
        adapter._active_sessions.pop(session_key, None)
        clarify_gateway.clear_session(session_key)

    assert len(handled) == 1
    assert handled[0].source.profile == "aya"
    assert (
        handled[0].metadata["_hermes_clarify_response_only"]
        == "strict-mention-multiplexed-answer"
    )


def test_pending_clarify_uses_canonical_active_profile_session_key():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.set_message_handler(capture)
    session_key = arm_thread_clarify(
        adapter,
        "strict-mention-active-profile-answer",
        profile="aya",
        stamp_adapter_profile=False,
    )
    adapter._active_sessions[session_key] = asyncio.Event()

    def fail_new_session(*_):
        raise AssertionError("active-profile clarify reply started a new agent turn")

    adapter._start_session_processing = fail_new_session

    try:
        run(
            adapter._handle_slack_message(
                slack_event(
                    "use the active multiplexed profile",
                    ts="101.000",
                    thread_ts="100.000",
                )
            )
        )
    finally:
        adapter._active_sessions.pop(session_key, None)
        clarify_gateway.clear_session(session_key)

    assert len(handled) == 1
    assert handled[0].source.profile is None
    assert (
        handled[0].metadata["_hermes_clarify_response_only"]
        == "strict-mention-active-profile-answer"
    )


def test_unmentioned_bot_cannot_answer_pending_clarify_with_allow_bots_all():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "allow_bots": "all",
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture
    session_key = arm_thread_clarify(adapter, "strict-mention-peer-bot-answer")

    try:
        run(
            adapter._handle_slack_message(
                slack_event(
                    "peer bot output",
                    ts="101.000",
                    thread_ts="100.000",
                    user="UPEERBOT",
                    bot_id="BPEERBOT",
                    subtype="bot_message",
                )
            )
        )
    finally:
        clarify_gateway.clear_session(session_key)

    assert handled == []


def test_resolved_bot_user_identity_reaches_final_authorization_source():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "allow_bots": "mentions",
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def bot_users_info(**_):
        return {
            "ok": True,
            "user": {
                "id": "UPEERBOT",
                "is_bot": True,
                "profile": {"display_name": "Peer Bot"},
            },
        }

    async def capture(event):
        handled.append(event)

    adapter._app = SimpleNamespace(client=SimpleNamespace(users_info=bot_users_info))
    adapter.handle_message = capture
    adapter.set_authorization_check(lambda *_: False)

    run(
        adapter._handle_slack_message(
            slack_event(
                "<@UBOT> resolved bot status",
                ts="101.000",
                thread_ts="100.000",
                user="UPEERBOT",
                client_msg_id=None,
            )
        )
    )

    assert len(handled) == 1
    assert handled[0].source.is_bot is True
    assert handled[0].source.user_id == "UPEERBOT"


def test_registered_human_auth_gate_preserves_explicit_workflow_bot_traffic():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "allow_bots": "mentions",
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []
    authorization_calls = []

    async def capture(event):
        handled.append(event)

    def human_authorization(*args):
        authorization_calls.append(args)
        return True

    adapter.handle_message = capture
    adapter.set_authorization_check(human_authorization)

    run(
        adapter._handle_slack_message(
            slack_event(
                "<@UBOT> workflow status",
                ts="101.000",
                thread_ts="100.000",
                user="",
                bot_id="BWORKFLOW",
                subtype="bot_message",
            )
        )
    )

    assert authorization_calls == []
    assert len(handled) == 1
    assert handled[0].source.is_bot is True
    assert "_hermes_clarify_response_only" not in handled[0].metadata


def test_known_self_bot_user_cannot_take_pending_clarify_bypass():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "allow_bots": "all",
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture
    session_key = arm_thread_clarify(adapter, "strict-mention-self-bot-answer")

    try:
        run(
            adapter._handle_slack_message(
                slack_event(
                    "self bot output without bot markers",
                    ts="101.000",
                    thread_ts="100.000",
                    user="UBOT",
                    client_msg_id="cmid-self-bot",
                )
            )
        )
    finally:
        clarify_gateway.clear_session(session_key)

    assert handled == []


def test_unknown_bot_status_cannot_take_pending_clarify_bypass():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "allow_bots": "all",
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture
    session_key = arm_thread_clarify(adapter, "strict-mention-unknown-bot-answer")

    try:
        run(
            adapter._handle_slack_message(
                slack_event(
                    "ordinary-looking app bot output",
                    ts="101.000",
                    thread_ts="100.000",
                    user="UUNKNOWNBOT",
                )
            )
        )
    finally:
        clarify_gateway.clear_session(session_key)

    assert handled == []


def test_malformed_user_name_lookup_cannot_poison_clarify_bot_identity():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "allow_bots": "all",
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def malformed_users_info(**_):
        return {"ok": False, "error": "transient_failure"}

    async def capture(event):
        handled.append(event)

    adapter._app = SimpleNamespace(
        client=SimpleNamespace(users_info=malformed_users_info)
    )
    adapter.handle_message = capture
    adapter.set_authorization_check(lambda *_: True)

    assert (
        run(
            SlackAdapter._resolve_user_name(
                adapter,
                "UPOISON",
                chat_id="C123",
                team_id="T1",
            )
        )
        == "UPOISON"
    )
    assert ("T1", "UPOISON") not in adapter._user_is_bot_cache

    session_key = arm_thread_clarify(adapter, "strict-mention-poison-proof")
    try:
        run(
            adapter._handle_slack_message(
                slack_event(
                    "unmentioned answer after malformed lookup",
                    ts="101.000",
                    thread_ts="100.000",
                    user="UPOISON",
                    client_msg_id="cmid-poison-attempt",
                )
            )
        )
    finally:
        clarify_gateway.clear_session(session_key)

    assert handled == []


def test_failed_user_lookup_with_human_shaped_payload_stays_unknown():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "allow_bots": "all",
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def failed_users_info(**_):
        return {
            "ok": False,
            "error": "transient_failure",
            "user": {"is_bot": False, "profile": {"display_name": "Unknown"}},
        }

    async def capture(event):
        handled.append(event)

    adapter._app = SimpleNamespace(client=SimpleNamespace(users_info=failed_users_info))
    adapter.handle_message = capture
    adapter.set_authorization_check(lambda *_: True)

    assert (
        run(
            adapter._resolve_user_is_bot(
                "UFAILED",
                chat_id="C123",
                team_id="T1",
            )
        )
        is None
    )
    assert ("T1", "UFAILED") not in adapter._user_is_bot_cache

    session_key = arm_thread_clarify(adapter, "strict-mention-failed-lookup")
    try:
        run(
            adapter._handle_slack_message(
                slack_event(
                    "unmentioned answer after failed lookup",
                    ts="101.000",
                    thread_ts="100.000",
                    user="UFAILED",
                    client_msg_id="cmid-failed-lookup",
                )
            )
        )
    finally:
        clarify_gateway.clear_session(session_key)

    assert handled == []


def test_malformed_user_lookup_does_not_assert_human_identity():
    adapter = make_adapter()

    async def malformed_users_info(**_):
        return {"ok": False, "error": "transient_failure"}

    adapter._app = SimpleNamespace(
        client=SimpleNamespace(users_info=malformed_users_info)
    )

    assert (
        run(
            adapter._resolve_user_is_bot(
                "UUNKNOWNBOT",
                chat_id="C123",
                team_id="T1",
            )
        )
        is None
    )
    assert ("T1", "UUNKNOWNBOT") not in adapter._user_is_bot_cache


def test_pending_clarify_does_not_bypass_strict_mention_for_commands():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture
    session_key = arm_thread_clarify(adapter, "strict-mention-command-answer")

    try:
        run(
            adapter._handle_slack_message(
                slack_event("!stop", ts="101.000", thread_ts="100.000")
            )
        )
    finally:
        clarify_gateway.clear_session(session_key)

    assert handled == []


def test_native_choice_clarify_does_not_open_strict_thread_to_unrelated_text():
    adapter = make_adapter(
        {
            "allowed_channels": ["C123"],
            "require_mention": True,
            "strict_mention": True,
            "reply_in_thread": True,
        }
    )
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture
    session_key = arm_thread_clarify(
        adapter,
        "strict-mention-native-choice",
        awaiting_text=False,
    )

    try:
        run(
            adapter._handle_slack_message(
                slack_event(
                    "unrelated thread chatter",
                    ts="101.000",
                    thread_ts="100.000",
                )
            )
        )
    finally:
        clarify_gateway.clear_session(session_key)

    assert handled == []
