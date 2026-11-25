"""Bot-author turns may end in silence; human turns keep the fallback.

LOCAL/API surfaces keep ``SessionSource.is_bot`` False, so the bot flag only
arrives on ``turn_author``. ``execute_admission`` must surface a *trusted* bot
author as ``reply_expected=False`` (persisted, so crash recovery agrees), and
ACP must divide the final remainder by the *forwarded* prefix -- a held marker
was never shown, so it must not eat the head of a legitimate final.
"""

import json
from types import SimpleNamespace

from gateway.platforms.event import MessageEvent
from gateway.response_filters import silence_allowed
from gateway.session import Platform, SessionSource
from gateway.session_ingress import (
    apply_bot_author_reply_expected, trusted_bot_author)


def _event(reply_expected=None):
    source = SessionSource(
        platform=Platform.LOCAL, chat_id="c", chat_type="dm")
    return MessageEvent(text="hi", source=source,
                        reply_expected=reply_expected)


def test_trusted_bot_author_marks_reply_not_expected():
    event = _event()
    apply_bot_author_reply_expected(
        event, {"id": "bot:x", "name": "x", "is_bot": True}, trusted=True)
    assert event.reply_expected is False
    assert silence_allowed(None, event.reply_expected) is True


def test_untrusted_bot_claim_grants_nothing():
    event = _event()
    apply_bot_author_reply_expected(
        event, {"id": "bot:x", "name": "x", "is_bot": True})
    assert event.reply_expected is None


def test_human_author_leaves_unknown_reply_expected():
    event = _event()
    apply_bot_author_reply_expected(
        event, {"id": "u1", "name": "u", "is_bot": False}, trusted=True)
    assert event.reply_expected is None
    assert silence_allowed(None, event.reply_expected) is False


def test_explicit_reply_expected_is_not_overwritten():
    event = _event(reply_expected=True)
    apply_bot_author_reply_expected(
        event, {"id": "bot:x", "name": "x", "is_bot": True}, trusted=True)
    assert event.reply_expected is True


def test_missing_author_leaves_unknown_reply_expected():
    event = _event()
    apply_bot_author_reply_expected(event, None, trusted=True)
    assert event.reply_expected is None


def _row(payload):
    return {"payload": payload}


def test_trusted_bot_author_provenance():
    bot = {"id": "anyone", "name": "x", "is_bot": True}
    # Server-committed descriptor: trusts the flag as-is.
    assert trusted_bot_author(
        None, _row({"local_automation_v1": {"turn_author": bot}}), bot) is True
    # Client claims: only a bot id grants.
    assert trusted_bot_author(
        None, _row({"cli_turn_v1": {"turn_author": {"id": "bot:x", "name": "x", "is_bot": True}}}),
        {"id": "bot:x", "name": "x", "is_bot": True}) is True
    assert trusted_bot_author(
        None, _row({"cli_turn_v1": {"turn_author": bot}}), bot) is False
    assert trusted_bot_author(
        None, _row({"api_turn_v1": {"turn_author": {"id": "bot:x", "name": "x", "is_bot": True}}}),
        {"id": "bot:x", "name": "x", "is_bot": True}) is True
    assert trusted_bot_author(
        None, _row({"api_turn_v1": {"turn_author": bot}}), bot) is False
    # No author: nothing to trust.
    assert trusted_bot_author(None, _row({}), None) is False


def test_trusted_bot_author_a2a_forward():
    policy = SimpleNamespace(
        source="a2a", platform="a2a",
        request_json=json.dumps({"source": "a2a",
                                 "a2a_identity": ["a", "b", "peer-bot", "d"]}))
    assert trusted_bot_author(
        policy, _row({}), {"id": "peer-bot", "name": "peer-bot", "is_bot": True}) is True


def _remainder(streamed, final, *, forwarded=""):
    from acp_adapter.gateway_reply import GatewayReplyMixin

    mixin = GatewayReplyMixin.__new__(GatewayReplyMixin)
    mixin._streamed = {"a": streamed}  # type: ignore[attr-defined]
    mixin._forwarded = {"a": forwarded}  # type: ignore[attr-defined]
    mixin._segments = {}  # type: ignore[attr-defined]
    mixin._cut = {}  # type: ignore[attr-defined]
    return mixin._final_remainder("a", {"text": final})


def test_held_marker_plus_warning_shows_warning_alone():
    warning = "⚠️ The model returned only a silence marker"
    assert _remainder("NO_REPLY", warning) == warning


def test_marker_final_after_marker_is_silent():
    assert _remainder("NO_REPLY", "NO_REPLY") == ""


def test_normal_stream_still_concatenates():
    # Only a bare streamed marker stands alone.
    assert _remainder("hello", "hello world", forwarded="hello") == " world"


def test_held_prefix_never_eats_a_legitimate_final():
    # Deltas held as a possible marker forwarded nothing.
    assert _remainder("NO", "NO") == "NO"
    assert _remainder("N", "NO") == "NO"
    assert _remainder("NO", "NOT") == "NOT"


def test_final_remainder_pops_the_admission_state():
    from acp_adapter.gateway_reply import GatewayReplyMixin

    mixin = GatewayReplyMixin.__new__(GatewayReplyMixin)
    mixin._streamed = {"a": "hello"}  # type: ignore[attr-defined]
    mixin._forwarded = {"a": "hello"}  # type: ignore[attr-defined]
    mixin._segments = {}  # type: ignore[attr-defined]
    mixin._cut = {}  # type: ignore[attr-defined]
    assert mixin._final_remainder("a", {"text": "hello world"}) == " world"
    assert mixin._streamed == {} and mixin._forwarded == {}


def _managed_authority():
    from gateway.response_filters import silence_allowed

    source = SessionSource(platform=Platform.LOCAL, chat_id="c", chat_type="dm")
    runner = SimpleNamespace(
        _silence_allowed_for_source=lambda src, kind, expected, result: silence_allowed(
            kind, expected, agent_result=result))
    return SimpleNamespace(
        sessions={"s": SimpleNamespace(source=source)}, runner=runner)


def _shape(row, final):
    from gateway.session_managed_worker import _shape_managed_silence

    return _shape_managed_silence(
        _managed_authority(), SimpleNamespace(session_id="s"), row, None,
        {"final_response": final})


def test_managed_human_marker_becomes_the_visible_fallback():
    from gateway.run_turn import _unexpected_silence_reply

    shaped = _shape({"payload": {"text": "hi"}}, "NO_REPLY")
    assert shaped["final_response"] == _unexpected_silence_reply()


def test_managed_bot_marker_stays_silent():
    row = {"payload": {"cli_turn_v1": {
        "turn_author": {"id": "bot:x", "name": "x", "is_bot": True}}}}
    assert _shape(row, "NO_REPLY")["final_response"] == "NO_REPLY"


def test_managed_non_marker_is_untouched():
    assert _shape({"payload": {"text": "hi"}}, "answer")["final_response"] == "answer"
