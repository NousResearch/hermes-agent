"""Bot-author turns may end in silence; human turns keep the fallback.

LOCAL/API surfaces keep ``SessionSource.is_bot`` False, so the bot flag only
arrives on ``turn_author``. ``execute_admission`` must surface it as
``reply_expected=False`` (persisted, so crash recovery agrees), and ACP must
not concatenate a held marker with the warning.
"""

from gateway.platforms.event import MessageEvent
from gateway.response_filters import silence_allowed
from gateway.session import Platform, SessionSource
from gateway.session_ingress import apply_bot_author_reply_expected


def _event(reply_expected=None):
    source = SessionSource(
        platform=Platform.LOCAL, chat_id="c", chat_type="dm")
    return MessageEvent(text="hi", source=source,
                        reply_expected=reply_expected)


def test_bot_author_marks_reply_not_expected():
    event = _event()
    apply_bot_author_reply_expected(
        event, {"id": "bot:x", "name": "x", "is_bot": True})
    assert event.reply_expected is False
    assert silence_allowed(None, event.reply_expected) is True


def test_human_author_leaves_unknown_reply_expected():
    event = _event()
    apply_bot_author_reply_expected(
        event, {"id": "u1", "name": "u", "is_bot": False})
    assert event.reply_expected is None
    assert silence_allowed(None, event.reply_expected) is False


def test_explicit_reply_expected_is_not_overwritten():
    event = _event(reply_expected=True)
    apply_bot_author_reply_expected(
        event, {"id": "bot:x", "name": "x", "is_bot": True})
    assert event.reply_expected is True


def test_missing_author_leaves_unknown_reply_expected():
    event = _event()
    apply_bot_author_reply_expected(event, None)
    assert event.reply_expected is None


def _remainder(streamed, final):
    from acp_adapter.gateway_reply import GatewayReplyMixin

    mixin = GatewayReplyMixin.__new__(GatewayReplyMixin)
    mixin._streamed = {"a": streamed}  # type: ignore[attr-defined]
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
    assert _remainder("hello", "hello world") == " world"
