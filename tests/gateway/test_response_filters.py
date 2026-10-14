from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.response_filters import (
    INTERNAL_NOTIFICATION_DISPLAY_KIND,
    display_kind_for_event,
    is_autonomous_silence_response,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
    is_machinery_display_kind,
)
from gateway.session import SessionSource


def test_webhook_route_event_is_machinery_but_a_human_turn_is_not():
    """A webhook route is built by the adapter after HMAC auth. It must count as
    machinery so a bare silence marker is not rewritten into the user-facing
    warning. A human platform, and the separate Microsoft Graph webhook, stay
    ordinary turns."""
    webhook = MessageEvent(
        text="[SILENT]",
        source=SessionSource(
            platform=Platform.WEBHOOK,
            chat_id="webhook:mail:delivery-1",
            chat_type="webhook",
            user_id="webhook:mail",
        ),
    )
    kind = display_kind_for_event(webhook)
    assert kind == INTERNAL_NOTIFICATION_DISPLAY_KIND
    assert is_machinery_display_kind(kind)

    human = MessageEvent(
        text="[SILENT]",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm"),
    )
    assert display_kind_for_event(human) is None

    graph = MessageEvent(
        text="[SILENT]",
        source=SessionSource(platform=Platform.MSGRAPH_WEBHOOK, chat_id="g1", chat_type="dm"),
    )
    assert display_kind_for_event(graph) is None


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_autonomous_silence_accepts_marker_with_own_line_note():
    """The loose rule for cron/webhook lanes: marker + explanation suppresses."""
    assert is_autonomous_silence_response("[SILENT]")
    assert is_autonomous_silence_response("[SILENT]\n\nNothing new this tick.")
    assert is_autonomous_silence_response("2 deals filtered\n\n[SILENT]")
    assert is_autonomous_silence_response("no_reply\nduplicate inbound, already handled")
    assert is_autonomous_silence_response("[SILENT] No changes detected")


def test_translated_sentinel_is_silence_in_every_form_the_english_one_is():
    """#110935: a lane that answers the cron instruction in its own language translates the
    sentinel; ``[静默]`` must suppress delivery exactly like ``[SILENT]`` (exact, own-line note,
    reordered lines, bracketless, edge punctuation)."""
    assert is_intentional_silence_response("[静默]")
    assert is_intentional_silence_response("**沉默**")
    assert is_autonomous_silence_response("[静默]\n\nNothing new this tick.")
    assert is_autonomous_silence_response("2 deals filtered\n\n[沉默]")
    assert is_autonomous_silence_response("静默")


def test_prose_mentioning_the_translated_sentinel_is_delivered():
    assert not is_intentional_silence_response("status: 静默 means the lane is quiet")
    assert not is_autonomous_silence_response("the lane said 静默 mid-sentence and kept talking")


def test_autonomous_lane_agrees_with_interactive_lane_on_cjk_punctuation_variants():
    """A Chinese lane emits fullwidth brackets or a trailing ``。``; cron/webhook must suppress
    exactly what the interactive predicate suppresses, or the two lanes drift on the new tokens."""
    for variant in ("【静默】", "静默。", "【沉默】", "沉默。", "**[静默]**", "NO_REPLY."):
        assert is_intentional_silence_response(variant)
        assert is_autonomous_silence_response(variant) == is_intentional_silence_response(variant), variant
