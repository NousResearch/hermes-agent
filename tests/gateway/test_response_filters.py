from gateway.response_filters import (
    is_autonomous_silence_response,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
)


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


def test_translated_sentinel_suppresses_delivery():
    """Regression: a non-English lane answers the silence instruction in its own language.

    2026-09-14 — a deepseek cron lane had nothing to report and answered the cron
    instruction ("respond with exactly [SILENT]") with "[静默]". The control token
    was not a known marker, so the whole token was delivered to the user's DM as
    content.
    """
    assert is_intentional_silence_response("[静默]")
    assert is_autonomous_silence_response("[静默]")


def test_translated_sentinel_takes_the_same_forms_as_the_english_one():
    """Same loose rule: own-line note, reordered lines, bracketless; the exact rule strips punctuation."""
    assert is_autonomous_silence_response("[静默]\n\nNothing new this tick.")
    assert is_autonomous_silence_response("2 deals filtered\n\n[沉默]")
    assert is_autonomous_silence_response("静默")
    assert is_intentional_silence_response("**沉默**")


def test_prose_mentioning_the_translated_sentinel_is_delivered():
    assert not is_intentional_silence_response("status: 静默 means the lane is quiet")
    assert not is_autonomous_silence_response("the lane said 静默 mid-sentence and kept talking")
