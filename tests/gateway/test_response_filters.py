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


def test_misspelled_sentinel_is_intentional_silence():
    """#147526: the sentinel is typed by the model, so an autonomous lane occasionally
    misspells it; a whitespace-free token within <=2 edits of an ASCII marker must suppress
    delivery, or the control token reaches the chat as the whole message body."""
    assert is_intentional_silence_response("[SLIENT]")
    assert is_intentional_silence_response("[SBYLENT]")
    assert is_intentional_silence_response("SBYLENT")  # brackets dropped, #51438 style
    assert is_intentional_silence_response(".[SBILENT].")
    assert is_intentional_silence_response("NOREPL")  # NO_REPLY minus underscore and tail
    assert is_intentional_silence_response("[silent]")


def test_misspelled_sentinel_suppresses_the_cron_lane_too():
    """The loose autonomous rule inherits the typo tolerance: a misspelled marker on its
    own closing line still suppresses, matching the exact-marker behaviour."""
    assert is_autonomous_silence_response("[SBYLENT]")
    assert is_autonomous_silence_response("[SBYLENT]\nNothing new this tick.")
    assert is_autonomous_silence_response("2 deals filtered\n\n[SLIENT]")


def test_prose_and_distant_tokens_are_not_silence():
    """Fuzzy matching must stay a typo rule: prose (any whitespace), tokens >2 edits from
    every marker, and blank input keep being delivered/error paths."""
    assert not is_intentional_silence_response("SILENT please wait")
    assert not is_intentional_silence_response("[SBYLENT] drift committed, changelog updated")
    assert not is_intentional_silence_response("REPLY")
    assert not is_intentional_silence_response("OK")
    assert not is_intentional_silence_response("")


def test_translated_sentinel_polish_is_silence_in_every_form_the_english_one_is():
    """#110935 pattern in a pl deployment (production 2026-11: "[CICHO]"): the translated
    sentinel must suppress exactly like ``[SILENT]``, and its prose mentions must deliver."""
    assert is_intentional_silence_response("[CICHO]")
    assert is_intentional_silence_response("**CISZA**")
    assert is_autonomous_silence_response("[CICHO]\n\nNothing new this tick.")
    assert is_autonomous_silence_response("2 deals filtered\n\nCICHO")
    assert is_autonomous_silence_response("cicho")
    assert not is_intentional_silence_response("Cicho w lesie było")
    assert not is_autonomous_silence_response("odpowiedź brzmiała cicho i tyle")
