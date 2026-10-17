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


def test_trailing_bracketed_marker_on_the_prose_line_is_autonomous_silence():
    """Prose + marker as the final SENTENCE on the same line is the same intent as a
    marker on its own line — the model only forgot the newline — and the bare sentinel
    must never ship. A marker integrated grammatically (after a colon, comma or
    preposition) is the marker quoted as data and stays delivered."""
    assert is_autonomous_silence_response("All clear today. [SILENT]")
    assert is_autonomous_silence_response("2 deals filtered. [静默]")
    assert is_autonomous_silence_response("nothing to report. [SILENT].")
    assert is_autonomous_silence_response("一切正常。[沉默]")
    # council follow-up: wrapping decoration and ellipsis are still a tacked-on sentence
    assert is_autonomous_silence_response("All clear today. ([SILENT])")
    assert is_autonomous_silence_response('All clear today. "[SILENT]"')
    assert is_autonomous_silence_response("Waiting for the build… [SILENT]")
    # the marker quoted as data: integrated, not a tacked-on sentence
    assert not is_autonomous_silence_response("Watchdog probe result: [SILENT]")
    assert not is_autonomous_silence_response("The agent replied with [静默]")
    assert not is_autonomous_silence_response("Summary continues, then [SILENT]")
    # bare (unbracketed) trailing words stay delivered — same safety as the prefix rule
    assert not is_autonomous_silence_response("the retry stayed silent")
    assert not is_autonomous_silence_response("nothing to report. NO_REPLY")
    # negatives: the bracketed token sits inside the line, prose continues past it
    assert not is_autonomous_silence_response("the update mentions [SILENT] but then continues")
    assert not is_autonomous_silence_response("the lane said [静默] mid-sentence and kept talking")
    # the interactive EXACT rule is unchanged by the trailing-sentence law
    assert not is_intentional_silence_response("All clear today. [SILENT]")


def test_autonomous_lane_agrees_with_interactive_lane_on_cjk_punctuation_variants():
    """A Chinese lane emits fullwidth brackets or a trailing ``。``; cron/webhook must suppress
    exactly what the interactive predicate suppresses, or the two lanes drift on the new tokens."""
    for variant in ("【静默】", "静默。", "【沉默】", "沉默。", "**[静默]**", "NO_REPLY."):
        assert is_intentional_silence_response(variant)
        assert is_autonomous_silence_response(variant) == is_intentional_silence_response(variant), variant
