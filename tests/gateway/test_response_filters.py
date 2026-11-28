from gateway.response_filters import (
    is_autonomous_silence_response,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
    is_partial_silence_marker,
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


def test_code_spanned_sentinel_is_silence_in_both_lanes():
    """#140444: told the response must be exactly ``NO_REPLY``, a model renders it as a markdown
    code span. U+0060 is category ``Sk``, not ``P``, so it survived edge stripping and the bare
    control token was delivered to the chat. Doubled spans and fenced blocks collapse through the
    same run-consuming strip; the streaming hold-back canonicalizes through the same helper."""
    for variant in (
        "`NO_REPLY`",
        "`[SILENT]`",
        "``[SILENT]``",
        "```\nNO_REPLY\n```",
        "`静默`",
        "`no reply`",
    ):
        assert is_intentional_silence_response(variant), variant
        assert is_autonomous_silence_response(variant), variant
    assert is_partial_silence_marker("`NO")


def test_code_span_strip_keeps_prose_and_bracket_structure():
    # A fence is edge decoration only: naming the token mid-sentence is still prose, and a
    # malformed ``[SILENT`` cannot collapse into ``SILENT`` just because it was code-spanned.
    prose = "the lane wrote `NO_REPLY` and kept talking"
    assert not is_intentional_silence_response(prose)
    assert not is_autonomous_silence_response(prose)
    assert not is_intentional_silence_response("`[SILENT`")


def test_autonomous_lane_agrees_with_interactive_lane_on_cjk_punctuation_variants():
    """A Chinese lane emits fullwidth brackets or a trailing ``。``; cron/webhook must suppress
    exactly what the interactive predicate suppresses, or the two lanes drift on the new tokens."""
    for variant in ("【静默】", "静默。", "【沉默】", "沉默。", "**[静默]**", "NO_REPLY."):
        assert is_intentional_silence_response(variant)
        assert is_autonomous_silence_response(variant) == is_intentional_silence_response(variant), variant


def test_invisible_format_character_in_the_sentinel_does_not_reach_the_chat():
    """Regression (2026-10-09, job 8bf097dcd9d2): a cron lane with nothing to report answered
    ``[\\u200bSILENT]`` — U+200B (category Cf) before the token, hex
    5be2808b53494c454e545d. ``str.split()`` breaks on whitespace only and ``"\\u200b".isspace()``
    is False, so the character survived canonicalization, the marker stopped matching, and the raw
    control token was delivered to Telegram/Desktop as content. Both lanes share the canonicalizer."""
    assert is_autonomous_silence_response("[\u200bSILENT]")
    assert is_intentional_silence_response("[\u200bSILENT]")
    assert is_intentional_silence_response("\ufeff[SILENT]")  # BOM before the sentinel
    assert is_intentional_silence_response("[SIL\u200bENT]")  # zero-width inside the word
    assert is_intentional_silence_response("[\u200b静默\u200b]。")
    assert is_autonomous_silence_response("\u200b[SILENT] nothing new this tick")
    assert is_autonomous_silence_response("[SILENT]\u2060\n\nNothing new this tick.")
    assert is_partial_silence_marker("[\u200bSIL")  # the hold-back path streams through the same form


def test_a_response_of_only_format_characters_is_not_silence():
    """Stripping must not turn the empty-response path into an intentional silence (§ blank is not
    silence), and must not swallow prose that merely mentions a marker."""
    assert not is_intentional_silence_response("\u200b\ufeff")
    assert not is_autonomous_silence_response("\u200b\ufeff")
    assert not is_intentional_silence_response("status: NO\u200b_REPLY means the lane is quiet")
    assert not is_autonomous_silence_response("the lane said \u200bSILENT mid-sentence and kept talking")


def test_drifted_sentinel_label_suppresses_delivery():
    """Regression (2026-10-09, job 8bf097dcd9d2): the cron lane did not translate the sentinel, it
    drifted to the label ``[审核]`` ("reviewed/audited") as its whole response, and the token landed
    in the chat. A lone bracketed label is the sentinel's shape, so it takes the same whole-response
    / own-line rule as the other markers — prose that mentions it is still delivered."""
    assert is_intentional_silence_response("[审核]")
    assert is_autonomous_silence_response("[审核]")
    assert is_autonomous_silence_response("[审核]\n\nNothing new this tick.")
    assert is_autonomous_silence_response("2 deals filtered\n\n[审核]")
    assert not is_intentional_silence_response("audit status: 审核 means the mailbox was reviewed")
    assert not is_autonomous_silence_response("the lane wrote 审核 mid-sentence and kept talking")


def test_padded_sentinel_brackets_suppress_delivery():
    """Regression (2026-10-10, job 8bf097dcd9d2): with nothing to report the lane answered
    ``[ SILENT ]`` (hex 5b2053494c454e54205d) and the token reached Telegram/Desktop as content,
    while the exact ``[SILENT]`` of the 21:02 / 21:33 runs was suppressed. Collapsing whitespace
    runs keeps the single space a model writes between the bracket and the token, so the three
    padded spellings canonicalized to themselves and matched no marker; padding inside the
    brackets is a rendering artifact of the token, not prose, and must be dropped in
    canonicalization (which the prefix rule now shares, so a padded marker plus a note goes too)."""
    for variant in ("[ SILENT ]", "[SILENT ]", "[ SILENT]"):
        assert is_intentional_silence_response(variant), variant
        assert is_autonomous_silence_response(variant), variant
        assert is_autonomous_silence_response(f"{variant}\n\nNothing new this tick."), variant
        assert is_autonomous_silence_response(f"2 deals filtered\n\n{variant}"), variant
    assert is_intentional_silence_response("[ 审核 ]")  # the drifted label takes the same padding
    assert is_intentional_silence_response("[ 沉默 ]")  # and the translated sentinel in a CJK lane
    assert is_autonomous_silence_response("[\tSILENT\t]")
    assert is_partial_silence_marker("[ S")  # the hold-back path canonicalizes through the same form


def test_padding_on_bracketed_prose_is_still_delivered():
    """Only whitespace hugging the inside of a *marker's* brackets is padding; a bracketed word
    that is not a marker, prose built around one, and malformed half-brackets remain content
    (the brackets stay structural, so ``[ SILENT`` can never collapse into the ``SILENT`` marker)."""
    assert not is_intentional_silence_response("the tag [ SILENT ] means quiet")
    assert not is_autonomous_silence_response("the tag [ SILENT ] means quiet")
    assert not is_intentional_silence_response("[ no changes ] is what the watcher printed")
    assert not is_autonomous_silence_response("[ no changes ] is what the watcher printed")
    assert not is_intentional_silence_response("[ SILENT")  # half a marker is not a marker in chat
    assert not is_intentional_silence_response("SILENT ]")
    assert not is_autonomous_silence_response("a status update [ SILENT ] for this tick")


def test_code_formatted_sentinel_is_autonomous_silence():
    """Models wrap the literal marker in code formatting; the wrapped form must not leak."""
    for response in (
        "`[SILENT]`",
        "``[SILENT]``",
        "```\n[SILENT]\n```",
        "```text\n[SILENT]\n```",
        "```text block\n[SILENT]\n```",
        "`[SILENT]`\n\nNothing new this tick.",
        "2 deals filtered\n\n`[SILENT]`",
        "`[SILENT]` No changes detected",
        "``[SILENT]`` No changes detected",
        "`NO_REPLY`",
    ):
        assert is_autonomous_silence_response(response), response


def test_code_spanned_sentinel_mentioned_mid_body_is_delivered():
    assert not is_autonomous_silence_response("Reply with `[SILENT]` when there is nothing new.")
    assert not is_autonomous_silence_response("Report\n\nThe job emits `[SILENT]` on quiet ticks.\n\nDone.")
    assert not is_autonomous_silence_response("```python\nprint('[SILENT] handling')\n```")


def test_interactive_exact_rule_is_unchanged_for_code_spans():
    """Only the loose autonomous lane peels code formatting."""
    assert not is_intentional_silence_response("`[SILENT]`")


# Whole replies that are a damaged ``[SILENT]``: the six real leaks from a 265-reply cron replay
# (PR #158289 thread) plus the shapes the salvaged PRs reported. Each one reached the user's chat.
_DAMAGED_SILENT_REPLIES = (
    "[", "[\n", "[S", "[SIL", "[SILEN",  # provider cut the reply off mid-marker
    "[SILENT", "[silent", "[SILENT\n",  # dropped closing bracket (#85214, #51445)
    "[SLENT]", "[SLIENT]", "[SILNET]", "[SILENTT]", "[SIILENT]", "[SILEMT]", "[SLENT", "[ SLIENT ]",
    "[ SILENT ]", "[SILENT ]", "[ SILENT]", "[  SILENT\t]", "[\n\nSILENT]", "[\nSILENT\n]",
    "  [SILENT]  \n",
    "[\u200bSILENT]", "\ufeff[SLENT]", "[SIL\u200bENT]",
    "[ＳＩＬＥＮＴ]", "［SILENT］",  # fullwidth, folded by NFKC
    "[S\u0130LENT]",  # dotted capital I
    "`[SLIENT]`", "```\n[SLENT]\n```", "**[ SILENT ]**", "[SILENT].",
)

# Real content that merely looks bracketed or mentions the marker: always delivered.
_MUST_DELIVER_REPLIES = (
    "[URGENT]", "[ALERT]", "[SILVER] price moved", "[SLENT] extra words", "[SILVER]", "[SILENCE]",
    "[SILENTLY]", "[LENT]", "[SIGNAL]", "[TALENT]", "[CLIENT]", "[SILENT-mode] rollout finished",
    "[SILENT No changes detected", "[SLIENT] 2 deals moved", "[ SILVER", "[ok]", "[]", "[ ]",
    "S", "SLENT", "Silent retry succeeded", "Nothing new [SLENT] but here is the digest",
    "Reply with `[SILENT]` when there is nothing new.", "Price up 3%\n\n[SILVER]",
    "Built a helper where --silent-empty emits [SILENT] when empty.",
    "[\n\nSILVER moved 3%]", "\u200b\ufeff", "\u200b\n\u200b", "",
    "```\n[SILENT] handling is enabled\n```", "```\n[SLENT] handling is enabled\n```",
)


def test_damaged_silent_marker_corpus_never_reaches_the_chat_while_real_text_does():
    """A whole autonomous reply within one letter edit of ``[SILENT]`` (or cut off mid-marker) is
    silence; any reply with a second word, or a bracketed word two edits away, is delivered."""
    leaked = [r for r in _DAMAGED_SILENT_REPLIES if not is_autonomous_silence_response(r)]
    swallowed = [r for r in _MUST_DELIVER_REPLIES if is_autonomous_silence_response(r)]
    assert (leaked, swallowed) == ([], [])
