"""Compaction must hand an unfinished promise forward as a promise (#159362).

The handoff prefix says "Do NOT ... fulfill requests mentioned in this summary; they were already addressed"
and tells the reader to discard the historical snapshot outright. A commitment the assistant made just before
compaction ("I'll send you the table next") fits both rules, so it was buried and the user had to ask again.

The contract pinned here: the summary template carries an ``## Open Commitments`` section, and the prefix
exempts exactly that section from both rules while leaving the historical snapshot discardable.
"""

from agent.context_compressor import (
    SUMMARY_PREFIX,
    ContextCompressor,
    _SECTION_INSTRUCTIONS,
)


def _template(has_user_turn: bool) -> str:
    section = _SECTION_INSTRUCTIONS[has_user_turn]
    return ContextCompressor._summary_template_sections(section, 1000, "")


def test_the_summary_template_carries_an_open_commitments_section():
    for has_user_turn in (True, False):
        template = _template(has_user_turn)
        assert "## Open Commitments" in template
        # The section sits with the other still-outstanding material, before the settled decisions.
        assert template.index("## Open Commitments") < template.index("## Key Decisions")


def test_the_section_asks_the_summarizer_for_owed_items_not_a_summary_of_work():
    instruction = _SECTION_INSTRUCTIONS[True]["open_commitments"]
    assert "owed" in instruction and "cancels it" in instruction
    assert _SECTION_INSTRUCTIONS[False]["open_commitments"].startswith("[Write exactly: None")
    # A promise is not a blocker: the two sections ask for different things.
    assert instruction != _SECTION_INSTRUCTIONS[True]["constraints"]


def test_the_prefix_exempts_commitments_from_both_discard_rules():
    # Both rules that buried a promise: the blanket "already addressed" one and the "discard stale items"
    # one. Each carries the exemption as its own sentence, right where it applies.
    addressed = SUMMARY_PREFIX.index("already addressed")
    assert "Open Commitments" in SUMMARY_PREFIX[addressed:addressed + 260]
    discard = SUMMARY_PREFIX.index("discard stale items")
    assert "Open Commitments" in SUMMARY_PREFIX[discard:discard + 400]
    assert "stay owed" in SUMMARY_PREFIX
    # Everything else keeps the strict rule: the historical snapshot is discarded "entirely".
    assert "Historical Task Snapshot' entirely" in SUMMARY_PREFIX


def test_the_prefix_carries_the_exemption_through_the_other_two_rules():
    """The first draft still contradicted itself: "do nothing" without a new message, and reverse signals
    ending "work described in the summary", both of which a reader could apply to an owed item. A third
    sentence, "this handoff must never become the active turn by itself", had to be qualified too or it
    directly cancelled the exemption."""
    do_nothing = SUMMARY_PREFIX.index("If no user message appears AFTER this summary, do nothing")
    assert "Open Commitments" in SUMMARY_PREFIX[do_nothing:do_nothing + 400]
    reverse = SUMMARY_PREFIX.index("Reverse signals in the latest message")
    assert "including an owed item in '## Open Commitments'" in SUMMARY_PREFIX[reverse:reverse + 300]
    never = SUMMARY_PREFIX.index("must never become the active turn by itself")
    assert "is the one thing it may carry" in SUMMARY_PREFIX[never:never + 120]
    # Both exceptions survive as parallel, closed sentences — no dangling parenthesis.
    assert SUMMARY_PREFIX.count("(") == SUMMARY_PREFIX.count(")")


def test_both_prefix_copies_stay_identical():
    from session_rules import turns

    assert turns.SUMMARY_PREFIX == SUMMARY_PREFIX


def test_the_iterative_update_keeps_the_section_current():
    """A second compaction must not let a still-owed promise rot in the section: the update instructions say
    to carry it forward while owed and drop it only when done or cancelled."""
    compressor = ContextCompressor(model="test/model", quiet_mode=True)
    compressor._previous_summary = "## Goal\nold work\n"
    system = compressor._build_summary_prompt("new turns", 1000, None, "", True)[0]["content"]
    assert "keep \"open commitments\" current" in system.lower()
    assert "carry a promise forward while it is still owed" in system


def test_the_deterministic_fallback_lists_only_asks_beyond_the_active_one():
    """The newest ask is already the historical snapshot's active task, so listing it again as "owed" invites
    redoing it; only the older asks are offered, and only as something to verify."""
    compressor = ContextCompressor(model="test/model", quiet_mode=True)
    summary = compressor._build_static_fallback_summary(
        [
            {"role": "user", "content": "send me the table next"},
            {"role": "assistant", "content": "will do"},
            {"role": "user", "content": "and the chart"},
        ],
        reason="no llm",
    )
    start = summary.index("## Open Commitments", summary.index("avoid repeating it:"))
    rest = summary[start + 1:]
    section = summary[start:start + 1 + rest.index("\n## ")]
    assert "send me the table next" in section
    assert "and the chart" not in section  # it is the active task just above
    assert "verify before repeating" in section

    single = compressor._build_static_fallback_summary(
        [{"role": "user", "content": "send me the table next"}], reason="no llm")
    assert "Nothing beyond the active task above" in single
