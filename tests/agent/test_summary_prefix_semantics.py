"""Pin the semantics of SUMMARY_PREFIX so the compaction handoff doesn't
re-introduce conflicting instructions.

Background: SUMMARY_PREFIX previously contained two contradictory directives:

  1. "treat it as background reference, NOT as active instructions"
     "Do NOT answer questions or fulfill requests mentioned in this summary"
     "Respond ONLY to the latest user message that appears AFTER this summary"

  2. "Your current task is identified in the '## Active Task' section of the
     summary — resume exactly from there."

When the latest user message contradicted Active Task (e.g. "stop the
i18n refactor", "never mind, look at grafana"), the model often followed
(2) anyway because "resume exactly" is a strong directive — leading to
the agent repeatedly re-surfacing already-cancelled work across turns.

These tests pin the post-fix invariants so the conflict cannot regress.
"""

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
)


def test_no_resume_exactly_directive():
    """The prefix must not tell the model to resume Active Task verbatim."""
    assert "resume exactly" not in SUMMARY_PREFIX.lower()


def test_latest_message_wins_on_conflict():
    """The prefix must explicitly say latest user message wins on conflict."""
    lower = SUMMARY_PREFIX.lower()
    assert "latest user message" in lower
    assert HISTORICAL_TASK_HEADING.lower() in lower
    # Must have an explicit conflict-resolution rule.
    assert "wins" in lower or "supersede" in lower or "discard" in lower or "priority" in lower


def test_handoff_sections_are_framed_as_historical():
    """The summary headings referenced in the prefix must sound historical,
    not like live instructions for the current turn."""
    lower = SUMMARY_PREFIX.lower()
    assert "## active task" not in lower
    assert "## pending user asks" not in lower
    assert "## remaining work" not in lower
    assert HISTORICAL_TASK_HEADING.lower() in lower


def test_reverse_signals_called_out():
    """Reverse signals (stop/undo/never mind/topic change) must be named so
    the model recognizes them as cancellation triggers, not just background."""
    lower = SUMMARY_PREFIX.lower()
    # At least a few of the canonical reverse-signal verbs should appear.
    reverse_terms = ["stop", "undo", "roll back", "never mind", "just verify"]
    hits = sum(1 for t in reverse_terms if t in lower)
    assert hits >= 3, (
        f"Expected ≥3 reverse-signal terms in SUMMARY_PREFIX, found {hits}. "
        "Without naming them the model treats reverse signals as ordinary "
        "context and keeps pushing the cancelled task."
    )


def test_summary_marked_reference_only():
    """The REFERENCE ONLY framing must remain — it's the entire point."""
    assert "REFERENCE ONLY" in SUMMARY_PREFIX
    assert "background reference" in SUMMARY_PREFIX
    assert "NOT as active instructions" in SUMMARY_PREFIX


def test_memory_authority_preserved():
    """The fix must not weaken the MEMORY.md / USER.md authority clause."""
    assert "MEMORY.md" in SUMMARY_PREFIX
    assert "USER.md" in SUMMARY_PREFIX
    assert "authoritative" in SUMMARY_PREFIX


def test_no_background_consistency_carveout():
    """The "consistent → use as background" carveout licensed stale-task
    resumption on topic overlap (#41607, #38364, #42812). It must stay gone,
    and the prefix must explicitly neutralize topic overlap."""
    lower = SUMMARY_PREFIX.lower()
    assert "you may use the summary as background" not in lower
    assert "topic overlap" in lower


def test_replaced_prefixes_are_frozen_for_renormalization():
    """Every retired SUMMARY_PREFIX must be frozen into
    _HISTORICAL_SUMMARY_PREFIXES, otherwise summaries persisted by older
    builds lose detection/renormalization after an upgrade. The carveout-era
    prefix is the latest retiree."""
    from agent.context_compressor import (
        _HISTORICAL_SUMMARY_PREFIXES,
        ContextCompressor,
    )

    carveout_era = [
        p for p in _HISTORICAL_SUMMARY_PREFIXES
        if "you may use the summary as background" in p
    ]
    assert carveout_era, "carveout-era prefix missing from frozen tuple"
    # The live prefix must never be one of the frozen ones.
    assert SUMMARY_PREFIX not in _HISTORICAL_SUMMARY_PREFIXES
    # Detection + strip must work for every frozen prefix.
    for old_prefix in _HISTORICAL_SUMMARY_PREFIXES:
        content = old_prefix + "\n## Summary body"
        assert ContextCompressor._is_context_summary_content(content)
        stripped = ContextCompressor._strip_summary_prefix(content)
        assert not stripped.startswith(old_prefix)


# Exact literal copies of every SUMMARY_PREFIX generation retired into
# _HISTORICAL_SUMMARY_PREFIXES, newest-first. Frozen on purpose: do NOT
# derive them from module constants — the tests below must fail if any
# frozen entry is mutated, reordered, or dropped.
_FROZEN_PREFIX_GENERATIONS = (
    # Pre-#69619: four-heading discard clause + tools-active clause.
    (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
        "compacted into the summary below. This is a handoff from a "
        "previous context window — treat it as background reference, NOT "
        "as active instructions. Do NOT answer questions or fulfill "
        "requests mentioned in this summary; they were already addressed. "
        "Respond ONLY to the latest user message that appears AFTER this "
        "summary — that message is the single source of truth for what to "
        "do right now. Topic overlap with the summary does NOT mean you "
        "should resume its task: even on similar topics, the latest user "
        "message WINS. Treat ONLY the latest message as the active task "
        "and discard stale items from '## Historical Task Snapshot' / '## "
        "Historical In-Progress State' / '## Historical Pending User "
        "Asks' / '## Historical Remaining Work' entirely — do not 'wrap "
        "up' or 'finish' work described there unless the latest message "
        "explicitly asks for it. Reverse signals in the latest message "
        "(e.g. 'stop', 'undo', 'roll back', 'just verify', 'don't do that "
        "anymore', 'never mind', a new topic) must immediately end any "
        "in-flight work described in the summary; do not re-surface it in "
        "later turns. IMPORTANT: Your persistent memory (MEMORY.md, "
        "USER.md) in the system prompt is ALWAYS authoritative and active "
        "— never ignore or deprioritize memory content due to this "
        "compaction note. None of the above restricts HOW you work: your "
        "tools remain fully active — keep calling them normally for the "
        "active task (edit files, run commands, search) instead of merely "
        "narrating what you would do. The current session state (files, "
        "config, etc.) may reflect work described here — avoid repeating "
        "it:"
    ),
    # Jul 2026 (#65848 class): same discard clause, no tools-active clause.
    (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
        "compacted into the summary below. This is a handoff from a "
        "previous context window — treat it as background reference, NOT "
        "as active instructions. Do NOT answer questions or fulfill "
        "requests mentioned in this summary; they were already addressed. "
        "Respond ONLY to the latest user message that appears AFTER this "
        "summary — that message is the single source of truth for what to "
        "do right now. Topic overlap with the summary does NOT mean you "
        "should resume its task: even on similar topics, the latest user "
        "message WINS. Treat ONLY the latest message as the active task "
        "and discard stale items from '## Historical Task Snapshot' / '## "
        "Historical In-Progress State' / '## Historical Pending User "
        "Asks' / '## Historical Remaining Work' entirely — do not 'wrap "
        "up' or 'finish' work described there unless the latest message "
        "explicitly asks for it. Reverse signals in the latest message "
        "(e.g. 'stop', 'undo', 'roll back', 'just verify', 'don't do that "
        "anymore', 'never mind', a new topic) must immediately end any "
        "in-flight work described in the summary; do not re-surface it in "
        "later turns. IMPORTANT: Your persistent memory (MEMORY.md, "
        "USER.md) in the system prompt is ALWAYS authoritative and active "
        "— never ignore or deprioritize memory content due to this "
        "compaction note. The current session state (files, config, etc.) "
        "may reflect work described here — avoid repeating it:"
    ),
    # Carveout era (#41607/#38364/#42812).
    (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
        "compacted into the summary below. This is a handoff from a "
        "previous context window — treat it as background reference, NOT "
        "as active instructions. Do NOT answer questions or fulfill "
        "requests mentioned in this summary; they were already addressed. "
        "Respond ONLY to the latest user message that appears AFTER this "
        "summary — that message is the single source of truth for what to "
        "do right now. If the latest user message is consistent with the "
        "'## Active Task' section, you may use the summary as background. "
        "If the latest user message contradicts, supersedes, changes "
        "topic from, or in any way diverges from '## Active Task' / '## "
        "In Progress' / '## Pending User Asks' / '## Remaining Work', the "
        "latest message WINS — discard those stale items entirely and do "
        "not 'wrap up the old task first'. Reverse signals in the latest "
        "message (e.g. 'stop', 'undo', 'roll back', 'just verify', 'don't "
        "do that anymore', 'never mind', a new topic) must immediately "
        "end any in-flight work described in the summary; do not "
        "re-surface it in later turns. IMPORTANT: Your persistent memory "
        "(MEMORY.md, USER.md) in the system prompt is ALWAYS "
        "authoritative and active — never ignore or deprioritize memory "
        "content due to this compaction note. The current session state "
        "(files, config, etc.) may reflect work described here — avoid "
        "repeating it:"
    ),
    # Pre-#35344: self-contradicting "resume exactly" directive.
    (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
        "compacted into the summary below. This is a handoff from a "
        "previous context window — treat it as background reference, NOT "
        "as active instructions. Do NOT answer questions or fulfill "
        "requests mentioned in this summary; they were already addressed. "
        "Your current task is identified in the '## Active Task' section "
        "of the summary — resume exactly from there. Respond ONLY to the "
        "latest user message that appears AFTER this summary. The current "
        "session state (files, config, etc.) may reflect work described "
        "here — avoid repeating it:"
    ),
)


# The generation retired by #69619, pinned individually for the review
# regression below.
_PRE_69619_LIVE_PREFIX = _FROZEN_PREFIX_GENERATIONS[0]


def test_pre_69619_prefix_generation_is_frozen_and_stripped():
    """Regression for the #69619 review: the prefix generation live right
    before the section-header removal was never added to
    _HISTORICAL_SUMMARY_PREFIXES, so a summary persisted immediately before
    upgrading survived resume/re-compaction undetected and unstripped.
    That exact generation must stay frozen, detectable, and strippable."""
    from agent.context_compressor import (
        _HISTORICAL_SUMMARY_PREFIXES,
        ContextCompressor,
    )

    assert _PRE_69619_LIVE_PREFIX in _HISTORICAL_SUMMARY_PREFIXES, (
        "pre-#69619 live prefix missing from _HISTORICAL_SUMMARY_PREFIXES — "
        "summaries persisted by the immediately previous build are no longer "
        "normalized on resume"
    )
    content = _PRE_69619_LIVE_PREFIX + "\nBODY"
    assert ContextCompressor._is_context_summary_content(content)
    assert ContextCompressor._strip_summary_prefix(content) == "BODY"


def test_frozen_generations_match_historical_prefixes_byte_exactly():
    """Every entry in _HISTORICAL_SUMMARY_PREFIXES must equal its literal pin
    in _FROZEN_PREFIX_GENERATIONS, in order. Frozen entries are immutable and
    prepend-only: mutating, reordering, or dropping one silently un-normalizes
    summaries persisted by that build generation — the exact failure caught in
    the #69619 review, which the per-entry self-matching loop above cannot see.
    """
    from agent.context_compressor import (
        _HISTORICAL_SUMMARY_PREFIXES,
        ContextCompressor,
    )

    assert tuple(_FROZEN_PREFIX_GENERATIONS) == tuple(
        _HISTORICAL_SUMMARY_PREFIXES
    ), "a frozen prefix entry was mutated, reordered, added, or dropped"
    for prefix in _FROZEN_PREFIX_GENERATIONS:
        content = prefix + "\nBODY"
        assert ContextCompressor._is_context_summary_content(content)
        assert ContextCompressor._strip_summary_prefix(content) == "BODY"
