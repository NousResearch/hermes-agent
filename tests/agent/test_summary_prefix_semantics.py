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
    # Pre-#80622: tools-active + topic-overlap discard, but no
    # "if no user message appears AFTER this summary, do nothing" clause.
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
        "and discard stale items from '## Historical Task Snapshot' "
        "entirely — do not 'wrap up' or 'finish' work described there "
        "unless the latest message explicitly asks for it. Reverse "
        "signals in the latest message (e.g. 'stop', 'undo', 'roll "
        "back', 'just verify', 'don't do that anymore', 'never mind', a "
        "new topic) must immediately end any in-flight work described in "
        "the summary; do not re-surface it in later turns. IMPORTANT: "
        "Your persistent memory (MEMORY.md, USER.md) in the system "
        "prompt is ALWAYS authoritative and active — never ignore or "
        "deprioritize memory content due to this compaction note. None "
        "of the above restricts HOW you work: your tools remain fully "
        "active — keep calling them normally for the active task (edit "
        "files, run commands, search) instead of merely narrating what "
        "you would do. The current session state (files, config, etc.) "
        "may reflect work described here — avoid repeating it:"
    ),
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
# regression below. Index 1 after the #80622 freeze was prepended.
_PRE_69619_LIVE_PREFIX = _FROZEN_PREFIX_GENERATIONS[1]


def test_no_user_after_handoff_must_not_act():
    """#80622: a reference-only handoff with nothing after it must not
    resume historical work or call tools."""
    lower = SUMMARY_PREFIX.lower()
    assert "if no user message appears after this summary" in lower
    assert "do nothing" in lower
    assert "wait for a new user message" in lower
    assert "must never become the active turn" in lower


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


def test_frozen_prefix_generations_match_historical_tuple():
    """Every retired generation must stay byte-identical in
    _HISTORICAL_SUMMARY_PREFIXES (newest-first)."""
    from agent.context_compressor import _HISTORICAL_SUMMARY_PREFIXES

    assert tuple(_HISTORICAL_SUMMARY_PREFIXES[: len(_FROZEN_PREFIX_GENERATIONS)]) == (
        _FROZEN_PREFIX_GENERATIONS
    )


