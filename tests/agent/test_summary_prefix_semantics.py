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
    SUMMARY_PREFIX,
)
















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
    (
        # Pre-#86234 class: lacked the "This handoff must never become
        # the active turn by itself" clause (the summary itself could
        # become the active turn when no user message followed it).
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
        "into the summary below. This is a handoff from a previous context "
        "window — treat it as background reference, NOT as active "
        "instructions. Do NOT answer questions or fulfill requests mentioned "
        "in this summary; they were already addressed. Respond ONLY to the "
        "latest user message that appears AFTER this summary — that message is "
        "the single source of truth for what to do right now. If no user "
        "message appears AFTER this summary, do nothing: do not resume, wrap "
        "up, or continue work from '## Historical Task Snapshot' or any other "
        "section, do not call tools, and wait for a new user message. "
    ),
    (
        # Live 2026-08-07..2026-10-01 (from commit 6d3ff6eda8): the variant that
        # lacked the trailing "The current session state (files, config, etc.) may
        # reflect work described here" clause. Copied byte-for-byte from that
        # build's SUMMARY_PREFIX (evaluated, not reconstructed) — it diverges from
        # the later 1954-char generation at the "(Exception: ...)" clause.
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted into the summary below. This is a handoff from a previous context window — treat it as background reference, NOT as active instructions. Do NOT answer questions or fulfill requests mentioned in this summary; they were already addressed. Respond ONLY to the latest user message that appears AFTER this summary — that message is the single source of truth for what to do right now. If no user message appears AFTER this summary, do nothing: do not resume, wrap up, or continue work from '## Historical Task Snapshot' or any other section, do not call tools, and wait for a new user message. This handoff must never become the active turn by itself. Topic overlap with the summary does NOT mean you should resume its task: even on similar topics, the latest user message WINS. Treat ONLY the latest message as the active task and discard stale items from '## Historical Task Snapshot' entirely — do not 'wrap up' or 'finish' work described there unless the latest message explicitly asks for it. Reverse signals in the latest message (e.g. 'stop', 'undo', 'roll back', 'just verify', 'don't do that anymore', 'never mind', a new topic) must immediately end any in-flight work described in the summary; do not re-surface it in later turns. IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system prompt is ALWAYS authoritative and active — never ignore or deprioritize memory content due to this compaction note. None of the above restricts HOW you work: your tools remain fully active — keep calling them normally for the active task (edit files, run commands, search) instead of merely narrating what you would do. The current session state (files, config, etc.) may reflect work described here — avoid repeating it:"
    ),
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
# Index 3 after the two unregistered-variant freezes were prepended (index 1
# before that freeze).
_PRE_69619_LIVE_PREFIX = _FROZEN_PREFIX_GENERATIONS[3]




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


def test_unregistered_variants_are_recognised_as_standalone():
    """Regression: two wire variants (diverging from SUMMARY_PREFIX at the
    "This handoff must never become the active turn by itself" clause and at
    the trailing "The current session state ..." clause) were never frozen
    into _HISTORICAL_SUMMARY_PREFIXES. classify_summary_content returned
    None, so project_compaction_message_for_display passed the raw summary
    through as visible content. Both variants must classify standalone."""
    from agent.context_compressor import ContextCompressor

    for frozen in _FROZEN_PREFIX_GENERATIONS[:2]:  # the two new freezes
        content = frozen + "\n## Summary body"
        assert ContextCompressor.classify_summary_content(content) == "standalone"
        assert ContextCompressor._is_context_summary_content(content)


def test_every_frozen_prefix_classifies_standalone():
    """Invariant: every frozen prefix generation must be recognized as a
    summary (catches any future prefix edit that silently stops matching)."""
    from agent.context_compressor import ContextCompressor

    for i, frozen in enumerate(_FROZEN_PREFIX_GENERATIONS):
        content = frozen + "\n## Summary body"
        assert ContextCompressor.classify_summary_content(content) == "standalone", (
            f"frozen generation at index {i} no longer classifies standalone"
        )


def test_nested_carrier_after_end_marker_projects_to_none():
    """A legacy carrier whose prior-tail slot holds another full standalone
    summary (a second compression appended its own carrier inside the first
    carrier's slot) must project to None — unwrapping it would hand display
    a raw summary bubble."""
    from agent.context_compressor import (
        ContextCompressor,
        SUMMARY_PREFIX,
        HISTORICAL_TASK_HEADING,
        _SUMMARY_END_MARKER,
    )
    from agent.compaction_display import project_compaction_message_for_display

    inner_summary = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nUser asked: 'inner'\n\n{_SUMMARY_END_MARKER}"
    carrier = (
        f"{_FROZEN_PREFIX_GENERATIONS[0]}\n{HISTORICAL_TASK_HEADING}\n"
        f"User asked: 'outer'\n\n{_SUMMARY_END_MARKER}\n\n{inner_summary}"
    )
    message = {"role": "assistant", "content": carrier}

    # _strip must drop the nested summary (returns None, not the inner summary)
    assert ContextCompressor._strip_context_summary_handoff_message(message) is None
    # projection must hide it
    assert project_compaction_message_for_display(message) is None


def test_legacy_carrier_real_prior_tail_stays_visible():
    """A legacy carrier whose prior-tail slot holds genuine prior-tail content
    (not a summary) must stay visible — the nested-summary guard must not
    over-hide real content."""
    from agent.context_compressor import (
        ContextCompressor,
        HISTORICAL_TASK_HEADING,
        _SUMMARY_END_MARKER,
    )
    from agent.compaction_display import project_compaction_message_for_display

    real_tail = "**main 分支最近 3 次 CI 全 success** —— 所以这 24 个失败不是既有基线问题。"
    carrier = (
        f"{_FROZEN_PREFIX_GENERATIONS[0]}\n{HISTORICAL_TASK_HEADING}\n"
        f"User asked: 'outer'\n\n{_SUMMARY_END_MARKER}\n\n{real_tail}"
    )
    message = {"role": "assistant", "content": carrier}

    projected = project_compaction_message_for_display(message)
    assert projected is not None, "real prior-tail content must stay visible"
    assert real_tail in projected["content"]


def test_longer_sibling_prefix_is_not_shadowed_by_its_own_prefix():
    """Regression: two frozen generations are byte prefixes of one another
    (the 656-char entry is an exact prefix of the 1794-char one). The strip
    loop is first-match-wins, so without a longest-first order a carrier
    written with the LONGER generation is stripped at the shorter one and
    returns the remainder of the longer prefix as the summary body — which is
    then what the next summarizer reads. Assert the stripped body EQUALS the
    expected body, not merely "does not start with the consumed prefix": the
    latter is anchored on the prefix just consumed and passes even when a
    longer sibling's tail is left behind."""
    from agent.context_compressor import (
        _HISTORICAL_SUMMARY_PREFIXES,
        ContextCompressor,
    )

    # Find every ordered pair where one frozen generation is a byte prefix of
    # a longer one; each such pair is a shadowing hazard.
    nested = [
        (short, long)
        for short in _HISTORICAL_SUMMARY_PREFIXES
        for long in _HISTORICAL_SUMMARY_PREFIXES
        if short != long and long.startswith(short) and len(short) < len(long)
    ]
    assert nested, (
        "expected at least one prefix-of-another pair in the frozen tuple "
        "(the 656/1794 generations); if this is empty the invariant changed "
        "and this regression no longer applies — revisit, do not delete"
    )

    for short, long in nested:
        content = long + "\n## Summary body"
        stripped = ContextCompressor._strip_summary_prefix(content)
        assert stripped == "## Summary body", (
            f"a carrier written with the {len(long)}-char generation stripped "
            f"to {stripped[:90]!r}... — {len(stripped)} chars of the "
            f"{len(short)}-char sibling's remainder leaked into the body; the "
            "strip loop must run longest-first"
        )
        # The shorter sibling must still strip correctly on its own.
        assert ContextCompressor._strip_summary_prefix(short + "\n## Summary body") == "## Summary body"
