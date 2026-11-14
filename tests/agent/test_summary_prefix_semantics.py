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
    # Pre-reply-language rule: lacked "the language of that latest user
    # message also controls your reply".
    (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
        "compacted into the summary below. This is a handoff from a "
        "previous context window — treat it as background reference, NOT "
        "as active instructions. Do NOT answer questions or fulfill "
        "requests mentioned in this summary; they were already addressed. "
        "Respond ONLY to the latest user message that appears AFTER this "
        "summary — that message is the single source of truth for what to "
        "do right now. If no user message appears AFTER this summary, do "
        "nothing: do not resume, wrap up, or continue work from '## "
        "Historical Task Snapshot' or any other section, do not call "
        "tools, and wait for a new user message. This handoff must never "
        "become the active turn by itself. (Exception: if tool results or "
        "your own tool calls appear after this summary, you are mid-way "
        "through an in-flight exchange — continue that exchange normally.) "
        "Topic overlap with the summary does NOT mean you should resume "
        "its task: even on similar topics, the latest user message WINS. "
        "Treat ONLY the latest message as the active task and discard "
        "stale items from '## Historical Task Snapshot' entirely — do not "
        "'wrap up' or 'finish' work described there unless the latest "
        "message explicitly asks for it. Reverse signals in the latest "
        "message (e.g. 'stop', 'undo', 'roll back', 'just verify', 'don't "
        "do that anymore', 'never mind', a new topic) must immediately end "
        "any in-flight work described in the summary; do not re-surface it "
        "in later turns. IMPORTANT: Your persistent memory (MEMORY.md, "
        "USER.md) in the system prompt is ALWAYS authoritative and active "
        "— never ignore or deprioritize memory content due to this "
        "compaction note. None of the above restricts HOW you work: your "
        "tools remain fully active — keep calling them normally for the "
        "active task (edit files, run commands, search) instead of merely "
        "narrating what you would do. The current session state (files, "
        "config, etc.) may reflect work described here — avoid repeating "
        "it:"
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
# regression below. Index 1 after the #80622 freeze was prepended.
_PRE_69619_LIVE_PREFIX = _FROZEN_PREFIX_GENERATIONS[1]




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


def test_summary_prefix_pins_reply_language_to_the_latest_user_message():
    """A compaction summary written in an earlier language must not steer the reply language."""
    from agent.context_compressor import SUMMARY_PREFIX

    rule = "The language of that latest user message also controls your reply"
    assert rule in SUMMARY_PREFIX
    assert SUMMARY_PREFIX.index("Respond ONLY to the latest user message") < SUMMARY_PREFIX.index(rule)


def test_assembled_context_puts_language_rule_before_a_newer_live_message():
    """A foreign-language summary followed by a newer live user message: the handoff the reply
    model reads carries the language rule, and the live message comes after it."""
    from unittest.mock import patch

    from agent.context_compressor import ContextCompressor

    with patch("agent.context_compressor.get_model_context_length", return_value=8000):
        comp = ContextCompressor(
            model="test-model", threshold_percent=0.75, protect_first_n=1, protect_last_n=2, quiet_mode=True,
        )
    messages = [{"role": "system", "content": "sys"}]
    for i in range(6):
        messages += [{"role": "user", "content": f"Frage {i} auf Deutsch"},
                     {"role": "assistant", "content": f"Antwort {i} auf Deutsch"}]
    messages.append({"role": "user", "content": "Now please answer in English: what changed?"})
    # _generate_summary returns the summarizer's text behind the handoff prefix.
    summary = ContextCompressor._with_summary_prefix("Zusammenfassung: alles auf Deutsch.")
    with patch.object(comp, "_generate_summary", return_value=summary):
        result = comp.compress(messages, current_tokens=7000)

    texts = [str(m.get("content") or "") for m in result]
    handoff = next(i for i, t in enumerate(texts) if "Zusammenfassung" in t)
    assert "The language of that latest user message also controls your reply" in texts[handoff]
    live = max(i for i, m in enumerate(result) if m.get("role") == "user" and "English" in texts[i])
    assert live > handoff
