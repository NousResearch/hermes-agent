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
    HISTORICAL_IN_PROGRESS_HEADING,
    HISTORICAL_PENDING_ASKS_HEADING,
    HISTORICAL_REMAINING_WORK_HEADING,
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
    assert HISTORICAL_PENDING_ASKS_HEADING.lower() in lower
    assert HISTORICAL_REMAINING_WORK_HEADING.lower() in lower
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
    assert HISTORICAL_IN_PROGRESS_HEADING.lower() in lower


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
