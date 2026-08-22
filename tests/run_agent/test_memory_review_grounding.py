"""Regression tests for the evidence-grounding rule in memory review prompts.

The background memory review historically saved INFERRED user attributes
(personality, skill level, generalized preferences) that were never stated in
the conversation — the over-inference failure mode measured in the Aug 2026
A/B validation (87.5% inferred/fabricated entries under the un-grounded
prompt vs 30% with the evidence rule; see PR body for methodology).

These tests pin the invariant that both memory-bearing review prompts carry
the grounding rule, so a future prompt edit can't silently drop it.
"""

from agent.background_review import (
    _COMBINED_REVIEW_PROMPT,
    _MEMORY_EVIDENCE_RULE,
    _MEMORY_REVIEW_PROMPT,
    _SKILL_REVIEW_PROMPT,
)


def test_memory_prompt_carries_evidence_rule():
    assert _MEMORY_EVIDENCE_RULE in _MEMORY_REVIEW_PROMPT


def test_combined_prompt_carries_evidence_rule():
    assert _MEMORY_EVIDENCE_RULE in _COMBINED_REVIEW_PROMPT


def test_skill_prompt_unchanged_by_evidence_rule():
    # The rule targets memory writes; the skill-only prompt should not carry it.
    assert _MEMORY_EVIDENCE_RULE not in _SKILL_REVIEW_PROMPT


def test_evidence_rule_core_clauses_present():
    # Behavior contract: the rule must forbid inference from indirect signals
    # and one-off-to-standing-preference generalization, and require a
    # supporting user message per entry.
    for clause in (
        "explicitly stated",
        "Do NOT infer",
        "one-time request into a standing preference",
        "specific user message",
    ):
        assert clause in _MEMORY_EVIDENCE_RULE


def test_memory_prompt_still_ends_with_nothing_to_save_escape():
    # The 'Nothing to save.' escape hatch must survive the prompt edit —
    # it's what keeps quiet sessions from generating junk writes.
    assert "Nothing to save." in _MEMORY_REVIEW_PROMPT
    assert _MEMORY_REVIEW_PROMPT.rstrip().endswith("and stop.")
