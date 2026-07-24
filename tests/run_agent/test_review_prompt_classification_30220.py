"""Regression tests for #30220.

#30220: the background self-improvement review (``_spawn_background_review``
in run_agent.py) was misclassifying content across the three persistent
stores Hermes maintains:

  * SKILL.md   — procedural knowledge ("how to do this class of task")
  * USER.md    — user profile data (preferences, communication style)
  * MEMORY.md  — environment facts (tool quirks, project conventions)

Three concrete failure modes were called out in the issue:

  1. ``_COMBINED_REVIEW_PROMPT`` and ``_SKILL_REVIEW_PROMPT`` contained
     "Be ACTIVE — most sessions produce at least one skill update" plus
     "A pass that does nothing is a missed learning opportunity, not a
     neutral outcome", which biased the reviewer toward saving
     low-signal entries and polluting the library.
  2. ``_COMBINED_REVIEW_PROMPT`` told the reviewer that BOTH memory and
     skills "should carry user-preference lessons when relevant",
     contradicting the single-source-of-truth principle.
  3. ``_MEMORY_REVIEW_PROMPT`` lumped USER.md (preferences) and
     MEMORY.md (environment facts) into a single "memory" target with
     no guidance on which store to write to.

These tests pin the new contract:

  * Clear priority: SKILL → USER → MEMORY.
  * One fact → one store, never duplicated.
  * "Nothing to save." is a first-class, frequent outcome.
  * The memory review explicitly names both ``target='user'`` and
    ``target='memory'`` routings.

The tests are behaviour assertions, not snapshot tests — they will
survive unrelated rewording but fire immediately if the regression
patterns from #30220 creep back in.
"""

from __future__ import annotations

import re

import pytest

from agent.background_review import (
    _COMBINED_REVIEW_PROMPT,
    _MEMORY_REVIEW_PROMPT,
    _SKILL_REVIEW_PROMPT,
)


# ──────────────────────────────────────────────────────────────────────
# 1. Pre-#30220 false-positive pressure must be gone from EVERY prompt
# ──────────────────────────────────────────────────────────────────────
_FALSE_POSITIVE_PHRASES = (
    "most sessions produce at least one skill update",
    "missed learning opportunity",
    "not a neutral outcome",
    "don't reach for that conclusion as a default",
    "do not reach for that conclusion as a default",
)


class TestFalsePositivePressureRemoved:
    """The four phrases the issue cites must NOT survive in any prompt.
    Each one was an active bias toward saving low-signal entries."""

    @pytest.mark.parametrize("phrase", _FALSE_POSITIVE_PHRASES)
    def test_skill_review_prompt_does_not_contain(self, phrase: str) -> None:
        assert phrase.lower() not in _SKILL_REVIEW_PROMPT.lower(), (
            f"Pre-#30220 pressure phrase still present in _SKILL_REVIEW_PROMPT: "
            f"{phrase!r}"
        )

    @pytest.mark.parametrize("phrase", _FALSE_POSITIVE_PHRASES)
    def test_combined_review_prompt_does_not_contain(self, phrase: str) -> None:
        assert phrase.lower() not in _COMBINED_REVIEW_PROMPT.lower(), (
            f"Pre-#30220 pressure phrase still present in _COMBINED_REVIEW_PROMPT: "
            f"{phrase!r}"
        )

    @pytest.mark.parametrize("phrase", _FALSE_POSITIVE_PHRASES)
    def test_memory_review_prompt_does_not_contain(self, phrase: str) -> None:
        # The memory-only prompt never had skill pressure, but pin it
        # too so a future regression can't sneak in here either.
        assert phrase.lower() not in _MEMORY_REVIEW_PROMPT.lower(), (
            f"Pre-#30220 pressure phrase still present in _MEMORY_REVIEW_PROMPT: "
            f"{phrase!r}"
        )


# ──────────────────────────────────────────────────────────────────────
# 2. "Nothing to save." is first-class in every prompt
# ──────────────────────────────────────────────────────────────────────
class TestNothingToSaveIsFirstClass:
    """Every prompt must keep ``"Nothing to save."`` as a real option
    AND frame it neutrally — no "but don't reach for that conclusion"
    anti-default clause."""

    def test_skill_review_prompt_marks_nothing_to_save_as_valid_outcome(
        self,
    ) -> None:
        prompt = _SKILL_REVIEW_PROMPT
        assert "Nothing to save." in prompt
        assert "valid and frequent outcome" in prompt.lower()

    def test_combined_review_prompt_marks_nothing_to_save_as_valid_outcome(
        self,
    ) -> None:
        prompt = _COMBINED_REVIEW_PROMPT
        assert "Nothing to save." in prompt
        assert "valid and frequent outcome" in prompt.lower()

    def test_memory_review_prompt_marks_nothing_to_save_as_valid_outcome(
        self,
    ) -> None:
        prompt = _MEMORY_REVIEW_PROMPT
        assert "Nothing to save." in prompt
        assert "valid and frequent outcome" in prompt.lower()


# ──────────────────────────────────────────────────────────────────────
# 3. _MEMORY_REVIEW_PROMPT distinguishes USER.md vs MEMORY.md
# ──────────────────────────────────────────────────────────────────────
class TestMemoryReviewDistinguishesUserAndMemoryStores:
    """The pre-#30220 prompt mentioned only "the memory tool" with no
    indication that ``target='user'`` and ``target='memory'`` are
    different stores. The new prompt must teach the reviewer which
    target to use for each kind of fact."""

    def test_names_both_files_explicitly(self) -> None:
        assert "USER.md" in _MEMORY_REVIEW_PROMPT
        assert "MEMORY.md" in _MEMORY_REVIEW_PROMPT

    def test_specifies_target_user_for_user_profile(self) -> None:
        assert "target='user'" in _MEMORY_REVIEW_PROMPT

    def test_specifies_target_memory_for_environment_facts(self) -> None:
        assert "target='memory'" in _MEMORY_REVIEW_PROMPT

    def test_associates_user_target_with_preferences(self) -> None:
        """The USER.md target must be presented as the home of
        communication style, preferences, work style, etc."""
        prompt = _MEMORY_REVIEW_PROMPT
        user_section = prompt.split("USER.md", 1)[1].split("MEMORY.md", 1)[0]
        lower = user_section.lower()
        assert any(
            k in lower
            for k in (
                "communication style",
                "preferences",
                "work style",
                "tone",
                "verbosity",
            )
        ), "USER.md bullet must mention preference-shaped content"

    def test_associates_memory_target_with_environment(self) -> None:
        """The MEMORY.md target must be presented as the home of
        environment / tool / project facts."""
        prompt = _MEMORY_REVIEW_PROMPT
        memory_section = prompt.split("MEMORY.md", 1)[1]
        lower = memory_section.lower()
        assert any(
            k in lower
            for k in (
                "environment",
                "tool quirk",
                "project convention",
                "config",
            )
        ), "MEMORY.md bullet must mention environment-shaped content"


# ──────────────────────────────────────────────────────────────────────
# 4. _COMBINED_REVIEW_PROMPT has explicit priority + one-place rule
# ──────────────────────────────────────────────────────────────────────
class TestCombinedReviewHasPriorityAndOnePlaceRule:
    """The issue's "Expected behavior" section:
        * Clear priority: SKILL → USER → MEMORY
        * One fact → one place, never duplicated"""

    def test_lists_all_three_stores(self) -> None:
        prompt = _COMBINED_REVIEW_PROMPT
        assert "SKILL" in prompt
        assert "USER.md" in prompt
        assert "MEMORY.md" in prompt

    def test_skill_listed_before_user_and_memory(self) -> None:
        """Priority order is positional — SKILL must appear first in
        the "Three stores are available" block. Use the first
        occurrence of each name as the anchor."""
        prompt = _COMBINED_REVIEW_PROMPT
        skill_idx = prompt.index("SKILL")
        user_idx = prompt.index("USER.md")
        memory_idx = prompt.index("MEMORY.md")
        assert skill_idx < user_idx < memory_idx, (
            f"Expected SKILL < USER.md < MEMORY.md order, got "
            f"SKILL@{skill_idx} USER.md@{user_idx} MEMORY.md@{memory_idx}"
        )

    def test_priority_order_is_explicit(self) -> None:
        """The prompt must name the order, not just present items in
        sequence. The numbered list "1. **SKILL**", "2. **USER.md**",
        "3. **MEMORY.md**" is the canonical form."""
        prompt = _COMBINED_REVIEW_PROMPT
        # Use a relaxed regex so future re-numbering / re-wording survives.
        assert re.search(r"1\.\s*\*\*SKILL", prompt), (
            "SKILL must be priority 1"
        )
        assert re.search(r"2\.\s*\*\*USER\.md", prompt), (
            "USER.md must be priority 2"
        )
        assert re.search(r"3\.\s*\*\*MEMORY\.md", prompt), (
            "MEMORY.md must be priority 3"
        )

    def test_priority_word_appears(self) -> None:
        """The word "priority" (or "preference") must appear near the
        list so the reviewer reads it as ordered, not parallel."""
        prompt = _COMBINED_REVIEW_PROMPT.lower()
        assert "priority order" in prompt or "preference order" in prompt

    def test_explicit_one_fact_one_store_rule(self) -> None:
        """The pre-#30220 'Both should carry user-preference lessons
        when relevant.' clause was the root cause of duplication. The
        new prompt must explicitly forbid duplication."""
        prompt = _COMBINED_REVIEW_PROMPT
        lower = prompt.lower()
        # The exact pre-#30220 clause must be gone.
        assert "both should carry user-preference lessons" not in lower, (
            "pre-#30220 dual-write encouragement must be removed"
        )
        # The new one-place rule must be present.
        assert (
            "one fact → one store" in prompt
            or "one fact -> one store" in lower
            or "one store, never duplicated" in lower
        ), "must explicitly forbid writing the same fact to multiple stores"

    def test_user_preference_routing_is_disjoint(self) -> None:
        """User-preference lessons must land in EXACTLY ONE of
        {skill, USER.md} — never both. The pre-#30220 prompt told the
        reviewer to write to both."""
        prompt = _COMBINED_REVIEW_PROMPT.lower()
        assert "exactly one of {skill, user.md}" in prompt or (
            "exactly one" in prompt and "skill" in prompt and "user.md" in prompt
        ), "must explicitly route user-preference to exactly one of skill/USER.md"


# ──────────────────────────────────────────────────────────────────────
# 5. Issue number is cited inline so future readers find the rationale
# ──────────────────────────────────────────────────────────────────────
class TestPromptsCiteIssueNumber:
    """The behavioural changes here aren't obvious from reading the
    prompt out of context. Cite #30220 inline so a future maintainer
    grep'ing for the issue number lands directly on the relevant
    paragraphs."""

    def test_skill_review_prompt_cites_30220(self) -> None:
        assert "#30220" in _SKILL_REVIEW_PROMPT

    def test_combined_review_prompt_cites_30220(self) -> None:
        assert "#30220" in _COMBINED_REVIEW_PROMPT

    def test_memory_review_prompt_cites_30220(self) -> None:
        assert "#30220" in _MEMORY_REVIEW_PROMPT


# ──────────────────────────────────────────────────────────────────────
# 6. Module-level exports are stable so AIAgent fork still resolves them
# ──────────────────────────────────────────────────────────────────────
class TestPromptsRemainImportableAndStringTyped:
    """The forked review agent reads the prompts via
    ``getattr(agent, '_..._REVIEW_PROMPT', module_default)``. A regression
    that turns any of them into None / bytes / a callable would silently
    break the entire self-improvement loop."""

    @pytest.mark.parametrize(
        "prompt",
        [
            _SKILL_REVIEW_PROMPT,
            _MEMORY_REVIEW_PROMPT,
            _COMBINED_REVIEW_PROMPT,
        ],
        ids=["skill", "memory", "combined"],
    )
    def test_prompt_is_non_empty_string(self, prompt) -> None:
        assert isinstance(prompt, str)
        # Real content, not just punctuation / placeholders.
        assert len(prompt) > 200

    def test_aiagent_exposes_class_level_aliases(self) -> None:
        """``AIAgent`` re-exports each module constant as a class
        attribute so legacy code paths (and the test helpers in
        tests/run_agent/test_background_review*.py) can monkeypatch
        them per-instance. The aliases must continue to round-trip."""
        from run_agent import AIAgent

        assert AIAgent._SKILL_REVIEW_PROMPT == _SKILL_REVIEW_PROMPT
        assert AIAgent._MEMORY_REVIEW_PROMPT == _MEMORY_REVIEW_PROMPT
        assert AIAgent._COMBINED_REVIEW_PROMPT == _COMBINED_REVIEW_PROMPT
