"""Regression tests for background review prompt leakage (#32858).

The bug: when background-review prompts are passed as user_message,
the LLM misinterprets the system-generated operational guidelines
(like "Be ACTIVE") as user preferences and writes them to USER.md.
Honcho then ingests these false entries as permanent user traits.
"""

from __future__ import annotations

import re
import sys
from unittest.mock import patch

import pytest

# Ensure the hermes-agent package is importable from the worktree
sys.path.insert(0, "")

from agent import background_review


# ---------------------------------------------------------------------------
# Layer 1 + 2: Prompt framing + self-referential content
# ---------------------------------------------------------------------------


class TestReviewPromptsHaveSystemDisclaimer:
    """All three review prompts must include a system disclaimer that
    prevents the LLM from treating the prompt's own guidelines as user
    input."""

    def test_memory_review_prompt_has_disclaimer(self):
        prompt = background_review._MEMORY_REVIEW_PROMPT
        assert _contains_disclaimer(prompt), (
            "_MEMORY_REVIEW_PROMPT is missing the system disclaimer. "
            "Without it, the review agent treats 'focus on expectations "
            "about how you should behave' as user input."
        )

    def test_skill_review_prompt_has_disclaimer(self):
        prompt = background_review._SKILL_REVIEW_PROMPT
        assert _contains_disclaimer(prompt), (
            "_SKILL_REVIEW_PROMPT is missing the system disclaimer. "
            "Without it, the review agent treats 'Be ACTIVE' and the "
            "preference order as user directives."
        )

    def test_combined_review_prompt_has_disclaimer(self):
        prompt = background_review._COMBINED_REVIEW_PROMPT
        assert _contains_disclaimer(prompt), (
            "_COMBINED_REVIEW_PROMPT is missing the system disclaimer. "
            "Without it, the review agent treats the combined memory+skill "
            "guidelines as user input."
        )

    def test_disclaimer_appears_first_in_each_prompt(self):
        """The disclaimer must be the FIRST thing the LLM sees — not buried."""
        for name, prompt in [
            ("_MEMORY_REVIEW_PROMPT", background_review._MEMORY_REVIEW_PROMPT),
            ("_SKILL_REVIEW_PROMPT", background_review._SKILL_REVIEW_PROMPT),
            ("_COMBINED_REVIEW_PROMPT", background_review._COMBINED_REVIEW_PROMPT),
        ]:
            stripped = prompt.strip()
            assert _contains_disclaimer_at_start(stripped), (
                f"{name}: system disclaimer must appear at the start "
                f"of the prompt so the LLM sees it first."
            )


# ---------------------------------------------------------------------------
# Layer 5: Competitor PR gap — ensure all three prompts are fixed
# ---------------------------------------------------------------------------


class TestAllPromptsFixed:
    """Competitor PR #32862 only fixed _SKILL_REVIEW_PROMPT and
    _COMBINED_REVIEW_PROMPT but missed _MEMORY_REVIEW_PROMPT.
    This test ensures full coverage."""

    def test_prompts_are_distinct_and_all_fixed(self):
        prompts = {
            "_MEMORY_REVIEW_PROMPT": background_review._MEMORY_REVIEW_PROMPT,
            "_SKILL_REVIEW_PROMPT": background_review._SKILL_REVIEW_PROMPT,
            "_COMBINED_REVIEW_PROMPT": background_review._COMBINED_REVIEW_PROMPT,
        }
        # All prompts must have the disclaimer
        for name, prompt in prompts.items():
            assert _contains_disclaimer(prompt), (
                f"{name}: competitor PR #32862 missed this prompt."
            )
        # Prompts must be distinct (not collapsed into one)
        assert len(set(prompts.values())) == 3, (
            "All three prompts must remain distinct — "
            "they serve different review modes."
        )


# ---------------------------------------------------------------------------
# Layer 3: Memory-write binding check (the review fork inherits parent store)
# ---------------------------------------------------------------------------


class TestSpawnBackgroundReviewThread:
    """The review thread correctly selects prompts with disclaimers."""

    def test_memory_only_returns_memory_prompt(self):
        """When review_memory=True, review_skills=False: _MEMORY_REVIEW_PROMPT."""
        from run_agent import AIAgent

        agent = object.__new__(AIAgent)
        agent.model = "fake-model"
        agent.platform = "telegram"
        agent.provider = "openai"
        agent.base_url = ""
        agent.api_key = ""
        agent.api_mode = ""
        agent.session_id = "test-session"
        agent._parent_session_id = ""
        agent._credential_pool = None
        agent._memory_store = object()
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._cached_system_prompt = "test-cached-system-prompt"
        import datetime as _dt
        agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
        agent.background_review_callback = None
        agent.status_callback = None
        agent._safe_print = lambda *_args, **_kwargs: None

        _, prompt = background_review.spawn_background_review_thread(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
            review_skills=False,
        )

        assert _contains_disclaimer(prompt), (
            "Memory-only review prompt must include the system disclaimer. "
            "Without it, the LLM treats memory-review guidelines as user preferences."
        )

    def test_skills_only_returns_skill_prompt(self):
        """When review_memory=False, review_skills=True: _SKILL_REVIEW_PROMPT (default)."""
        from run_agent import AIAgent

        agent = object.__new__(AIAgent)
        agent.model = "fake-model"
        agent.platform = "telegram"
        agent.provider = "openai"
        agent.base_url = ""
        agent.api_key = ""
        agent.api_mode = ""
        agent.session_id = "test-session"
        agent._parent_session_id = ""
        agent._credential_pool = None
        agent._memory_store = object()
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._cached_system_prompt = "test-cached-system-prompt"
        import datetime as _dt
        agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
        agent.background_review_callback = None
        agent.status_callback = None
        agent._safe_print = lambda *_args, **_kwargs: None

        _, prompt = background_review.spawn_background_review_thread(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=False,
            review_skills=True,
        )

        assert _contains_disclaimer(prompt), (
            "Skill-only review prompt must include the system disclaimer. "
            "Without it, the LLM treats skill-review guidelines as user preferences."
        )

    def test_combined_returns_combined_prompt(self):
        """When both are True: _COMBINED_REVIEW_PROMPT."""
        from run_agent import AIAgent

        agent = object.__new__(AIAgent)
        agent.model = "fake-model"
        agent.platform = "telegram"
        agent.provider = "openai"
        agent.base_url = ""
        agent.api_key = ""
        agent.api_mode = ""
        agent.session_id = "test-session"
        agent._parent_session_id = ""
        agent._credential_pool = None
        agent._memory_store = object()
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._cached_system_prompt = "test-cached-system-prompt"
        import datetime as _dt
        agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
        agent.background_review_callback = None
        agent.status_callback = None
        agent._safe_print = lambda *_args, **_kwargs: None

        _, prompt = background_review.spawn_background_review_thread(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
            review_skills=True,
        )

        assert _contains_disclaimer(prompt), (
            "Combined review prompt must include the system disclaimer. "
            "Without it, the LLM treats combined-review guidelines as user preferences."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contains_disclaimer(text: str) -> bool:
    """Check that the text contains a system-disclaimer block."""
    # Must contain language indicating this is system-generated, not user input
    has_system_note = (
        "System Note" in text
        or "system instruction" in text.lower()
        or "generated automatically" in text.lower()
        or "NOT by the user" in text
    )
    # Must instruct the LLM not to save prompt guidelines as user preferences
    has_anti_leak = (
        "do not save" in text.lower()
        or "Do NOT save" in text
        or "do not capture" in text.lower()
        or "must not" in text.lower()
    )
    return has_system_note and has_anti_leak


def _contains_disclaimer_at_start(text: str) -> bool:
    """Check that the disclaimer appears within the first 300 chars."""
    prefix = text[:300]
    return _contains_disclaimer(prefix)
