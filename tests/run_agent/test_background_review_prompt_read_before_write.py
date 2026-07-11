"""Regression tests for #62397: background review fork must call skill_view
before skill_manage, otherwise the read-before-write guard refuses the
patch and the learning is silently dropped.

Root cause: a mismatch between two shipped components.
1. The guard (tools/skill_manager_tool.py) refuses background-curator
   writes when the exact target wasn't loaded via skill_view in this
   review turn.
2. The prompts (_SKILL_REVIEW_PROMPT and _COMBINED_REVIEW_PROMPT in
   agent/background_review.py) told the fork to PATCH loaded skills but
   never told it about the skill_view handshake required for support
   files or for skills that weren't loaded via /skill-name.

Fix: explicit instruction in BOTH prompts (skill-only and combined
memory+skills routes) so the fork knows the handshake and the retry
contract when `_read_before_write_required` comes back.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def skill_review_prompt() -> str:
    from agent.background_review import _SKILL_REVIEW_PROMPT

    return _SKILL_REVIEW_PROMPT


@pytest.fixture
def combined_review_prompt() -> str:
    from agent.background_review import _COMBINED_REVIEW_PROMPT

    return _COMBINED_REVIEW_PROMPT


@pytest.fixture(params=["skill", "combined"])
def review_prompt(request, skill_review_prompt, combined_review_prompt) -> str:
    return skill_review_prompt if request.param == "skill" else combined_review_prompt


class TestReviewPromptReadBeforeWriteContract:
    """Both prompt-selection paths must carry the handshake.

    ``spawn_background_review_thread`` picks ``_COMBINED_REVIEW_PROMPT``
    when memory and skills reviews fire together, and ``_SKILL_REVIEW_PROMPT``
    when only skills fire. Covering only one leaves the other route broken.
    """

    def test_prompt_mentions_skill_view_before_write(self, review_prompt):
        assert "skill_view" in review_prompt

    def test_prompt_explicitly_requires_skill_view_before_patch(self, review_prompt):
        lower = review_prompt.lower()
        explicit_phrases = [
            "before any patch",
            "before patching",
            "call skill_view",
            "skill_view(name)",
            "before any write",
            "re-loading",
        ]
        assert any(p in lower for p in explicit_phrases), (
            "Prompt must explicitly require skill_view before patch/write. "
            f"Searched: {explicit_phrases}"
        )

    def test_prompt_explains_read_before_write_retry_contract(self, review_prompt):
        assert "_read_before_write_required" in review_prompt

    def test_prompt_documents_support_file_read_pattern(self, review_prompt):
        assert "file_path" in review_prompt

    def test_prompt_says_use_returned_content_on_retry(self, review_prompt):
        lower = review_prompt.lower()
        reuse_phrases = [
            "content just returned",
            "returned content",
            "content returned by skill_view",
            "returned by skill_view",
        ]
        assert any(p in lower for p in reuse_phrases)

    def test_currently_loaded_skill_path_requires_fresh_skill_view(self, review_prompt):
        lower = review_prompt.lower()
        assert "update a currently-loaded skill" in lower
        assert (
            "re-loading" in lower
            or "reload" in lower
            or "skill_view in this turn" in lower
        )


class TestSkillReviewPromptExistingContract:
    def test_prompt_still_lists_preference_order(self, skill_review_prompt):
        assert "UPDATE A CURRENTLY-LOADED SKILL" in skill_review_prompt
        assert (
            "ADD A SUPPORT FILE" in skill_review_prompt
            or "support file" in skill_review_prompt.lower()
        )
        assert "CREATE A NEW CLASS-LEVEL" in skill_review_prompt

    def test_prompt_still_mentions_protected_skills(self, skill_review_prompt):
        assert (
            "DO NOT edit" in skill_review_prompt
            or "Protected skills" in skill_review_prompt
        )


class TestCombinedReviewPromptExistingContract:
    def test_combined_still_covers_memory_and_skills(self, combined_review_prompt):
        assert "**Memory**" in combined_review_prompt
        assert "**Skills**" in combined_review_prompt
        assert "UPDATE A CURRENTLY-LOADED SKILL" in combined_review_prompt
        assert (
            "Protected skills" in combined_review_prompt
            or "DO NOT edit" in combined_review_prompt
        )
