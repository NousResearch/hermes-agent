"""Regression tests for #62397: background review fork must call skill_view
before skill_manage, otherwise the read-before-write guard refuses the
patch and the learning is silently dropped.

Root cause: a mismatch between two shipped components.
1. The guard (tools/skill_manager_tool.py) refuses background-curator
   writes when the exact target wasn't loaded via skill_view in this
   review turn.
2. The prompt (_SKILL_REVIEW_PROMPT in agent/background_review.py)
   told the fork to PATCH loaded skills but never told it about the
   skill_view handshake required for support files (templates/,
   scripts/, references/) or for skills that weren't loaded via
   /skill-name but were located via skills_list.

Fix: explicit instruction in the prompt so the fork
(a) knows to call skill_view before any write_file / patch / edit,
(b) knows the retry contract when _read_before_write_required comes
    back, and
(c) understands support-file writes need skill_view(name, file_path=...).

Tests below assert the prompt carries each contract clause so a future
prompt edit can't silently drop them again.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def skill_review_prompt() -> str:
    from agent.background_review import _SKILL_REVIEW_PROMPT

    return _SKILL_REVIEW_PROMPT


class TestSkillReviewPromptReadBeforeWriteContract:
    """The background-review fork must be told, in plain language, that:

    1. ANY write to a skill file (SKILL.md or a support file under
       references/ / templates/ / scripts/) requires a prior skill_view
       call in THIS review turn.
    2. If the guard returns _read_before_write_required, the fork must
       call skill_view(name) (or skill_view(name, file_path=...) for a
       support file) and retry the write with the returned content.
    3. This applies even when the skill is being CREATED — the guard
       fires for patch/edit/write_file/delete/remove_file.
    """

    def test_prompt_mentions_skill_view_before_write(self, skill_review_prompt):
        # The fork must be told to load the target via skill_view BEFORE
        # any patch / write_file / edit action.
        assert "skill_view" in skill_review_prompt, (
            "Background review prompt must reference skill_view so the "
            "fork loads targets before patching"
        )

    def test_prompt_explicitly_requires_skill_view_before_patch(self, skill_review_prompt):
        # The contract must be stated explicitly (not buried in the
        # middle of an unrelated paragraph where the LLM can miss it).
        # We check for any of the common phrasings.
        lower = skill_review_prompt.lower()
        explicit_phrases = [
            "before any patch",
            "before patching",
            "call skill_view",
            "skill_view(name)",
            "before any write",
        ]
        assert any(p in lower for p in explicit_phrases), (
            "Prompt must explicitly require skill_view before patch/write. "
            f"Searched: {explicit_phrases}"
        )

    def test_prompt_explains_read_before_write_retry_contract(self, skill_review_prompt):
        """If the guard returns _read_before_write_required, the fork must
        know to call skill_view and retry. Without this clause the fork
        silently drops the learning."""
        assert "_read_before_write_required" in skill_review_prompt, (
            "Prompt must surface the exact guard signal name so the fork "
            "knows what to do when it sees it"
        )

    def test_prompt_documents_support_file_read_pattern(self, skill_review_prompt):
        """support_file writes need skill_view(name, file_path=...) — the
        fork can't just call skill_view(name) for templates/scripts/
        references/* writes. Without this clause every support-file write
        fails the guard."""
        assert "file_path" in skill_review_prompt, (
            "Prompt must show the skill_view(name, file_path=...) form "
            "for support-file writes (references/, templates/, scripts/)"
        )

    def test_prompt_says_use_returned_content_on_retry(self, skill_review_prompt):
        """On retry, the fork must reuse the content skill_view just
        returned (not re-derive it from memory) so the guard sees a
        consistent in-memory version."""
        lower = skill_review_prompt.lower()
        reuse_phrases = [
            "content just returned",
            "returned content",
            "content returned by skill_view",
            "returned by skill_view",
        ]
        assert any(p in lower for p in reuse_phrases), (
            "Prompt must tell the fork to reuse content returned by "
            "skill_view on retry. Searched: " + str(reuse_phrases)
        )


class TestSkillReviewPromptExistingContract:
    """Belt-and-suspenders: the prompt's existing shape (target library,
    preference order, etc.) must NOT regress with our edit."""

    def test_prompt_still_lists_preference_order(self, skill_review_prompt):
        assert "UPDATE A CURRENTLY-LOADED SKILL" in skill_review_prompt
        assert "ADD A SUPPORT FILE" in skill_review_prompt or "support file" in skill_review_prompt.lower()
        assert "CREATE A NEW CLASS-LEVEL" in skill_review_prompt

    def test_prompt_still_mentions_protected_skills(self, skill_review_prompt):
        assert "DO NOT edit" in skill_review_prompt or "Protected skills" in skill_review_prompt
