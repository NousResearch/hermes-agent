"""Tests for agent.skill_muter — LLM-driven SKILL.md mutation."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agent.skill_muter import (
    DEFAULT_MUTATION_PROMPT_TEMPLATE,
    ApplyResult,
    DefaultSkillMuter,
    DefaultSkillMuterApplier,
    MutationContext,
    MutationProposal,
    build_mutation_prompt,
    parse_mutation_response,
)


class TestBuildMutationPrompt:
    def test_includes_all_signals(self):
        ctx = MutationContext(
            skill_id="my-skill",
            current_content="# original",
            strategy="optimize",
            private_score=0.4,
            public_score=0.9,
            correction_rate=0.1,
            success_rate=0.7,
            notes="reward hacking suspected",
        )
        out = build_mutation_prompt(ctx)
        assert "my-skill" in out
        assert "# original" in out
        assert "0.400" in out
        assert "0.900" in out
        assert "optimize" in out
        assert "reward hacking suspected" in out

    def test_default_template_has_placeholders(self):
        # Sanity: the shipped template has all the required slots.
        for placeholder in (
            "{skill_id}",
            "{current_content}",
            "{strategy}",
            "{private_score}",
            "{public_score}",
        ):
            assert placeholder in DEFAULT_MUTATION_PROMPT_TEMPLATE


class TestParseMutationResponse:
    def test_parses_raw_content(self):
        text = "# new SKILL.md\n\nSome new content."
        new, reason = parse_mutation_response(text)
        assert new == textwrap.dedent("# new SKILL.md\n\nSome new content.").strip()
        assert reason == ""

    def test_strips_fenced_block(self):
        text = "```markdown\n# new\n```"
        new, _ = parse_mutation_response(text)
        assert new == "# new"

    def test_extracts_reasoning_section(self):
        text = "# new SKILL.md\n\n<reasoning>\nI rewrote this because...\n</reasoning>"
        new, reason = parse_mutation_response(text)
        assert "new SKILL.md" in new
        assert "rewrote this because" in reason
        assert "<reasoning>" not in new

    def test_empty_response_returns_none(self):
        assert parse_mutation_response("") == (None, "")
        assert parse_mutation_response("   \n  ") == (None, "")

    # Bug 5 fix: content BEFORE <reasoning> must be preserved.
    def test_reasoning_block_with_real_content_before(self):
        """Bug 5: content before <reasoning> must NOT be silently dropped."""
        text = "# Real Skill\n\nThis is the real skill content.\n<reasoning>I thought about improving it</reasoning>"
        new_content, reasoning = parse_mutation_response(text)
        assert new_content is not None
        assert "Real Skill" in new_content
        assert "This is the real skill content" in new_content
        assert "I thought about improving it" in reasoning

    def test_reasoning_with_only_whitespace_before(self):
        """Only whitespace before <reasoning> → content is None."""
        text = "  \n<reasoning>only reasoning, no content</reasoning>"
        new_content, reasoning = parse_mutation_response(text)
        assert new_content is None

    # Bug 6 fix: preamble-only response must be rejected.
    def test_preamble_only_short_no_hash(self):
        """Bug 6: short preamble without heading must fail."""
        text = "Here is the new SKILL.md:"
        new_content, _ = parse_mutation_response(text)
        assert new_content is None, "preamble-only should be treated as failure"

    def test_preamble_with_real_heading_after(self):
        """Preamble followed by real heading → preamble stripped, content kept."""
        text = "Here is the improved skill:\n# Improved Skill\n\nContent here."
        new_content, reasoning = parse_mutation_response(text)
        assert new_content is not None
        assert "# Improved Skill" in new_content

    # Bug 7 fix: bool-as-score rejected.
    def test_bool_score_rejected(self):
        """Bug 7: {\"score\": true} must NOT be accepted as score=1."""
        from agent.llm_judge import parse_score_text
        text = '{"score": true, "reasoning": "looks good"}'
        score, _ = parse_score_text(text)
        assert score is None, "bool is not a valid score type"


class TestMutator:
    def test_returns_failure_when_aux_client_missing(self):
        mutator = DefaultSkillMuter()
        with patch.dict(sys.modules, {"agent.auxiliary_client": None}):
            result = mutator.mutate(
                MutationContext(skill_id="s", current_content="x", strategy="optimize")
            )
        assert result.success is False
        assert "auxiliary_client unavailable" in (result.error or "")

    def test_returns_failure_when_call_llm_raises(self):
        mutator = DefaultSkillMuter()
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("provider down"),
        ):
            result = mutator.mutate(
                MutationContext(skill_id="s", current_content="x", strategy="optimize")
            )
        assert result.success is False
        assert "provider down" in (result.error or "")

    def test_parses_clean_llm_response(self):
        class _Choice:
            class _Msg:
                content = "# new SKILL.md\n\nBetter content."

            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            model = "mutator-model"

        mutator = DefaultSkillMuter()
        with patch("agent.auxiliary_client.call_llm", return_value=_Resp()):
            result = mutator.mutate(
                MutationContext(
                    skill_id="my-skill",
                    current_content="# old",
                    strategy="optimize",
                )
            )
        assert result.success
        assert "Better content" in result.new_content
        assert result.model == "mutator-model"


# ---------------------------------------------------------------------------
# Applier
# ---------------------------------------------------------------------------


class TestApplier:
    def _setup_skill(self, tmp_path: Path, content: str = "# original") -> Path:
        skill_dir = tmp_path / "skills" / "demo-skill"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")
        return skill_file

    def test_apply_writes_new_content_and_cleans_up_backup(self, tmp_path: Path):
        """After a successful apply, the backup is deleted (Bug 3 fix)."""
        skill_file = self._setup_skill(tmp_path)
        applier = DefaultSkillMuterApplier()

        proposal = MutationProposal(
            new_content="# mutated", success=True, reasoning="test"
        )
        result = applier.apply("demo-skill", proposal, hermes_home=tmp_path)

        assert result.success is True
        assert skill_file.read_text(encoding="utf-8") == "# mutated"
        # Backup must be DELETED after success — prevents stale-rollback Bug 3.
        backup_path = tmp_path / "skills" / "demo-skill" / "SKILL.md.bak"
        assert not backup_path.exists(), "backup must be deleted after successful apply"

    def test_apply_fails_keeps_backup_for_rollback(self, tmp_path: Path):
        """When apply fails, the backup is preserved so rollback can restore."""
        skill_file = self._setup_skill(tmp_path)
        applier = DefaultSkillMuterApplier()

        # Manually corrupt the write by patching write_text to fail.
        original_write = Path.write_text

        def failing_write(self, *args, **kwargs):
            if str(self).endswith(".tmp"):
                raise OSError("disk full")
            return original_write(self, *args, **kwargs)

        proposal = MutationProposal(new_content="# mutated", success=True)
        with patch.object(Path, "write_text", failing_write):
            result = applier.apply("demo-skill", proposal, hermes_home=tmp_path)

        assert result.success is False
        # Original must be intact.
        assert skill_file.read_text(encoding="utf-8") == "# original"
        # Backup must still exist for rollback to work.
        backup_path = tmp_path / "skills" / "demo-skill" / "SKILL.md.bak"
        assert backup_path.exists()


# ---------------------------------------------------------------------------
# parse_mutation_response — Bug 5 + Bug 6
# ---------------------------------------------------------------------------


class TestApplier:
    def _setup_skill(self, tmp_path: Path, content: str = "# original") -> Path:
        skill_dir = tmp_path / "skills" / "demo-skill"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")
        return skill_file

    def test_apply_refuses_unsuccessful_proposal(self, tmp_path: Path):
        self._setup_skill(tmp_path)
        applier = DefaultSkillMuterApplier()

        proposal = MutationProposal(
            new_content="", success=False, error="mutator failed"
        )
        result = applier.apply("demo-skill", proposal, hermes_home=tmp_path)

        assert result.success is False
        assert "mutator failed" in (result.error or "")
        # Original untouched.
        assert (tmp_path / "skills" / "demo-skill" / "SKILL.md").read_text(
            encoding="utf-8"
        ) == "# original"

    def test_apply_refuses_empty_content(self, tmp_path: Path):
        self._setup_skill(tmp_path)
        applier = DefaultSkillMuterApplier()

        proposal = MutationProposal(new_content="   \n  ", success=True)
        result = applier.apply("demo-skill", proposal, hermes_home=tmp_path)

        assert result.success is False

    def test_rollback_returns_false_after_successful_apply(self, tmp_path: Path):
        """Bug 3 fix: backup is deleted after successful apply, so rollback returns False."""
        skill_file = self._setup_skill(tmp_path)
        applier = DefaultSkillMuterApplier()

        # Apply a mutation (success).
        proposal = MutationProposal(new_content="# mutated", success=True)
        result = applier.apply("demo-skill", proposal, hermes_home=tmp_path)
        assert result.success is True
        assert skill_file.read_text(encoding="utf-8") == "# mutated"

        # Rollback now returns False because backup was deleted (intentional).
        rolled = applier.rollback("demo-skill", hermes_home=tmp_path)
        assert rolled is False, "backup was deleted after apply, rollback must fail gracefully"

    def test_rollback_returns_false_when_no_backup(self, tmp_path: Path):
        self._setup_skill(tmp_path)
        applier = DefaultSkillMuterApplier()
        # No apply first, so no backup exists.
        assert applier.rollback("demo-skill", hermes_home=tmp_path) is False

    def test_apply_to_new_skill_creates_file(self, tmp_path: Path):
        """If the skill doesn't exist yet, apply creates SKILL.md (no backup)."""
        # Do not call _setup_skill — skill dir doesn't exist.
        applier = DefaultSkillMuterApplier()

        proposal = MutationProposal(new_content="# brand new skill", success=True)
        result = applier.apply("brand-new", proposal, hermes_home=tmp_path)

        assert result.success is True
        skill_file = tmp_path / "skills" / "brand-new" / "SKILL.md"
        assert skill_file.exists()
        assert skill_file.read_text(encoding="utf-8") == "# brand new skill"
        # No backup expected (there was nothing to back up).
        assert result.backup_path is None
