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

    def test_apply_writes_new_content_and_creates_backup(self, tmp_path: Path):
        skill_file = self._setup_skill(tmp_path)
        applier = DefaultSkillMuterApplier()

        proposal = MutationProposal(
            new_content="# mutated", success=True, reasoning="test"
        )
        result = applier.apply("demo-skill", proposal, hermes_home=tmp_path)

        assert result.success is True
        assert skill_file.read_text(encoding="utf-8") == "# mutated"
        backup_path = tmp_path / "skills" / "demo-skill" / "SKILL.md.bak"
        assert backup_path.exists()
        assert backup_path.read_text(encoding="utf-8") == "# original"

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

    def test_rollback_restores_original(self, tmp_path: Path):
        skill_file = self._setup_skill(tmp_path)
        applier = DefaultSkillMuterApplier()

        # Apply a mutation.
        proposal = MutationProposal(new_content="# mutated", success=True)
        applier.apply("demo-skill", proposal, hermes_home=tmp_path)
        assert skill_file.read_text(encoding="utf-8") == "# mutated"

        # Roll back.
        rolled = applier.rollback("demo-skill", hermes_home=tmp_path)
        assert rolled is True
        assert skill_file.read_text(encoding="utf-8") == "# original"

    def test_rollback_returns_false_when_no_backup(self, tmp_path: Path):
        self._setup_skill(tmp_path)
        applier = DefaultSkillMuterApplier()
        # No apply first, so no backup exists.
        assert applier.rollback("demo-skill", hermes_home=tmp_path) is False

    def test_apply_to_nonexistent_skill_creates_file_with_backup(self, tmp_path: Path):
        """If the skill doesn't exist yet, apply should still create
        SKILL.md (no backup, since there's nothing to back up).
        """
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
