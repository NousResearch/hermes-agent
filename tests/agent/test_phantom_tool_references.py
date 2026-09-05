"""Phantom tool references: system-prompt blocks must not name tools the
session can't call (Blank Slate audit, Aug 2026).

Covers:
  * HERMES_AGENT_HELP_GUIDANCE degrades to the docs-only variant when the
    skill tools aren't loaded.
  * execution_guidance_text() drops web_search lines when web tools are off.
  * The coding operating brief drops the `todo` sentence when the todo tool
    isn't loaded.
  * ESSENTIAL_SKILLS can't be disabled via config, and the CLI writer strips
    them from persisted disabled lists.
"""

from pathlib import Path


class TestHermesAgentHelpGuidance:
    def test_skill_variant_used_when_skill_view_present(self):
        from agent.prompt_builder import HERMES_AGENT_HELP_GUIDANCE
        assert "skill_view(name='hermes-agent')" in HERMES_AGENT_HELP_GUIDANCE

    def test_no_skills_variant_has_no_skill_view_reference(self):
        from agent.prompt_builder import HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS
        assert "skill_view" not in HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS
        assert "hermes-agent.nousresearch.com/docs" in HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS


class TestExecutionGuidanceText:
    def test_full_text_when_web_search_available(self):
        from agent.prompt_builder import (
            OPENAI_MODEL_EXECUTION_GUIDANCE,
            execution_guidance_text,
        )
        assert execution_guidance_text({"web_search", "terminal"}) == (
            OPENAI_MODEL_EXECUTION_GUIDANCE
        )

    def test_full_text_when_toolset_unknown(self):
        from agent.prompt_builder import (
            OPENAI_MODEL_EXECUTION_GUIDANCE,
            execution_guidance_text,
        )
        assert execution_guidance_text(None) == OPENAI_MODEL_EXECUTION_GUIDANCE

    def test_web_search_dropped_without_web_tools(self):
        from agent.prompt_builder import execution_guidance_text
        text = execution_guidance_text({"terminal", "read_file"})
        assert "web_search" not in text
        # The surrounding structure survives.
        assert "<mandatory_tool_use>" in text
        assert "<missing_context>" in text
        assert "(search_files, read_file, etc.)" in text


class TestCodingBriefTodoGating:
    def _brief(self, valid_tool_names):
        from agent.coding_context import CODING_PROFILE, RuntimeMode
        mode = RuntimeMode(
            profile=CODING_PROFILE, surface="cli", cwd=Path.cwd(),
        )
        prefix, _ws, _tr = mode.system_prompt_parts(
            valid_tool_names=valid_tool_names
        )
        assert prefix, "coding profile must emit an operating brief"
        return prefix[0]

    def test_todo_kept_when_tool_available(self):
        brief = self._brief({"todo_list", "terminal", "read_file"})
        assert "Track multi-step work with `todo_list`" in brief

    def test_todo_dropped_when_tool_missing(self):
        brief = self._brief({"terminal", "read_file"})
        assert "`todo`" not in brief
        # The path:line half of the merged bullet survives.
        assert "path:line" in brief

    def test_unknown_toolset_keeps_full_brief(self):
        brief = self._brief(None)
        assert "Track multi-step work with `todo_list`" in brief


class TestEssentialSkillsUndisableable:
    def test_agent_side_reader_strips_essential(self, monkeypatch, tmp_path):
        import agent.skill_utils as su
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "skills:\n  disabled:\n    - hermes-agent\n    - some-other-skill\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(su, "get_config_path", lambda: cfg)
        su._RAW_CONFIG_CACHE.clear()
        disabled = su.get_disabled_skill_names(platform="cli")
        assert "hermes-agent" not in disabled
        assert "some-other-skill" in disabled

    def test_cli_side_reader_strips_essential(self):
        from hermes_cli.skills_config import get_disabled_skills
        cfg = {"skills": {"disabled": ["hermes-agent", "other"]}}
        disabled = get_disabled_skills(cfg)
        assert "hermes-agent" not in disabled
        assert "other" in disabled

    def test_cli_side_writer_strips_essential(self, monkeypatch):
        import hermes_cli.skills_config as sc
        saved = {}
        monkeypatch.setattr(sc, "save_config", lambda cfg: saved.update(cfg))
        cfg = {}
        sc.save_disabled_skills(cfg, {"hermes-agent", "other"})
        assert cfg["skills"]["disabled"] == ["other"]

    def test_skill_manage_delete_refused(self):
        from tools.skill_manager_guards import _pinned_guard
        msg = _pinned_guard("hermes-agent")
        assert msg is not None
        assert "essential" in msg.lower()


class TestEssentialOnlySync:
    def test_opted_out_sync_seeds_only_essential(self, monkeypatch, tmp_path):
        """A profile with .no-bundled-skills still gets the hermes-agent skill."""
        import tools.skills_sync as ss

        home = tmp_path / ".hermes"
        home.mkdir()
        (home / ss.NO_BUNDLED_SKILLS_MARKER).write_text("", encoding="utf-8")

        bundled = tmp_path / "bundled"
        for cat, name in [
            ("autonomous-ai-agents", "hermes-agent"),
            ("media", "gif-search"),
        ]:
            d = bundled / cat / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: x\n---\nbody\n",
                encoding="utf-8",
            )

        monkeypatch.setattr(ss, "_hermes_home", lambda: home)
        monkeypatch.setattr(ss, "_get_bundled_dir", lambda: bundled)
        monkeypatch.setattr(ss, "_build_external_skill_index", lambda: set())

        result = ss.sync_skills(quiet=True)

        assert result["skipped_opt_out"] is True
        assert result["copied"] == ["hermes-agent"]
        assert (home / "skills" / "autonomous-ai-agents" / "hermes-agent" / "SKILL.md").exists()
        assert not (home / "skills" / "media").exists()
