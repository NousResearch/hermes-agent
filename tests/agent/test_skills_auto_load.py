"""skills.auto_load: pinned skills land in every new session's prompt, resolved once per agent."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _write_skill(root, name, body):
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Test.\n---\n\n{body}\n")
    return skill_dir / "SKILL.md"


def _bare_agent(session_id="auto-load-test"):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.valid_tool_names = {"skills_list", "skill_view", "skill_manage"}
    agent.model = "test-model"
    agent.provider = "test"
    agent.pass_session_id = False
    agent.skip_context_files = False
    agent._context_cwd_is_launch_artifact = True  # no project-context walk; keeps the build tmp-home only
    agent.load_soul_identity = False
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent._memory_manager = None
    agent._memory_store = None
    agent.session_id = session_id
    agent.platform = "telegram"
    agent._tool_use_enforcement = False
    agent.ephemeral_system_prompt = None
    agent._cached_system_prompt = None
    agent._auto_load_skills_resolved = False
    agent._auto_load_skills_result = ("", [], [])
    return agent


class TestBuildAutoLoadPrompt:
    def test_loads_configured_skills_and_reports_missing(self, tmp_path):
        """Config AND skill lookup resolve under *home_override* (profile-scoped), not the ambient home."""
        from agent.skill_commands import build_auto_load_prompt

        home = tmp_path / "profile-home"
        _write_skill(home / "skills", "pinned-skill", "PINNED CONTENT")
        (home / "config.yaml").write_text(
            "skills:\n  auto_load: ['pinned-skill', ' pinned-skill ', 'no-such-skill', 7]\n", encoding="utf-8")
        prompt, loaded, missing = build_auto_load_prompt(task_id="s1", home_override=home)
        assert loaded == ["pinned-skill"]
        assert missing == ["no-such-skill"]
        assert "PINNED CONTENT" in prompt
        assert prompt.count("auto-loaded via config (skills.auto_load)") == 1

    def test_explicit_preload_dedupes_against_auto_loaded_names(self, tmp_path):
        from agent.skill_commands import build_preloaded_skills_prompt

        _write_skill(tmp_path, "pinned-skill", "PINNED CONTENT")
        _write_skill(tmp_path, "extra-skill", "EXTRA CONTENT")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            prompt, loaded, missing = build_preloaded_skills_prompt(
                ["pinned-skill", "extra-skill"], task_id="s1", excluded_loaded_names={"pinned-skill"},
            )
        # Still counted as resolved (so a typo elsewhere degrades gracefully), but not rendered twice.
        assert loaded == ["pinned-skill", "extra-skill"] and missing == []
        assert "PINNED CONTENT" not in prompt and "EXTRA CONTENT" in prompt


class TestSharedPromptPath:
    @pytest.mark.parametrize("tool_names", [set(), {"terminal"}])
    def test_auto_load_without_skills_tools(self, tmp_path, monkeypatch, tool_names):
        """Pinned guidance is full content, not a callable-tool capability."""
        monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
        skill_file = _write_skill(tmp_path, "pinned-skill", "FULL PINNED BODY")
        cfg = {"skills": {"auto_load": ["pinned-skill"]}}
        agent = _bare_agent()
        agent.valid_tool_names = tool_names.copy()
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path), patch(
            "hermes_cli.config.load_config_readonly", return_value=cfg
        ):
            first = agent._build_system_prompt()
            assert first.count("FULL PINNED BODY") == 1
            assert agent.valid_tool_names == tool_names
            assert agent._auto_load_skills_result[1] == ["pinned-skill"]
            from agent.skill_commands import build_preloaded_skills_prompt

            explicit, loaded, missing = build_preloaded_skills_prompt(
                ["pinned-skill"], excluded_loaded_names=set(agent._auto_load_skills_result[1])
            )
            assert loaded == ["pinned-skill"] and missing == []
            assert "FULL PINNED BODY" not in explicit
            skill_file.write_text("MUTATED BODY", encoding="utf-8")
            cfg["skills"]["auto_load"] = []
            agent._cached_system_prompt = None
            rebuilt = agent._build_system_prompt()
            assert rebuilt == first
            assert agent.valid_tool_names == tool_names

    def test_prompt_is_byte_stable_after_config_and_skill_mutation(self, tmp_path, monkeypatch):
        """The whole point: rebuilds (model switch, compression) reuse the first resolution."""
        monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
        skill_file = _write_skill(tmp_path, "stable-skill", "ORIGINAL SKILL BYTES")
        cfg = {"skills": {"auto_load": ["stable-skill"]}}
        agent = _bare_agent()
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path), \
             patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            first = agent._build_system_prompt()
            assert "ORIGINAL SKILL BYTES" in first and agent._auto_load_skills_result[1] == ["stable-skill"]
            skill_file.write_text("---\nname: stable-skill\ndescription: Test.\n---\n\nMUTATED BYTES\n")
            cfg["skills"]["auto_load"] = []
            agent.model = "after-switch"
            agent._cached_system_prompt = None
            rebuilt = agent._build_system_prompt()
        assert "ORIGINAL SKILL BYTES" in rebuilt and "MUTATED BYTES" not in rebuilt

    def test_separate_homes_without_tools(self, tmp_path, monkeypatch):
        from agent.system_prompt import _auto_load_parts

        monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
        agents = []
        for name in ("first", "second"):
            home = tmp_path / name
            _write_skill(home / "skills", "pinned-skill", f"BODY FROM {name}")
            (home / "config.yaml").write_text(
                "skills:\n  auto_load: [pinned-skill]\n", encoding="utf-8"
            )
            agent = _bare_agent(name)
            agent.valid_tool_names = set()
            with patch("agent.system_prompt._agent_home", return_value=home):
                parts = _auto_load_parts(agent)
            assert "BODY FROM " + name in "".join(parts)
            agents.append((agent, parts))
        for agent, parts in agents:
            assert _auto_load_parts(agent) == parts
        assert "BODY FROM second" not in "".join(agents[0][1])
        assert "BODY FROM first" not in "".join(agents[1][1])

    @pytest.mark.parametrize("tool_names", [set(), {"skill_view"}])
    def test_gates_suppress_auto_load(self, tmp_path, monkeypatch, tool_names):
        """Ignore-rules and internal forks still suppress pinned guidance."""
        _write_skill(tmp_path, "stable-skill", "ORIGINAL SKILL BYTES")
        cfg = {"skills": {"auto_load": ["stable-skill"]}}
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path), \
             patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            monkeypatch.setenv("HERMES_IGNORE_RULES", "true")
            agent = _bare_agent()
            agent.valid_tool_names = tool_names.copy()
            assert "ORIGINAL SKILL BYTES" not in agent._build_system_prompt()
            assert agent._auto_load_skills_resolved is True and agent._auto_load_skills_result == ("", [], [])

            monkeypatch.delenv("HERMES_IGNORE_RULES")
            child = _bare_agent("child")
            child.valid_tool_names = tool_names.copy()
            child.skip_context_files = True
            assert "ORIGINAL SKILL BYTES" not in child._build_system_prompt()
            no_skills = _bare_agent("no-skills")
            no_skills.valid_tool_names = {"memory"}
            assert "ORIGINAL SKILL BYTES" in no_skills._build_system_prompt()
            assert "ORIGINAL SKILL BYTES" in _bare_agent("full")._build_system_prompt()
