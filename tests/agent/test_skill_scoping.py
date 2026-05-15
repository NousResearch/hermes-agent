"""Tests for agent-scoped skill filtering (Phase 2, P1).

Covers:
- extract_skill_agents()  — normal / boundary
- get_agent_profile_skills()  — normal / missing / empty
- _skill_matches_agent() — bidirectional lock logic
- build_skills_system_prompt(agent_id=...) — integration-level filtering
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.skill_utils import extract_skill_agents, get_agent_profile_skills
from agent.prompt_builder import _skill_matches_agent, build_skills_system_prompt


# =============================================================================
# extract_skill_agents
# =============================================================================


class TestExtractSkillAgents:
    """Tests for extract_skill_agents() normal and boundary cases."""

    def test_list_of_strings(self):
        """Normal case: frontmatter has a list of agent names."""
        fm = {"agents": ["claude", "codex", "openclaw"]}
        assert extract_skill_agents(fm) == ["claude", "codex", "openclaw"]

    def test_single_string(self):
        """String value is wrapped into a single-element list."""
        fm = {"agents": "claude"}
        assert extract_skill_agents(fm) == ["claude"]

    def test_field_absent(self):
        """Missing 'agents' key returns empty list."""
        assert extract_skill_agents({}) == []

    def test_field_none(self):
        """None value returns empty list."""
        assert extract_skill_agents({"agents": None}) == []

    def test_invalid_type_int(self):
        """Non-list, non-str type returns empty list."""
        assert extract_skill_agents({"agents": 42}) == []

    def test_invalid_type_dict(self):
        """Dict value returns empty list."""
        assert extract_skill_agents({"agents": {"a": "b"}}) == []

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert extract_skill_agents({"agents": []}) == []

    def test_strips_whitespace(self):
        """Agent names are stripped of whitespace."""
        fm = {"agents": ["  claude ", "codex  "]}
        assert extract_skill_agents(fm) == ["claude", "codex"]

    def test_mixed_valid_and_empty(self):
        """Empty strings are filtered out."""
        fm = {"agents": ["claude", "", "  ", "codex"]}
        assert extract_skill_agents(fm) == ["claude", "codex"]


# =============================================================================
# get_agent_profile_skills
# =============================================================================


class TestGetAgentProfileSkills:
    """Tests for get_agent_profile_skills() with mock registry."""

    @pytest.fixture
    def registry_content(self):
        """Sample agent-registry.json content for testing."""
        return {
            "agents": {
                "claude": {
                    "subagent_profile": {
                        "skills": ["code-review", "git-operations"],
                        "toolsets": ["file", "terminal"],
                    }
                },
                "codex": {
                    "subagent_profile": {
                        "skills": ["code-review"],
                        "toolsets": ["file", "terminal"],
                    }
                },
                "openclaw": {
                    "subagent_profile": {
                        "toolsets": ["terminal", "file"],
                    }
                },
                "no-profile": {
                    "toolsets": ["file"],
                },
                "string-skills": {
                    "subagent_profile": {
                        "skills": "single-skill",
                        "toolsets": ["file"],
                    }
                },
            }
        }

    def test_normal(self, registry_content, tmp_path):
        """Normal case: agent has skills list."""
        reg_path = tmp_path / "agent-registry.json"
        reg_path.write_text(json.dumps(registry_content), encoding="utf-8")
        skills = get_agent_profile_skills("claude", registry_path=reg_path)
        assert skills == ["code-review", "git-operations"]

    def test_agent_missing(self, registry_content, tmp_path):
        """Agent key not in registry returns empty list."""
        reg_path = tmp_path / "agent-registry.json"
        reg_path.write_text(json.dumps(registry_content), encoding="utf-8")
        skills = get_agent_profile_skills("nonexistent", registry_path=reg_path)
        assert skills == []

    def test_no_skills_field(self, registry_content, tmp_path):
        """Agent with no 'skills' field returns empty list."""
        reg_path = tmp_path / "agent-registry.json"
        reg_path.write_text(json.dumps(registry_content), encoding="utf-8")
        skills = get_agent_profile_skills("openclaw", registry_path=reg_path)
        assert skills == []

    def test_no_subagent_profile(self, registry_content, tmp_path):
        """Agent with no 'subagent_profile' key returns empty list."""
        reg_path = tmp_path / "agent-registry.json"
        reg_path.write_text(json.dumps(registry_content), encoding="utf-8")
        skills = get_agent_profile_skills("no-profile", registry_path=reg_path)
        assert skills == []

    def test_string_skills(self, registry_content, tmp_path):
        """String 'skills' value is wrapped into a list."""
        reg_path = tmp_path / "agent-registry.json"
        reg_path.write_text(json.dumps(registry_content), encoding="utf-8")
        skills = get_agent_profile_skills("string-skills", registry_path=reg_path)
        assert skills == ["single-skill"]

    def test_missing_registry_file(self, tmp_path):
        """Non-existent registry file returns empty list (no crash)."""
        missing = tmp_path / "does-not-exist.json"
        skills = get_agent_profile_skills("claude", registry_path=missing)
        assert skills == []

    def test_invalid_json(self, tmp_path):
        """Invalid JSON returns empty list (no crash)."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not valid json", encoding="utf-8")
        skills = get_agent_profile_skills("claude", registry_path=bad_path)
        assert skills == []

    def test_empty_skills_list(self, registry_content, tmp_path):
        """Empty skills list returns empty list."""
        reg_path = tmp_path / "agent-registry.json"
        registry_content["agents"]["empty-skills"] = {
            "subagent_profile": {"skills": [], "toolsets": ["file"]}
        }
        reg_path.write_text(json.dumps(registry_content), encoding="utf-8")
        skills = get_agent_profile_skills("empty-skills", registry_path=reg_path)
        assert skills == []


# =============================================================================
# _skill_matches_agent
# =============================================================================


class TestSkillMatchesAgent:
    """Tests for _skill_matches_agent bidirectional lock logic."""

    def test_agent_id_none_passes(self):
        """agent_id=None → skill always shows (backward compat)."""
        assert _skill_matches_agent("my-skill", "general", None, None) is True

    def test_agent_profile_skills_none_passes(self):
        """No agent profile skills → skill shows."""
        assert _skill_matches_agent("my-skill", "general", "claude", None) is True

    def test_agent_profile_skills_empty_blocks(self):
        """Empty set → no skills pass."""
        assert _skill_matches_agent("my-skill", "general", "claude", set()) is False

    def test_skill_in_whitelist_passes(self):
        """Skill in agent's profile skills → shown."""
        assert (
            _skill_matches_agent(
                "code-review", "general", "claude", {"code-review", "git-ops"}
            )
            is True
        )

    def test_skill_not_in_whitelist_blocked(self):
        """Skill NOT in agent's profile skills → hidden."""
        assert (
            _skill_matches_agent(
                "web-research", "general", "claude", {"code-review", "git-ops"}
            )
            is False
        )


# =============================================================================
# build_skills_system_prompt(agent_id=...) integration
# =============================================================================


class TestBuildSkillsSystemPromptAgentFilter:
    """Integration-level test: build_skills_system_prompt with agent_id.

    We mock the skills directory to contain a few test skills and verify that
    only skills matching the agent's profile appear in the output.
    """

    @pytest.fixture
    def skills_tree(self, tmp_path):
        """Create a minimal skills tree with three skills."""
        cats = tmp_path / "categories"
        for cat, skill_name, agents in [
            ("code", "code-review", ["claude", "codex"]),
            ("code", "git-operations", ["claude"]),
            ("research", "web-search", ["openclaw"]),
        ]:
            skill_dir = cats / cat / skill_name
            skill_dir.mkdir(parents=True)
            agents_yaml = "\n".join(f"  - {a}" for a in agents)
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: {skill_name}\ndescription: A test skill\n"
                f"agents:\n{agents_yaml}\n---\n# {skill_name}\n\nBody.\n",
                encoding="utf-8",
            )
        return cats

    @pytest.fixture
    def mock_hermes_home(self, tmp_path):
        """Point get_hermes_home to tmp_path so skills directory resolves there."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)
        return hermes_home

    def _build_skills_prompt(self, skills_dir, agent_id=None):
        """Helper: call build_skills_system_prompt with patched skills dir."""
        with patch("agent.prompt_builder.get_skills_dir", return_value=skills_dir):
            with patch(
                "agent.prompt_builder.get_all_skills_dirs",
                return_value=[skills_dir],
            ):
                return build_skills_system_prompt(
                    available_tools=None,
                    available_toolsets=None,
                    agent_id=agent_id,
                )

    def test_no_agent_id_shows_all(self, skills_tree):
        """agent_id=None → all skills shown."""
        prompt = self._build_skills_prompt(skills_tree, agent_id=None)
        assert "code-review" in prompt
        assert "git-operations" in prompt
        assert "web-search" in prompt

    def test_agent_id_filters_only_profile_skills(self, skills_tree):
        """agent_id=claude → only skills in claude's profile appear.

        We mock get_agent_profile_skills to return only skills for the given agent.
        """
        with patch(
            "agent.prompt_builder.get_agent_profile_skills",
            return_value=["code-review", "git-operations"],
        ):
            prompt = self._build_skills_prompt(skills_tree, agent_id="claude")
        assert "code-review" in prompt
        assert "git-operations" in prompt
        assert "web-search" not in prompt

    def test_agent_id_no_profile_skills_shows_all(self, skills_tree):
        """agent_id set but no profile skills → backward compat: show all."""
        with patch(
            "agent.prompt_builder.get_agent_profile_skills",
            return_value=[],
        ):
            prompt = self._build_skills_prompt(skills_tree, agent_id="claude")
        assert "code-review" in prompt
        assert "git-operations" in prompt
        assert "web-search" in prompt

    def test_caching_respects_agent_id(self, skills_tree):
        """Cache key includes agent_id so different agents get different prompts."""
        with patch(
            "agent.prompt_builder.get_agent_profile_skills",
            side_effect=lambda aid: {
                "claude": ["code-review", "git-operations"],
                "codex": ["code-review"],
            }.get(aid, []),
        ):
            prompt_claude = self._build_skills_prompt(skills_tree, agent_id="claude")
            prompt_codex = self._build_skills_prompt(skills_tree, agent_id="codex")

        # claude sees git-operations; codex does not
        assert "git-operations" in prompt_claude
        assert "git-operations" not in prompt_codex


# =============================================================================
# P2: Main-agent whitelist tests
# =============================================================================


class TestHermesWhitelist:
    """P2: 主 Agent 白名单验证"""

    def test_hermes_core_skills_defined(self):
        """HERMES_CORE_SKILLS 包含核心 skill"""
        from agent.prompt_builder import HERMES_CORE_SKILLS
        assert "hermes-knowledge-architecture" in HERMES_CORE_SKILLS
        assert "hermes-cron-management" in HERMES_CORE_SKILLS
        assert "office-hours" in HERMES_CORE_SKILLS
        assert len(HERMES_CORE_SKILLS) >= 7

    def test_hermes_whitelist_no_leak_generic_skills(self):
        """主 Agent 白名单不包含通用编码/设计 skill"""
        from agent.prompt_builder import HERMES_CORE_SKILLS
        generic = {"spike", "sketch", "github-code-review", "python-debugpy", "design-md", "claude-design"}
        leaked = HERMES_CORE_SKILLS & generic
        assert not leaked, f"hermes whitelist leaked generic skills: {leaked}"

    @pytest.fixture
    def skills_tree(self, tmp_path):
        """Create a minimal skills tree with skills for testing scoping."""
        cats = tmp_path / "categories"
        for cat, skill_name in [
            ("code", "code-review"),
            ("code", "git-operations"),
            ("code", "python-debugpy"),
            ("research", "web-search"),
            ("research", "spike"),
            ("design", "design-md"),
            ("design", "sketch"),
            ("core", "hermes-knowledge-architecture"),
            ("core", "hermes-cron-management"),
            ("core", "hermes-subagent-delegation"),
            ("core", "hermes-gateway-debug"),
            ("core", "hermes-multi-agent-research"),
            ("core", "hermes-webui"),
            ("core", "hermes-agent-skill-authoring"),
            ("core", "office-hours"),
        ]:
            skill_dir = cats / cat / skill_name
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: {skill_name}\ndescription: A skill\n---\n# {skill_name}\n\nBody.\n",
                encoding="utf-8",
            )
        return cats

    def _build_with_mocks(self, skills_dir, agent_id, profile_skills=None):
        """Call build_skills_system_prompt with patched deps."""
        from agent.prompt_builder import build_skills_system_prompt
        with patch("agent.prompt_builder.get_skills_dir", return_value=skills_dir):
            with patch("agent.prompt_builder.get_all_skills_dirs", return_value=[skills_dir]):
                if profile_skills is not None:
                    with patch(
                        "agent.prompt_builder.get_agent_profile_skills",
                        return_value=profile_skills,
                    ):
                        return build_skills_system_prompt(agent_id=agent_id)
                return build_skills_system_prompt(agent_id=agent_id)

    def _count_skills(self, prompt: str) -> int:
        """Count skill entries in the rendered prompt."""
        return prompt.count("\n    - ")

    def test_build_with_agent_none_shows_all(self, skills_tree):
        """agent_id=None → 全量（向后兼容）"""
        prompt = self._build_with_mocks(skills_tree, agent_id=None)
        count = self._count_skills(prompt)
        assert count >= 14, f"Expected many skills, got {count}"

    @pytest.mark.parametrize("agent_id,profile_skills,min_count", [
        ("hermes", [], 8),          # 主 Agent 白名单（忽略 profile，只从 HERMES_CORE_SKILLS 取）
        ("deepseek-tui", ["code-review", "git-operations"], 1),
        ("pirlo", ["design-md", "sketch"], 1),
    ])
    def test_build_with_agent_shows_scoped(self, skills_tree, agent_id, profile_skills, min_count):
        """各种 agent_id 返回正确数量的 skill"""
        prompt = self._build_with_mocks(skills_tree, agent_id=agent_id, profile_skills=profile_skills)
        count = self._count_skills(prompt)
        assert count >= min_count, f"agent_id={agent_id}: expected >= {min_count} skills, got {count}"
