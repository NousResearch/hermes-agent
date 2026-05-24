"""Unit tests for _skill_passes_project_filters and parse_project_skill_config."""

import pytest
import tempfile
from pathlib import Path

from agent.prompt_builder import (
    _skill_passes_project_filters,
    parse_project_skill_config,
)


class TestSkillPassesProjectFilters:
    """Tests for the module-level filter function."""

    # ── No filters: everything passes ──
    def test_no_filters_passes(self):
        assert _skill_passes_project_filters("any-skill", "any-cat") is True

    # ── Positive include by name ──
    def test_include_by_name_matches(self):
        assert _skill_passes_project_filters(
            "my-skill", "devops", include_set={"my-skill"}
        ) is True

    def test_include_by_name_case_insensitive(self):
        assert _skill_passes_project_filters(
            "My-Skill", "devops", include_set={"my-skill"}
        ) is True

    def test_include_by_name_no_match(self):
        assert _skill_passes_project_filters(
            "other-skill", "devops", include_set={"my-skill"}
        ) is False

    # ── Positive include by category ──
    def test_include_by_category_matches(self):
        assert _skill_passes_project_filters(
            "any-skill", "devops", include_set={"devops"}
        ) is True

    def test_include_by_category_prefix(self):
        assert _skill_passes_project_filters(
            "ci-skill", "devops/ci", include_set={"devops"}
        ) is True

    def test_include_by_category_no_match(self):
        assert _skill_passes_project_filters(
            "any-skill", "security", include_set={"devops"}
        ) is False

    # ── Positive include by alt_name ──
    def test_include_by_alt_name_matches(self):
        assert _skill_passes_project_filters(
            "frontmatter-name", "general",
            include_set={"dir-name"}, alt_name="dir-name"
        ) is True

    def test_include_by_alt_name_no_match(self):
        assert _skill_passes_project_filters(
            "frontmatter-name", "general",
            include_set={"dir-name"}, alt_name="other-name"
        ) is False

    # ── Categories include ──
    def test_cats_include_matches(self):
        assert _skill_passes_project_filters(
            "any-skill", "software-development",
            cats_include_set={"software-development"}
        ) is True

    def test_cats_include_prefix(self):
        assert _skill_passes_project_filters(
            "any-skill", "software-development/testing",
            cats_include_set={"software-development"}
        ) is True

    def test_cats_include_no_match(self):
        assert _skill_passes_project_filters(
            "any-skill", "gaming",
            cats_include_set={"software-development"}
        ) is False

    # ── OR semantics: include + cats_include ──
    def test_include_or_cats_include(self):
        # Name doesn't match include_set, but category matches cats_include_set
        assert _skill_passes_project_filters(
            "any-skill", "software-development",
            include_set={"specific-skill"},
            cats_include_set={"software-development"}
        ) is True

    # ── Negative exclude by name ──
    def test_exclude_by_name(self):
        assert _skill_passes_project_filters(
            "bad-skill", "devops", exclude_set={"bad-skill"}
        ) is False

    def test_exclude_by_name_case_insensitive(self):
        assert _skill_passes_project_filters(
            "Bad-Skill", "devops", exclude_set={"bad-skill"}
        ) is False

    # ── Negative exclude by category ──
    def test_exclude_by_category(self):
        assert _skill_passes_project_filters(
            "any-skill", "gaming", exclude_set={"gaming"}
        ) is False

    def test_exclude_by_category_prefix(self):
        assert _skill_passes_project_filters(
            "any-skill", "gaming/rpg", exclude_set={"gaming"}
        ) is False

    # ── Negative exclude by alt_name ──
    def test_exclude_by_alt_name(self):
        assert _skill_passes_project_filters(
            "frontmatter-name", "general",
            exclude_set={"dir-name"}, alt_name="dir-name"
        ) is False

    # ── Negative cats_exclude ──
    def test_cats_exclude(self):
        assert _skill_passes_project_filters(
            "any-skill", "gaming",
            cats_exclude_set={"gaming"}
        ) is False

    # ── Include wins over exclude? No — exclude runs first for same field ──
    def test_exclude_overrides_include_same_set(self):
        # Name in both include and exclude: exclude wins (checked after include)
        assert _skill_passes_project_filters(
            "skill-x", "devops",
            include_set={"skill-x"}, exclude_set={"skill-x"}
        ) is False

    # ── Include set but empty → treated as no filter ──
    def test_empty_include_passes(self):
        assert _skill_passes_project_filters(
            "any-skill", "any-cat", include_set=set()
        ) is True

    # ── None values ──
    def test_none_sets_passes(self):
        assert _skill_passes_project_filters(
            "any-skill", "any-cat",
            include_set=None, exclude_set=None,
            cats_include_set=None, cats_exclude_set=None
        ) is True

    # ── Real-world proxxied config ──
    def test_proxxied_config_include(self):
        """Simulate proxxied project: include only software-development skills."""
        proxxied_skills = [
            ("caveman", "software-development"),
            ("proxxied-cli", "software-development"),
            ("deep-review", "software-development"),
            ("github-pr-workflow", "github"),
            ("github-code-review", "github"),
        ]
        include = {"software-development", "github"}
        for name, cat in proxxied_skills:
            assert _skill_passes_project_filters(name, cat, include_set=include) is True

    def test_proxxied_config_exclude(self):
        """Simulate proxxied exclude: gaming, smart-home, etc."""
        excluded_skills = [
            ("gaming-skill", "gaming"),
            ("smart-home-skill", "smart-home"),
            ("email-skill", "email"),
        ]
        exclude = {"gaming", "smart-home", "email"}
        for name, cat in excluded_skills:
            assert _skill_passes_project_filters(name, cat, exclude_set=exclude) is False

    def test_top_level_skill_category_general(self):
        """Top-level skills get category 'general' — should match categories_include."""
        assert _skill_passes_project_filters(
            "my-skill", "general", cats_include_set={"general"}
        ) is True


class TestParseProjectSkillConfig:
    """Tests for the context-file/config.yaml skill config parser."""

    def test_empty_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = parse_project_skill_config()
        assert result == {}

    def test_bracket_format_include(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("skills.include: [skill-a, skill-b]")
        result = parse_project_skill_config()
        assert result["include"] == ["skill-a", "skill-b"]

    def test_bracket_format_exclude(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("skills.exclude: [gaming, smart-home]")
        result = parse_project_skill_config()
        assert result["exclude"] == ["gaming", "smart-home"]

    def test_categories_include(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text(
            "skills.categories.include: [software-development, github]"
        )
        result = parse_project_skill_config()
        assert result["categories_include"] == ["software-development", "github"]

    def test_categories_exclude(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text(
            "skills.categories.exclude: [gaming, smart-home, email]"
        )
        result = parse_project_skill_config()
        assert result["categories_exclude"] == ["gaming", "smart-home", "email"]

    def test_index_format_compact(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("skills.index_format: compact")
        result = parse_project_skill_config()
        assert result["index_format"] == "compact"

    def test_index_format_invalid_warns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("skills.index_format: invalid")
        result = parse_project_skill_config()
        assert "index_format" not in result  # invalid → not set

    def test_code_fence_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text(
            "```\n"
            "skills.include: [should-be-ignored]\n"
            "```\n"
            "skills.include: [real-skill]\n"
        )
        result = parse_project_skill_config()
        assert result["include"] == ["real-skill"]

    def test_yaml_list_format(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text(
            "skills.include:\n"
            "  - skill-a\n"
            "  - skill-b\n"
        )
        result = parse_project_skill_config()
        assert result["include"] == ["skill-a", "skill-b"]

    def test_hermes_md_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".hermes.md").write_text("skills.include: [from-dot-hermes]")
        result = parse_project_skill_config()
        assert result["include"] == ["from-dot-hermes"]

    def test_hermes_md_lowercase(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "hermes.md").write_text("skills.include: [from-hermes]")
        result = parse_project_skill_config()
        assert result["include"] == ["from-hermes"]

    def test_config_yaml_fallback(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_dir = tmp_path / "config.yaml"
        config_dir.write_text(
            "skills:\n"
            "  project:\n"
            "    include:\n"
            "      - from-config\n"
            "    index_format: compact\n"
        )
        result = parse_project_skill_config()
        assert result.get("include") == ["from-config"]
        assert result.get("index_format") == "compact"

    def test_context_overrides_config(self, tmp_path, monkeypatch):
        """Context file values should take precedence over config.yaml."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "AGENTS.md").write_text("skills.include: [from-context]")
        (tmp_path / "config.yaml").write_text(
            "skills:\n"
            "  project:\n"
            "    include:\n"
            "      - from-config\n"
        )
        result = parse_project_skill_config()
        # Context file wins for include
        assert "from-context" in result.get("include", [])

    def test_trailing_comment_not_injected(self, tmp_path, monkeypatch):
        """Comments after bracket close should NOT be treated as items."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text(
            "skills.include: [a, b]  # project tools"
        )
        result = parse_project_skill_config()
        assert "project" not in result.get("include", [])
        assert "tools" not in result.get("include", [])

    def test_lazy_index_format(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("skills.index_format: lazy")
        result = parse_project_skill_config()
        assert result["index_format"] == "lazy"
