"""Tests for agent/skill_utils.py."""

from pathlib import Path
from unittest.mock import patch

from agent.skill_utils import (
    extract_skill_conditions,
    get_disabled_skill_names,
    get_external_skills_dirs,
    is_excluded_skill_path,
    is_external_skill_path,
    is_skill_support_path,
    iter_skill_index_files,
    resolve_skill_config_values,
    skill_matches_platform,
    skill_matches_platform_list,
)


def test_metadata_as_dict_with_hermes():
    """Normal case: metadata is a dict containing hermes keys."""
    frontmatter = {
        "metadata": {
            "hermes": {
                "fallback_for_toolsets": ["toolset_a"],
                "requires_toolsets": ["toolset_b"],
                "fallback_for_tools": ["tool_x"],
                "requires_tools": ["tool_y"],
            }
        }
    }
    result = extract_skill_conditions(frontmatter)
    assert result["fallback_for_toolsets"] == ["toolset_a"]
    assert result["requires_toolsets"] == ["toolset_b"]
    assert result["fallback_for_tools"] == ["tool_x"]
    assert result["requires_tools"] == ["tool_y"]


def test_metadata_as_string_does_not_crash():
    """Bug case: metadata is a non-dict truthy value."""
    frontmatter = {"metadata": "some text"}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


def test_metadata_as_none():
    """metadata key is present but set to null/None."""
    frontmatter = {"metadata": None}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


def test_metadata_missing_entirely():
    """metadata key is absent from frontmatter."""
    frontmatter = {"name": "my-skill", "description": "Does stuff."}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


def test_iter_skill_index_files_prunes_dependency_dirs(tmp_path):
    real = tmp_path / "real-skill"
    real.mkdir()
    (real / "SKILL.md").write_text("---\nname: real-skill\n---\n", encoding="utf-8")

    nested = (
        tmp_path
        / "bring"
        / "scripts"
        / ".venv"
        / "lib"
        / "python3.13"
        / "site-packages"
        / "typer"
        / ".agents"
        / "skills"
        / "typer"
    )
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text("---\nname: typer\n---\n", encoding="utf-8")

    node_module = (
        tmp_path
        / "web-skill"
        / "node_modules"
        / "dep"
        / ".agents"
        / "skills"
        / "dep"
    )
    node_module.mkdir(parents=True)
    (node_module / "SKILL.md").write_text("---\nname: dep\n---\n", encoding="utf-8")

    found = list(iter_skill_index_files(tmp_path, "SKILL.md"))

    assert found == [real / "SKILL.md"]


def test_skill_config_helpers_share_raw_config_parse_cache(tmp_path, monkeypatch):
    """Repeated skill config helpers should parse config.yaml only once."""
    from agent import skill_utils

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    external = tmp_path / "external-skills"
    external.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        f"""
skills:
  disabled:
    - hidden-skill
  external_dirs:
    - {external}
  config:
    wiki:
      path: ~/wiki
""".strip(),
        encoding="utf-8",
    )
    parse_count = 0
    real_yaml_load = skill_utils.yaml_load

    def counting_yaml_load(text):
        nonlocal parse_count
        parse_count += 1
        return real_yaml_load(text)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    skill_utils._external_dirs_cache_clear()
    getattr(skill_utils, "_raw_config_cache_clear", lambda: None)()
    monkeypatch.setattr(skill_utils, "yaml_load", counting_yaml_load)

    assert get_disabled_skill_names() == {"hidden-skill"}
    assert get_external_skills_dirs() == [external.resolve()]
    assert resolve_skill_config_values(
        [{"key": "wiki.path", "description": "Wiki path"}]
    )["wiki.path"].endswith("/wiki")
    assert parse_count == 1


def test_skill_config_raw_cache_invalidates_on_config_edit(tmp_path, monkeypatch):
    """Editing config.yaml should invalidate the shared raw config cache."""
    from agent import skill_utils
    import os

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text("skills:\n  disabled: [old-skill]\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    skill_utils._external_dirs_cache_clear()
    assert get_disabled_skill_names() == {"old-skill"}

    config_path.write_text("skills:\n  disabled: [new-skill]\n", encoding="utf-8")
    os.utime(config_path, None)

    assert get_disabled_skill_names() == {"new-skill"}


def test_is_external_skill_path_matches_configured_external_dir(tmp_path, monkeypatch):
    from agent import skill_utils

    hermes_home = tmp_path / ".hermes"
    local_skills = hermes_home / "skills"
    external = tmp_path / "external-skills"
    local_skills.mkdir(parents=True)
    external.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    skill_utils._external_dirs_cache_clear()

    assert is_external_skill_path(external / "team-skill" / "SKILL.md") is True
    assert is_external_skill_path(local_skills / "local-skill" / "SKILL.md") is False


def test_iter_skill_index_files_prunes_skill_support_dirs(tmp_path):
    """Archived package SKILL.md files under support dirs are not active skills."""
    real = tmp_path / "umbrella"
    real.mkdir()
    (real / "SKILL.md").write_text("---\nname: umbrella\n---\n", encoding="utf-8")

    package = real / "references" / "old-skill-package"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text("---\nname: old-skill\n---\n", encoding="utf-8")
    (package / "DESCRIPTION.md").write_text(
        "---\ndescription: archived package\n---\n", encoding="utf-8"
    )

    script_package = real / "scripts" / "helper-skill"
    script_package.mkdir(parents=True)
    (script_package / "SKILL.md").write_text("---\nname: helper\n---\n", encoding="utf-8")

    found = list(iter_skill_index_files(tmp_path, "SKILL.md"))
    desc_found = list(iter_skill_index_files(tmp_path, "DESCRIPTION.md"))

    assert found == [real / "SKILL.md"]
    assert desc_found == []
    assert is_skill_support_path(package / "SKILL.md") is True
    assert is_excluded_skill_path(package / "SKILL.md") is True


def test_iter_skill_index_files_keeps_support_named_categories(tmp_path):
    """A category named scripts/templates/assets/references is still valid."""
    scripts_skill = tmp_path / "scripts" / "bash-helper"
    scripts_skill.mkdir(parents=True)
    (scripts_skill / "SKILL.md").write_text(
        "---\nname: bash-helper\n---\n", encoding="utf-8"
    )

    templates_skill = tmp_path / "templates" / "deck-template"
    templates_skill.mkdir(parents=True)
    (templates_skill / "SKILL.md").write_text(
        "---\nname: deck-template\n---\n", encoding="utf-8"
    )

    found = list(iter_skill_index_files(tmp_path, "SKILL.md"))

    assert found == [scripts_skill / "SKILL.md", templates_skill / "SKILL.md"]
    assert is_skill_support_path(scripts_skill / "SKILL.md") is False
    assert is_excluded_skill_path(scripts_skill / "SKILL.md") is False


def test_iter_skill_index_files_prunes_bak_dirs(tmp_path):
    live = tmp_path / "hermes-profile"
    live.mkdir()
    (live / "SKILL.md").write_text("---\nname: hermes-profile\n---\n", encoding="utf-8")

    backup = tmp_path / "hermes-profile.bak-20260510_233500"
    backup.mkdir()
    (backup / "SKILL.md").write_text(
        "---\nname: hermes-profile-backup\n---\n", encoding="utf-8"
    )

    found = list(iter_skill_index_files(tmp_path, "SKILL.md"))

    assert found == [live / "SKILL.md"]


def test_iter_skill_index_files_prunes_dot_bak_dirs(tmp_path):
    backup = tmp_path / ".bak-20260510_233500"
    backup.mkdir()
    (backup / "SKILL.md").write_text("---\nname: backup\n---\n", encoding="utf-8")

    found = list(iter_skill_index_files(tmp_path, "SKILL.md"))

    assert found == []


def test_iter_skill_index_files_prunes_backup_dirs(tmp_path):
    dot_backup = tmp_path / ".backup-old"
    dot_backup.mkdir()
    (dot_backup / "SKILL.md").write_text(
        "---\nname: dot-backup\n---\n", encoding="utf-8"
    )

    named_backup = tmp_path / "my-skill.backup-20260510"
    named_backup.mkdir()
    (named_backup / "SKILL.md").write_text(
        "---\nname: named-backup\n---\n", encoding="utf-8"
    )

    live = tmp_path / "live-skill"
    live.mkdir()
    (live / "SKILL.md").write_text("---\nname: live-skill\n---\n", encoding="utf-8")

    found = list(iter_skill_index_files(tmp_path, "SKILL.md"))

    assert found == [live / "SKILL.md"]


def test_is_excluded_skill_path_matches_bak_pattern():
    assert (
        is_excluded_skill_path(
            Path("skills/hermes-profile.bak-20260510_233500/SKILL.md")
        )
        is True
    )


def test_is_excluded_skill_path_does_not_exclude_live_skill():
    assert is_excluded_skill_path(Path("skills/hermes-profile/SKILL.md")) is False


def test_is_excluded_skill_path_matches_dot_backup_pattern():
    assert is_excluded_skill_path(Path("skills/.backup-old/nested/SKILL.md")) is True


def test_is_excluded_skill_path_allows_skill_with_bak_in_name():
    assert is_excluded_skill_path(Path("skills/bak-manager/SKILL.md")) is False


class TestSkillMatchesPlatformTermux:
    """Termux is Linux userland on Android."""

    def test_no_platforms_field_matches_everywhere(self):
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform({}) is True
            assert skill_matches_platform({"name": "foo"}) is True

    def test_linux_skill_loads_on_termux_android_platform(self):
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is True
            assert skill_matches_platform_list(fm["platforms"]) is True

    def test_linux_macos_windows_skill_loads_on_termux(self):
        fm = {"platforms": ["linux", "macos", "windows"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is True
            assert skill_matches_platform_list(fm["platforms"]) is True

    def test_linux_skill_loads_on_termux_linux_platform(self):
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "linux"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is True
            assert skill_matches_platform_list(fm["platforms"]) is True

    def test_macos_only_skill_still_excluded_on_termux(self):
        fm = {"platforms": ["macos"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is False
            assert skill_matches_platform_list(fm["platforms"]) is False

    def test_windows_only_skill_still_excluded_on_termux(self):
        fm = {"platforms": ["windows"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is False
            assert skill_matches_platform_list(fm["platforms"]) is False

    def test_explicit_termux_or_android_tag_matches(self):
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform({"platforms": ["termux"]}) is True
            assert skill_matches_platform({"platforms": ["android"]}) is True
            assert skill_matches_platform_list(["termux"]) is True
            assert skill_matches_platform_list(["android"]) is True

    def test_non_termux_android_does_not_widen(self):
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=False
        ):
            assert skill_matches_platform(fm) is False
            assert skill_matches_platform_list(fm["platforms"]) is False

    def test_linux_skill_on_real_linux_unaffected(self):
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "linux"), patch(
            "agent.skill_utils.is_termux", return_value=False
        ):
            assert skill_matches_platform(fm) is True
            assert skill_matches_platform_list(fm["platforms"]) is True

    def test_macos_skill_on_real_macos_unaffected(self):
        fm = {"platforms": ["macos"]}
        with patch("agent.skill_utils.sys.platform", "darwin"), patch(
            "agent.skill_utils.is_termux", return_value=False
        ):
            assert skill_matches_platform(fm) is True
            assert skill_matches_platform_list(fm["platforms"]) is True


class TestNormalizeSkillLookupName:
    def test_relative_path_unchanged(self, tmp_path, monkeypatch):
        from agent.skill_utils import normalize_skill_lookup_name

        # Relative identifiers early-return before any root lookup.
        assert normalize_skill_lookup_name("foo/bar") == "foo/bar"

    def test_absolute_under_skills_dir_becomes_relative(self, tmp_path, monkeypatch):
        from agent.skill_utils import normalize_skill_lookup_name

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "category" / "my-skill"
        skill_dir.mkdir(parents=True)
        # Patch the root skill_view() itself enforces — normalization reads
        # tools.skills_tool.SKILLS_DIR at call time so the two stay in sync.
        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
        assert normalize_skill_lookup_name(str(skill_dir)) == "category/my-skill"

    def test_absolute_via_symlink_uses_lexical_relative_path(self, tmp_path, monkeypatch):
        from agent.skill_utils import normalize_skill_lookup_name

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        external = tmp_path / "external" / "my-skill"
        external.mkdir(parents=True)
        link = skills_dir / "my-skill"
        try:
            link.symlink_to(external)
        except OSError:
            pytest.skip("Symlinks not supported")
        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
        assert normalize_skill_lookup_name(str(link)) == "my-skill"

    def test_untrusted_absolute_returned_unchanged(self, tmp_path, monkeypatch):
        from agent.skill_utils import normalize_skill_lookup_name

        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", tmp_path / "skills")
        monkeypatch.setattr("agent.skill_utils.get_skills_dir", lambda: tmp_path / "skills")
        outside = str(tmp_path / "outside" / "skill")
        assert normalize_skill_lookup_name(outside) == outside
