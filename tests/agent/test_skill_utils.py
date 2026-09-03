"""Tests for agent/skill_utils.py."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.skill_utils import (
    extract_skill_config_vars,
    extract_skill_conditions,
    get_disabled_skill_names,
    get_external_skills_dirs,
    is_excluded_skill_path,
    is_external_skill_path,
    is_skill_support_path,
    iter_skill_index_files,
    parse_config_string_list,
    parse_frontmatter,
    resolve_skill_config_values,
    skill_matches_platform,
    skill_matches_platform_list,
)












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
    assert resolve_skill_config_values([
        {"key": "wiki.path", "description": "Wiki path"}
    ])["wiki.path"].endswith("/wiki")
    assert parse_count == 1


class TestParseConfigStringList:
    """#86661: `hermes config set` and JSON-mode editor saves store lists as
    quoted strings (e.g. '["a","b"]'). Treating such a string as a single name
    made curated disabled lists silently filter nothing."""

    def test_json_array_string_parses(self):
        assert parse_config_string_list('["skill-a","skill-b"]') == [
            "skill-a",
            "skill-b",
        ]

    def test_python_literal_array_string_parses(self):
        # `hermes config set` can persist single-quoted Python-literal forms.
        assert parse_config_string_list("['skill-a']") == ["skill-a"]

    def test_scalar_string_means_one_name(self):
        # #13026: a scalar string still names a single entry.
        assert parse_config_string_list("skill-a") == ["skill-a"]

    def test_real_list_passes_through(self):
        assert parse_config_string_list(["skill-a", "skill-b"]) == [
            "skill-a",
            "skill-b",
        ]
        assert parse_config_string_list(("skill-a",)) == ["skill-a"]

    def test_none_returns_empty(self):
        assert parse_config_string_list(None) == []

    def test_malformed_json_falls_back_to_single_name(self):
        assert parse_config_string_list('["skill-a"') == ['["skill-a"']

    def test_empty_array_string_returns_empty(self):
        assert parse_config_string_list("[]") == []


class TestDisabledSkillsJsonArrayString:
    """The skills.disabled setting must honor a JSON-array string form, not
    treat the whole string as one dead skill name (#86661)."""

    def test_get_disabled_skill_names_parses_json_array_string(
        self, tmp_path, monkeypatch
    ):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "skills:\n  disabled: '[\"skill-a\",\"skill-b\"]'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from agent import skill_utils

        getattr(skill_utils, "_raw_config_cache_clear", lambda: None)()

        assert get_disabled_skill_names() == {"skill-a", "skill-b"}

    def test_get_disabled_skill_names_scalar_string_still_single_name(
        self, tmp_path, monkeypatch
    ):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "skills:\n  disabled: 'hidden-skill'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from agent import skill_utils

        getattr(skill_utils, "_raw_config_cache_clear", lambda: None)()

        assert get_disabled_skill_names() == {"hidden-skill"}






def _replace_preserving_size_and_mtime(path: Path, content: str) -> None:
    """Atomically replace *path* while preserving its weak cache metadata."""
    original_stat = path.stat()
    assert len(content.encode()) == original_stat.st_size
    replacement = path.with_suffix(".tmp")
    replacement.write_text(content, encoding="utf-8")
    replacement.replace(path)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    replaced_stat = path.stat()
    assert (replaced_stat.st_size, replaced_stat.st_mtime_ns) == (
        original_stat.st_size,
        original_stat.st_mtime_ns,
    )
    assert (replaced_stat.st_dev, replaced_stat.st_ino) != (
        original_stat.st_dev,
        original_stat.st_ino,
    )


def test_skill_config_cache_invalidates_after_same_stat_replacement(
    tmp_path, monkeypatch
):
    """Replacing config content must invalidate every skill config consumer."""
    from agent import skill_utils

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    external_a = tmp_path / "skills-a"
    external_b = tmp_path / "skills-b"
    external_a.mkdir()
    external_b.mkdir()
    config_path = hermes_home / "config.yaml"
    before = (
        "skills:\n"
        "  disabled: [skill-a]\n"
        f"  external_dirs: [{external_a}]\n"
        "  config:\n"
        "    demo:\n"
        "      value: value-a\n"
    )
    after = (
        "skills:\n"
        "  disabled: [skill-b]\n"
        f"  external_dirs: [{external_b}]\n"
        "  config:\n"
        "    demo:\n"
        "      value: value-b\n"
    )
    assert len(before.encode()) == len(after.encode())
    config_path.write_text(before, encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    skill_utils._external_dirs_cache_clear()
    assert get_disabled_skill_names() == {"skill-a"}
    assert get_external_skills_dirs() == [external_a.resolve()]
    assert resolve_skill_config_values([{"key": "demo.value"}]) == {
        "demo.value": "value-a"
    }
    _replace_preserving_size_and_mtime(config_path, after)

    assert get_disabled_skill_names() == {"skill-b"}
    assert get_external_skills_dirs() == [external_b.resolve()]
    assert resolve_skill_config_values([{"key": "demo.value"}]) == {
        "demo.value": "value-b"
    }


def test_skill_config_cache_invalidates_with_future_coarse_mtime(
    tmp_path, monkeypatch
):
    """Invalidation must not depend on comparing mtime with wall-clock time."""
    from agent import skill_utils

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    before = "skills:\n  disabled: [skill-a]\n"
    after = "skills:\n  disabled: [skill-b]\n"
    assert len(before) == len(after)
    config_path.write_text(before, encoding="utf-8")
    initial_stat = config_path.stat()
    two_seconds_ns = 2_000_000_000
    future_mtime_ns = (
        (initial_stat.st_mtime_ns + 86_400_000_000_000) // two_seconds_ns
    ) * two_seconds_ns
    os.utime(config_path, ns=(initial_stat.st_atime_ns, future_mtime_ns))

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    skill_utils._raw_config_cache_clear()
    assert get_disabled_skill_names() == {"skill-a"}
    _replace_preserving_size_and_mtime(config_path, after)

    assert get_disabled_skill_names() == {"skill-b"}


def test_skill_config_cache_stable_file_avoids_repeated_reads(tmp_path, monkeypatch):
    """An unchanged config keeps the stat-only fast path after the first read."""
    from agent import skill_utils

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text("skills:\n  disabled: [skill-a]\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    skill_utils._raw_config_cache_clear()
    assert get_disabled_skill_names() == {"skill-a"}

    with patch.object(
        Path,
        "read_text",
        side_effect=AssertionError("stable config should not be read again"),
    ):
        assert get_disabled_skill_names() == {"skill-a"}
        assert resolve_skill_config_values([]) == {}


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


def test_skill_support_path_uses_explicit_discovery_root_not_cwd(tmp_path, monkeypatch):
    discovery_root = tmp_path / "site-packages" / "skills"
    umbrella = discovery_root / "category" / "umbrella"
    nested = umbrella / "references" / "archived" / "SKILL.md"
    nested.parent.mkdir(parents=True)
    (umbrella / "SKILL.md").write_text("---\nname: umbrella\n---\n", encoding="utf-8")
    nested.write_text("---\nname: archived\n---\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    relative = nested.relative_to(discovery_root)
    assert is_skill_support_path(relative, root=discovery_root) is True
    assert is_excluded_skill_path(relative, root=discovery_root) is True


# ── skill_matches_platform on Termux ──────────────────────────────────────


class TestSkillMatchesPlatformTermux:
    """Termux is Linux userland on Android. Skills tagged platforms:[linux]
    must load there regardless of whether Python reports sys.platform as
    "linux" (pre-3.13) or "android" (3.13+). Reported by user @LikiusInik
    in May 2026 — only 3 built-in skills appeared on Termux because every
    github/productivity/mlops skill is tagged platforms:[linux,macos,windows]
    and sys.platform=="android" did not start with "linux".
    """

    def test_no_platforms_field_matches_everywhere(self):
        # Backward-compat default — skills without a platforms tag load
        # on any OS, Termux included.
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform({}) is True
            assert skill_matches_platform({"name": "foo"}) is True







    def test_non_termux_android_does_not_widen(self):
        # If we're somehow on a plain Android Python (not Termux), don't
        # silently load Linux skills — Termux is the supported environment.
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=False
        ):
            assert skill_matches_platform(fm) is False
            assert skill_matches_platform_list(fm["platforms"]) is False

    def test_linux_skill_on_real_linux_unaffected(self):
        # The non-Termux Linux path must not change.
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "linux"), patch(
            "agent.skill_utils.is_termux", return_value=False
        ):
            assert skill_matches_platform(fm) is True
            assert skill_matches_platform_list(fm["platforms"]) is True



class TestNormalizeSkillLookupName:
    def test_relative_path_unchanged(self, tmp_path, monkeypatch):
        from agent.skill_utils import normalize_skill_lookup_name

        # Relative identifiers early-return before any root lookup.
        assert normalize_skill_lookup_name("foo/bar") == "foo/bar"


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



# ── parse_frontmatter: UTF-8 BOM tolerance ─────────────────────────────────


class TestParseFrontmatterBOM:
    """A UTF-8 BOM (U+FEFF) on a Windows-saved SKILL.md must not defeat
    frontmatter parsing.

    Notepad and PowerShell ``>`` prepend a BOM when saving UTF-8;
    ``read_text(encoding="utf-8")`` (what ``_parse_skill_file`` uses) keeps
    it, so the bytes handed to ``parse_frontmatter`` start with a BOM ahead of
    the ``---`` fence. Before the fix the ``startswith("---")`` check returned
    False and the whole frontmatter was silently dropped — the skill loaded
    nameless, platform gating fell open, and env-var/config setup never fired.
    """

    SKILL = (
        "---\n"
        "name: my-skill\n"
        "description: Does a thing.\n"
        "platforms: [macos]\n"
        "metadata:\n"
        "  hermes:\n"
        "    config:\n"
        "      - key: my.key\n"
        "        description: A configured value\n"
        "---\n\n"
        "# My Skill\n\nBody text.\n"
    )

    def test_bom_frontmatter_matches_plain(self):
        plain_fm, plain_body = parse_frontmatter(self.SKILL)
        bom_fm, bom_body = parse_frontmatter("\ufeff" + self.SKILL)
        assert bom_fm == plain_fm
        assert bom_body == plain_body
        assert bom_fm["name"] == "my-skill"
        assert bom_fm["description"] == "Does a thing."




    def test_bom_platform_gating_regression(self):
        # The concrete harm: a macOS-only skill must be gated identically
        # whether or not the file carries a BOM. Empty frontmatter (the bug)
        # reads as "no platform restriction" and leaks the skill everywhere,
        # i.e. it would answer True on every host. Compare against the real
        # host's verdict instead of faking Windows — the fake only stood in
        # for "some non-macOS host", which the CI host already is.
        import sys

        expected = sys.platform == "darwin"
        with patch("agent.skill_utils.is_termux", return_value=False):
            plain_fm, _ = parse_frontmatter(self.SKILL)
            bom_fm, _ = parse_frontmatter("\ufeff" + self.SKILL)
            assert skill_matches_platform(plain_fm) is expected
            assert skill_matches_platform(bom_fm) is expected


    def test_real_file_read_path(self, tmp_path):
        # End-to-end: write the file the way a Windows editor does (utf-8-sig
        # emits a BOM), read it the way _parse_skill_file does (plain utf-8),
        # and confirm the frontmatter survives the round trip.
        f = tmp_path / "SKILL.md"
        f.write_text(self.SKILL, encoding="utf-8-sig")
        raw = f.read_text(encoding="utf-8")
        assert raw.startswith("\ufeff")  # BOM really is present on disk
        fm, _ = parse_frontmatter(raw)
        assert fm["name"] == "my-skill"
        assert fm["platforms"] == ["macos"]


class TestBOMToleranceSiblingSites:
    """The BOM fix must cover every independent frontmatter parser, not just
    the canonical ``parse_frontmatter`` — several modules reimplement the
    ``---`` fence check locally."""

    SKILL = "---\nname: bom-skill\ndescription: Saved by Notepad\n---\n\n# Body\n"


    def test_prompt_builder_strips_bom_frontmatter(self):
        # A BOM'd context file (AGENTS.md etc.) must not leak raw
        # frontmatter into the system prompt.
        from agent.prompt_builder import _strip_yaml_frontmatter

        out = _strip_yaml_frontmatter("\ufeff---\nfoo: bar\n---\nBody text\n")
        assert out.strip() == "Body text"

    def test_blueprints_split_frontmatter_bom(self):
        # str.lstrip() does NOT strip U+FEFF (it is not whitespace), so the
        # pre-existing lstrip() in _split_frontmatter never covered it.
        from tools.blueprints import _split_frontmatter

        fm = _split_frontmatter("\ufeff---\nname: bp\n---\nbody")
        assert fm is not None
        assert fm.get("name") == "bp"

