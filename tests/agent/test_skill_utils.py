"""Tests for agent/skill_utils.py."""

import sys
from pathlib import Path, PureWindowsPath
from unittest.mock import patch

import pytest

from agent.skill_utils import (
    _lexical_relative_path,
    extract_skill_config_vars,
    extract_skill_conditions,
    get_disabled_skill_names,
    get_external_skills_dirs,
    is_excluded_skill_path,
    is_external_skill_path,
    is_skill_support_path,
    iter_skill_index_files,
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


def test_pre_edit_snapshot_directories_are_excluded_without_overmatching(tmp_path):
    """Only the explicit pre-edit snapshot artifact convention is excluded."""
    snapshot = tmp_path / "category" / "kanban-lifecycle-pre-edit-snapshot-t_0bd96003" / "SKILL.md"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("---\nname: kanban-lifecycle\n---\n", encoding="utf-8")

    top_level_snapshot = tmp_path / "hermes-kanban-pre-edit-snapshot-t_0bd96003" / "SKILL.md"
    top_level_snapshot.parent.mkdir()
    top_level_snapshot.write_text("---\nname: kanban-lifecycle\n---\n", encoding="utf-8")

    legitimate = tmp_path / "snapshot" / "pre-edit" / "skill" / "SKILL.md"
    legitimate.parent.mkdir(parents=True)
    legitimate.write_text("---\nname: legitimate\n---\n", encoding="utf-8")

    assert is_excluded_skill_path(snapshot) is True
    assert is_excluded_skill_path(top_level_snapshot) is True
    assert is_excluded_skill_path(legitimate) is False
    assert list(iter_skill_index_files(tmp_path, "SKILL.md")) == [legitimate]


def test_pre_edit_snapshot_name_requires_canonical_task_id(tmp_path):
    legitimate = tmp_path / "photo-pre-edit-snapshot-helper" / "SKILL.md"
    legitimate.parent.mkdir(parents=True)
    legitimate.write_text("---\nname: photo\n---\n", encoding="utf-8")

    assert is_excluded_skill_path(legitimate) is False


@pytest.mark.parametrize(
    ("task_suffix", "excluded"),
    [
        ("abcdef1", False),
        ("abcdef12", True),
        ("abcdef123", False),
        ("ABCDEF12", False),
        ("abcdef1g", False),
    ],
)
def test_pre_edit_snapshot_task_suffix_is_exactly_lowercase_eight_hex(
    tmp_path, task_suffix, excluded
):
    """Only canonical generated task IDs hide directories and legacy .md files."""
    directory = tmp_path / f"ghost-pre-edit-snapshot-t_{task_suffix}"
    directory.mkdir()
    directory_skill = directory / "SKILL.md"
    directory_skill.write_text("---\nname: directory\n---\n", encoding="utf-8")
    legacy_skill = tmp_path / f"ghost-pre-edit-snapshot-t_{task_suffix}.md"
    legacy_skill.write_text("---\nname: legacy\n---\n", encoding="utf-8")

    assert is_excluded_skill_path(directory_skill) is excluded
    assert is_excluded_skill_path(legacy_skill, root=tmp_path) is excluded


def test_outside_explicit_root_does_not_inherit_ancestor_exclusions(tmp_path):
    """Outside-root paths are not classified using unrelated parent segments."""
    root = tmp_path / "configured-root"
    root.mkdir()

    outside_site_packages = tmp_path / "site-packages" / "live" / "SKILL.md"
    outside_site_packages.parent.mkdir(parents=True)
    outside_site_packages.write_text("---\nname: live\n---\n", encoding="utf-8")

    outside_support = tmp_path / "ancestor-skill" / "references" / "live" / "SKILL.md"
    outside_support.parent.mkdir(parents=True)
    (outside_support.parents[2] / "SKILL.md").write_text(
        "---\nname: ancestor\n---\n", encoding="utf-8"
    )
    outside_support.write_text("---\nname: live\n---\n", encoding="utf-8")

    outside_snapshot = tmp_path / "ghost-pre-edit-snapshot-t_abcdef12" / "SKILL.md"
    outside_snapshot.parent.mkdir()
    outside_snapshot.write_text("---\nname: ghost\n---\n", encoding="utf-8")

    relative_site_packages = Path("..") / "site-packages" / "live" / "SKILL.md"
    assert is_excluded_skill_path(outside_site_packages, root=root) is False
    assert is_skill_support_path(outside_site_packages, root=root) is False
    assert is_excluded_skill_path(outside_support, root=root) is False
    assert is_skill_support_path(outside_support, root=root) is False
    assert is_excluded_skill_path(outside_snapshot, root=root) is False
    assert is_skill_support_path(outside_snapshot, root=root) is False
    assert is_excluded_skill_path(relative_site_packages, root=root) is False
    assert is_skill_support_path(relative_site_packages, root=root) is False


@pytest.mark.parametrize(
    ("raw_path", "root", "outside"),
    [
        (r"C:/configured-root/live/SKILL.md", r"C:/configured-root", False),
        (r"C:/configured-root/./site-packages/live/SKILL.md", r"C:/configured-root", False),
        (r"C:/other/site-packages/live/SKILL.md", r"C:/configured-root", True),
        (r"D:/site-packages/live/SKILL.md", r"C:/configured-root", True),
        (r"D:site-packages/live/SKILL.md", r"C:/configured-root", True),
        (r"C:../site-packages/live/SKILL.md", r"C:/configured-root", True),
        (r"/site-packages/live/SKILL.md", r"C:/configured-root", True),
        (r"../site-packages/live/SKILL.md", r"C:/configured-root", True),
        (r"./site-packages/live/SKILL.md", r"C:/configured-root", False),
        (r"\\server\share\configured-root/live/SKILL.md", r"C:/configured-root", True),
        (r"\\?\C:\configured-root/live/SKILL.md", r"C:/configured-root", True),
        (
            r"\\server\share\configured-root/live/SKILL.md",
            r"\\server\share\configured-root",
            False,
        ),
    ],
)
def test_windows_lexical_path_matrix_is_deterministic(raw_path, root, outside):
    """Windows anchors and relative forms never inherit the wrong root."""
    result = _lexical_relative_path(
        PureWindowsPath(raw_path), root=PureWindowsPath(root)
    )
    assert result.outside_root is outside


def test_windows_lexical_exclusions_use_only_anchorless_relative_paths():
    root = PureWindowsPath("C:/configured-root")
    inside_site_packages = PureWindowsPath(
        "C:/configured-root/site-packages/live/SKILL.md"
    )
    for outside_path in (
        PureWindowsPath(r"D:site-packages/live/SKILL.md"),
        PureWindowsPath(r"C:../site-packages/live/SKILL.md"),
        PureWindowsPath(r"/site-packages/live/SKILL.md"),
        PureWindowsPath(r"../site-packages/live/SKILL.md"),
    ):
        assert is_excluded_skill_path(outside_path, root=root) is False
    assert is_excluded_skill_path(inside_site_packages, root=root) is True


def test_pure_windows_support_paths_do_not_probe_host_filesystem(monkeypatch):
    """Synthetic Windows paths on POSIX remain lexical-only."""
    calls = []
    monkeypatch.setattr(
        "agent.skill_utils._support_manifest_exists",
        lambda path: calls.append(path) or True,
    )
    nested = PureWindowsPath(
        "C:/configured-root/umbrella/references/archived/SKILL.md"
    )
    assert is_skill_support_path(
        nested, root=PureWindowsPath("C:/configured-root")
    ) is False
    assert calls == []


def test_support_manifest_probe_can_be_injected_platform_independently(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    nested = root / "umbrella" / "references" / "archived" / "SKILL.md"
    nested.parent.mkdir(parents=True)
    probe_calls = []
    monkeypatch.setattr(
        "agent.skill_utils._support_manifest_exists",
        lambda path: probe_calls.append(path) or path == root / "umbrella" / "SKILL.md",
    )

    assert is_skill_support_path(nested, root=root) is True
    assert probe_calls == [root / "umbrella" / "SKILL.md"]


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows paths")
def test_native_windows_support_root_probe_uses_concrete_paths(tmp_path):
    root = tmp_path / "skills"
    umbrella = root / "umbrella"
    nested = umbrella / "references" / "archived" / "SKILL.md"
    umbrella.mkdir(parents=True)
    nested.parent.mkdir()
    (umbrella / "SKILL.md").write_text("---\nname: umbrella\n---\n", encoding="utf-8")
    nested.write_text("---\nname: archived\n---\n", encoding="utf-8")

    assert is_skill_support_path(nested, root=root) is True
    assert is_excluded_skill_path(nested, root=root) is True


def test_exclusions_are_relative_to_discovery_root(tmp_path):
    """A snapshot-shaped profile/home parent must not hide its live skills."""
    profile_home = tmp_path / "worker-pre-edit-snapshot-t_abcdef12"
    skills_root = profile_home / "skills"
    live = skills_root / "live-skill" / "SKILL.md"
    live.parent.mkdir(parents=True)
    live.write_text("---\nname: live-skill\n---\n", encoding="utf-8")

    assert is_excluded_skill_path(live, root=skills_root) is False
    assert is_excluded_skill_path(live) is True
    outside = tmp_path / "outside" / "live-skill" / "SKILL.md"
    assert is_excluded_skill_path(outside, root=skills_root) is False


def test_legacy_snapshot_flat_files_use_only_recognized_skill_suffix(tmp_path):
    snapshot_stem = "ghost-pre-edit-snapshot-t_abcdef12"
    snapshot_md = tmp_path / f"{snapshot_stem}.md"
    snapshot_txt = tmp_path / f"{snapshot_stem}.txt"
    snapshot_md.write_text("stale", encoding="utf-8")
    snapshot_txt.write_text("not a recognized legacy skill file", encoding="utf-8")

    assert is_excluded_skill_path(snapshot_md, root=tmp_path) is True
    assert is_excluded_skill_path(snapshot_txt, root=tmp_path) is False
    assert list(iter_skill_index_files(tmp_path, snapshot_md.name)) == []


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
        # The concrete harm: a macOS-only skill must stay hidden on non-macOS
        # whether or not the file carries a BOM. Empty frontmatter (the bug)
        # reads as "no platform restriction" and leaks the skill everywhere.
        with patch("agent.skill_utils.sys.platform", "win32"), patch(
            "agent.skill_utils.is_termux", return_value=False
        ):
            plain_fm, _ = parse_frontmatter(self.SKILL)
            bom_fm, _ = parse_frontmatter("\ufeff" + self.SKILL)
            assert skill_matches_platform(plain_fm) is False
            assert skill_matches_platform(bom_fm) is False


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

