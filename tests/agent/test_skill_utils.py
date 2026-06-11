"""Tests for agent/skill_utils.py."""

from unittest.mock import patch

from agent.skill_utils import (
    extract_skill_config_vars,
    extract_skill_conditions,
    iter_skill_index_files,
    parse_frontmatter,
    skill_matches_platform,
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
    """Bug case: metadata is a non-dict truthy value (e.g. a YAML string)."""
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

    def test_linux_skill_loads_on_termux_android_platform(self):
        # Python 3.13+ on Termux reports sys.platform == "android".
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is True

    def test_linux_macos_windows_skill_loads_on_termux(self):
        # The common "[linux, macos, windows]" tag used by github-*,
        # productivity, mlops, etc.
        fm = {"platforms": ["linux", "macos", "windows"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is True

    def test_linux_skill_loads_on_termux_linux_platform(self):
        # Pre-3.13 Termux reports sys.platform == "linux" already — this
        # works without the Termux escape hatch but must still pass.
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "linux"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is True

    def test_macos_only_skill_still_excluded_on_termux(self):
        # macOS-only skills (apple-notes, imessage, ...) should NOT load
        # on Termux. The Termux fallback only widens platforms:[linux,...].
        fm = {"platforms": ["macos"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is False

    def test_windows_only_skill_still_excluded_on_termux(self):
        fm = {"platforms": ["windows"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is False

    def test_explicit_termux_or_android_tag_matches(self):
        # Skills can also opt in explicitly via platforms:[termux] or
        # platforms:[android] — both should match a Termux session.
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform({"platforms": ["termux"]}) is True
            assert skill_matches_platform({"platforms": ["android"]}) is True

    def test_non_termux_android_does_not_widen(self):
        # If we're somehow on a plain Android Python (not Termux), don't
        # silently load Linux skills — Termux is the supported environment.
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=False
        ):
            assert skill_matches_platform(fm) is False

    def test_linux_skill_on_real_linux_unaffected(self):
        # The non-Termux Linux path must not change.
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "linux"), patch(
            "agent.skill_utils.is_termux", return_value=False
        ):
            assert skill_matches_platform(fm) is True

    def test_macos_skill_on_real_macos_unaffected(self):
        fm = {"platforms": ["macos"]}
        with patch("agent.skill_utils.sys.platform", "darwin"), patch(
            "agent.skill_utils.is_termux", return_value=False
        ):
            assert skill_matches_platform(fm) is True


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

    def test_bom_body_has_no_leading_marker(self):
        _, body = parse_frontmatter("\ufeff" + self.SKILL)
        assert not body.startswith("\ufeff")
        assert body.lstrip().startswith("# My Skill")

    def test_bom_without_frontmatter_strips_marker(self):
        # A BOM'd file with no frontmatter still gets the invisible marker
        # removed from the body so it never reaches the system prompt.
        fm, body = parse_frontmatter("\ufeff# Heading\nText.\n")
        assert fm == {}
        assert body == "# Heading\nText.\n"

    def test_interior_bom_is_preserved(self):
        # Only the leading marker is stripped; a U+FEFF in the body is data.
        fm, body = parse_frontmatter("\ufeff---\nname: x\n---\nbo\ufeffdy\n")
        assert fm["name"] == "x"
        assert "\ufeff" in body

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

    def test_bom_config_vars_preserved(self):
        # metadata.hermes.config drives secure setup-on-load; it must survive
        # a BOM so Windows users still get prompted for the value.
        bom_fm, _ = parse_frontmatter("\ufeff" + self.SKILL)
        assert [v["key"] for v in extract_skill_config_vars(bom_fm)] == ["my.key"]

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
