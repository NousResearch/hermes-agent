"""Tests for agent/skill_utils.py."""

from unittest.mock import patch

from agent.skill_utils import (
    extract_skill_conditions,
    iter_skill_index_files,
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


# ── agentskills.io spec compliance: platforms/tags under metadata ───────────


def test_platforms_top_level_still_works():
    """Top-level platforms field (legacy) is still read correctly."""
    frontmatter = {"platforms": ["linux"]}
    result = skill_matches_platform(frontmatter)
    assert isinstance(result, bool)


def test_platforms_under_metadata_fallback():
    """agentskills.io format: platforms under metadata is also read."""
    frontmatter = {"metadata": {"platforms": ["linux"]}}
    result = skill_matches_platform(frontmatter)
    assert isinstance(result, bool)


def test_metadata_platforms_fallback_when_top_level_absent():
    """When top-level platforms is absent, metadata.platforms is used."""
    frontmatter = {"metadata": {"platforms": ["nonexistent-platform-xyz"]}}
    with patch("agent.skill_utils.sys.platform", "linux"):
        assert skill_matches_platform(frontmatter) is False


def test_metadata_tags_fallback_in_parse_frontmatter():
    """Tags under metadata: block are found by parse_frontmatter."""
    from agent.skill_utils import parse_frontmatter

    content = "---\nname: test-skill\ndescription: test\nmetadata:\n  tags: [ai, ml]\n---\nBody text."
    frontmatter, body = parse_frontmatter(content)
    assert frontmatter.get("metadata", {}).get("tags") == ["ai", "ml"]


def test_tags_top_level_still_read_by_skill_manage():
    """Top-level tags in SKILL.md are still returned by skill_manage."""
    from tools.skills_tool import _parse_tags

    assert _parse_tags("ai, ml") == ["ai", "ml"]
    assert _parse_tags("[ai, ml]") == ["ai", "ml"]
    assert _parse_tags("") == []


def test_tags_metadata_fallback():
    """_parse_tags handles metadata.tags fallback correctly."""
    from tools.skills_tool import _parse_tags

    assert _parse_tags("fine-tuning, llm") == ["fine-tuning", "llm"]
    assert _parse_tags("[fine-tuning, llm]") == ["fine-tuning", "llm"]
