"""Tests for agent/skill_utils.py — extract_skill_conditions metadata handling."""

from pathlib import Path

from agent.skill_utils import extract_skill_conditions, iter_skill_index_files


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


# ── iter_skill_index_files tests ──────────────────────────────────────────


class TestIterSkillIndexFiles:
    """Tests for iter_skill_index_files symlink cycle protection."""

    def test_finds_files_normally(self, tmp_path):
        """Baseline: finds SKILL.md in a flat directory."""
        skill = tmp_path / "my-skill" / "SKILL.md"
        skill.parent.mkdir()
        skill.write_text("---\nname: my-skill\n---\n")
        results = list(iter_skill_index_files(tmp_path, "SKILL.md"))
        assert len(results) == 1
        assert results[0] == skill

    def test_cyclic_symlink_does_not_loop(self, tmp_path):
        """A → B → A symlink cycle must not cause an infinite loop."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        # Create cyclic symlinks: a/link -> b, b/link -> a
        (dir_a / "link").symlink_to(dir_b)
        (dir_b / "link").symlink_to(dir_a)

        # Place a SKILL.md in dir_a so there's something to find
        (dir_a / "SKILL.md").write_text("---\nname: a\n---\n")

        # This must terminate (not hang)
        results = list(iter_skill_index_files(tmp_path, "SKILL.md"))
        assert len(results) == 1
        # Use resolve() — os.walk traversal order varies by platform,
        # so the reported path may be a/SKILL.md or b/link/SKILL.md
        # (both resolve to the same file).
        assert results[0].resolve() == (dir_a / "SKILL.md").resolve()

    def test_non_cyclic_symlink_still_followed(self, tmp_path):
        """Non-cyclic symlinks should still be followed and yield files."""
        real = tmp_path / "real-skills"
        real.mkdir()
        skill = real / "linked-skill" / "SKILL.md"
        skill.parent.mkdir()
        skill.write_text("---\nname: linked\n---\n")

        # Create a symlink that points to real-skills (no cycle)
        link = tmp_path / "skills-link"
        link.symlink_to(real)

        results = list(iter_skill_index_files(link, "SKILL.md"))
        assert len(results) == 1
        assert results[0] == link / "linked-skill" / "SKILL.md"

    def test_self_referencing_symlink(self, tmp_path):
        """A directory that symlinks to itself must not loop."""
        d = tmp_path / "self-ref"
        d.mkdir()
        (d / "loop").symlink_to(d)
        (d / "SKILL.md").write_text("---\nname: self\n---\n")

        results = list(iter_skill_index_files(tmp_path, "SKILL.md"))
        assert len(results) == 1
        assert results[0] == d / "SKILL.md"

    def test_excluded_dirs_still_skipped(self, tmp_path):
        """Excluded dirs (.git, .github, etc.) are still skipped."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "SKILL.md").write_text("---\nname: git\n---\n")

        normal = tmp_path / "real" / "SKILL.md"
        normal.parent.mkdir()
        normal.write_text("---\nname: real\n---\n")

        results = list(iter_skill_index_files(tmp_path, "SKILL.md"))
        assert len(results) == 1
        assert results[0] == normal

    def test_diamond_symlink_pattern(self, tmp_path):
        """Diamond pattern: A→C, B→C. Both symlinks should resolve without duplication."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_c = tmp_path / "c"
        dir_a.mkdir()
        dir_b.mkdir()
        dir_c.mkdir()

        (dir_a / "link").symlink_to(dir_c)
        (dir_b / "link").symlink_to(dir_c)
        (dir_c / "SKILL.md").write_text("---\nname: c\n---\n")

        results = list(iter_skill_index_files(tmp_path, "SKILL.md"))
        # The skill should be found via both paths but realpath dedup keeps it to 1
        assert len(results) == 1
