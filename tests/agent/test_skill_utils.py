"""Tests for agent/skill_utils.py."""

import os
import sys

import pytest

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


# ── iter_skill_index_files cycle detection (#18809) ───────────────────────


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="symlink cycles require POSIX semantics; Windows handling differs",
)
class TestIterSkillIndexFilesSymlinkCycles:
    """Regression tests for #18809 — ``os.walk(followlinks=True)`` does not
    detect symlink cycles on its own. ``iter_skill_index_files`` must guard
    against cyclic symlink trees so a malformed user skills directory does
    not hang skill discovery / agent startup."""

    def test_self_referencing_subdir_does_not_loop(self, tmp_path):
        """A subdirectory symlinked back to its ancestor used to recurse
        until the OS rejected the path. The function must now terminate."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        # A real skill so there is something to find.
        real_skill = skills_dir / "real-skill"
        real_skill.mkdir()
        (real_skill / "SKILL.md").write_text("# real skill")
        # Create a self-referencing cycle:
        #   skills/test-cycle/circular -> skills
        cycle_dir = skills_dir / "test-cycle"
        cycle_dir.mkdir()
        (cycle_dir / "circular").symlink_to(skills_dir, target_is_directory=True)

        # Without cycle detection this never returns.
        results = list(iter_skill_index_files(skills_dir, "SKILL.md"))

        # Real skill is still discovered.
        assert any(p.name == "SKILL.md" for p in results)
        # The cycle did not multiply the same SKILL.md.
        assert len(results) == 1

    def test_cycle_via_symlinked_sibling_does_not_loop(self, tmp_path):
        """Cycles can also form via two sibling directories cross-linking
        each other. The realpath-tracking guard should still catch this."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        a = skills_dir / "a"
        b = skills_dir / "b"
        a.mkdir()
        b.mkdir()
        (a / "SKILL.md").write_text("# a")
        (b / "SKILL.md").write_text("# b")
        # a/b_link -> b ; b/a_link -> a — creates a 2-cycle through symlinks.
        (a / "b_link").symlink_to(b, target_is_directory=True)
        (b / "a_link").symlink_to(a, target_is_directory=True)

        results = list(iter_skill_index_files(skills_dir, "SKILL.md"))

        # Both real SKILL.md files are still found, exactly once each.
        names = sorted(str(p.relative_to(skills_dir)) for p in results)
        assert names == [os.path.join("a", "SKILL.md"), os.path.join("b", "SKILL.md")]

    def test_normal_tree_unchanged(self, tmp_path):
        """Sanity: a plain skills tree without symlinks is unaffected."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        for name in ("alpha", "beta", "gamma"):
            (skills_dir / name).mkdir()
            (skills_dir / name / "SKILL.md").write_text(f"# {name}")

        results = list(iter_skill_index_files(skills_dir, "SKILL.md"))
        names = sorted(p.parent.name for p in results)
        assert names == ["alpha", "beta", "gamma"]
