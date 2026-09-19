"""Symlink-loop safety of the skills scanners (issue #104316).

``os.walk(..., followlinks=True)`` follows a symlink whose target is the walk
root itself or an ancestor of it, so a self-referential link (``ln -sf`` against
an existing symlink-to-directory dereferences and nests ``foo/foo -> foo``)
renders the same skill once per loop level in the system prompt index —
hundreds of duplicate entries for one skill, silently. Loop-forming links must
be pruned; links pointing outside the tree stay walkable.
"""

import os

import pytest

from agent.prompt_builder import _build_skills_manifest
from agent.skill_utils import iter_skill_index_files


def _write_skill(skill_dir):
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo skill\n---\n\n# Demo\n", encoding="utf-8")


@pytest.fixture
def looping_skills_dir(tmp_path):
    """A skill with a self-referential symlink (``demo/demo -> demo``)."""
    skills = tmp_path / "skills"
    demo = skills / "creative" / "demo"
    _write_skill(demo)
    try:
        (demo / "demo").symlink_to(demo, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")
    return skills


def test_iter_skill_index_files_ignores_self_referential_symlink(looping_skills_dir):
    found = list(iter_skill_index_files(looping_skills_dir, "SKILL.md"))

    assert found == [looping_skills_dir / "creative" / "demo" / "SKILL.md"]


def test_skills_manifest_ignores_self_referential_symlink(looping_skills_dir):
    manifest = _build_skills_manifest(looping_skills_dir)

    assert list(manifest) == ["creative/demo/SKILL.md"]


def test_iter_skill_index_files_ignores_indirect_symlink_cycle(tmp_path):
    """A -> B -> A: neither link points at an ancestor of the walk root
    directly, but resolving both lands inside the tree — still a cycle."""
    skills = tmp_path / "skills"
    a = skills / "a-skill"
    b = skills / "b-skill"
    _write_skill(a)
    _write_skill(b)
    try:
        (a / "loop").symlink_to(b, target_is_directory=True)
        (b / "loop").symlink_to(a, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")

    found = list(iter_skill_index_files(skills, "SKILL.md"))

    assert sorted(found) == [a / "SKILL.md", b / "SKILL.md"]


def test_iter_skill_index_files_keeps_symlink_outside_tree(tmp_path):
    """A symlink to a skill dir OUTSIDE the walk root is a legit layout —
    it must still be scanned (exactly once, no loop exists)."""
    skills = tmp_path / "skills"
    external = tmp_path / "elsewhere" / "linked-skill"
    _write_skill(external)
    skills.mkdir(parents=True)
    try:
        (skills / "linked").symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")

    found = list(iter_skill_index_files(skills, "SKILL.md"))

    assert [p.name for p in found] == ["SKILL.md"]
    assert os.path.islink(found[0].parent)
