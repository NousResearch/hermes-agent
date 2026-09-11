from pathlib import Path

from agent import skill_utils


def create_skill(root: Path, org: str, name: str) -> Path:
    path = root / "_wisdom" / org / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: Test.\n---\n# {name}\n", encoding="utf-8"
    )
    return path


def test_no_verified_marker_means_no_managed_wisdom_skills(tmp_path: Path):
    create_skill(tmp_path, "org-a", "shared-a")
    assert list(skill_utils.iter_skill_index_files(tmp_path, "SKILL.md")) == []


def test_only_last_verified_org_managed_tree_loads(tmp_path: Path):
    first = create_skill(tmp_path, "org-a", "shared-a")
    second = create_skill(tmp_path, "org-b", "shared-b")
    marker = tmp_path / "_wisdom" / ".active_org"
    marker.write_text("org-a\n", encoding="utf-8")
    assert list(skill_utils.iter_skill_index_files(tmp_path, "SKILL.md")) == [first]
    marker.write_text("org-b\n", encoding="utf-8")
    assert list(skill_utils.iter_skill_index_files(tmp_path, "SKILL.md")) == [second]


def test_wisdom_managed_path_is_distinct_from_org_mirror(tmp_path: Path):
    managed = create_skill(tmp_path, "org-a", "shared-a")
    assert skill_utils.is_wisdom_managed_path(managed, tmp_path)
    assert not skill_utils.is_org_mirror_path(managed, tmp_path)
