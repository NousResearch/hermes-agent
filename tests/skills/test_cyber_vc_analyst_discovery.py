from pathlib import Path

from agent.skill_utils import get_all_skills_dirs


def test_bundled_repo_skills_dir_is_considered_for_discovery():
    dirs = get_all_skills_dirs()
    repo_skills_dir = Path(__file__).resolve().parents[2] / "skills"

    assert dirs
    assert any(path.resolve() == repo_skills_dir.resolve() for path in dirs)
