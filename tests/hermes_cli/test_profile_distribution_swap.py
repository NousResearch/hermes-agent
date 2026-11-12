"""Force-install swap failures: a copy or swap that fails part-way keeps the old tree."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hermes_agent.hermes_cli.profile_distribution import install_distribution
from tests.hermes_cli import test_profile_distribution as _base

_make_staging_dir = _base._make_staging_dir
profile_env = _base.profile_env


def test_force_install_preserves_existing_dir_when_new_copy_fails(
    profile_env, monkeypatch
):
    staged = _make_staging_dir(profile_env, "src")
    plan = install_distribution(str(staged), name="target")
    skill = plan.target_dir / "skills" / "demo" / "SKILL.md"
    skill.write_text("old skill\n")
    (staged / "skills" / "demo" / "SKILL.md").write_text("new skill\n")
    original_copytree = shutil.copytree

    def fail_skill_root_copy(src, dst, *args, **kwargs):
        if Path(src).name == "demo":
            Path(dst).mkdir(
                parents=True, exist_ok=True
            )  # a partial copy, then the lock bites
            (Path(dst) / "SKILL.md").write_text("partial\n")
            raise OSError("[WinError 145] The directory is not empty")
        return original_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(
        "hermes_agent.hermes_cli.profile_distribution.shutil.copytree", fail_skill_root_copy
    )

    with pytest.raises(OSError, match="WinError 145"):
        install_distribution(str(staged), name="target", force=True)

    assert skill.read_text() == "old skill\n"
    assert not list((plan.target_dir / "skills").glob(".*hermes-dist*"))


def test_force_install_restores_existing_dir_when_staged_swap_fails(
    profile_env, monkeypatch
):
    staged = _make_staging_dir(profile_env, "src")
    plan = install_distribution(str(staged), name="target")
    skill = plan.target_dir / "skills" / "demo" / "SKILL.md"
    skill.write_text("old skill\n")
    (staged / "skills" / "demo" / "SKILL.md").write_text("new skill\n")
    original_rename = Path.rename

    def fail_staged_swap(path, target):
        if "demo" in path.name and "staging-" in path.name:
            raise OSError("staged swap failed")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staged_swap)

    with pytest.raises(OSError, match="staged swap failed"):
        install_distribution(str(staged), name="target", force=True)

    assert skill.read_text() == "old skill\n"
    assert not list((plan.target_dir / "skills").glob(".*hermes-dist*"))


def test_force_install_locked_old_tree_is_not_discovered(profile_env, monkeypatch):
    from hermes_agent.agent.skill_utils import is_excluded_skill_path, iter_skill_index_files
    import hermes_agent.hermes_cli.profile_distribution as pd

    staged = _make_staging_dir(profile_env, "src")
    plan = install_distribution(str(staged), name="target")
    skills = plan.target_dir / "skills"
    (skills / "demo" / "SKILL.md").write_text("old skill\n")
    (staged / "skills" / "demo" / "SKILL.md").write_text("new skill\n")
    original_remove = pd._remove_existing

    def locked_backup(path):
        if "demo" in path.name and "old-" in path.name:
            raise PermissionError("[WinError 32] in use")
        return original_remove(path)

    monkeypatch.setattr(pd, "_remove_existing", locked_backup)

    install_distribution(str(staged), name="target", force=True)

    assert "old skill\n" in [
        p.read_text() for p in skills.rglob("SKILL.md")
    ]  # the leftover is on disk
    walked = [p.read_text() for p in iter_skill_index_files(skills, "SKILL.md")]
    globbed = [
        p.read_text() for p in skills.rglob("SKILL.md") if not is_excluded_skill_path(p)
    ]
    for found in (walked, globbed):
        assert "new skill\n" in found
        assert "old skill\n" not in found
