"""Regression test for #126612: ``repair-official --restore`` backups land under
``skills/.restore-backups/`` and must not be discoverable as active Skills."""

from pathlib import Path

from agent.skill_utils import is_excluded_skill_path, iter_skill_index_files


def _seed(skills: Path) -> None:
    (skills / "pytorch-fsdp").mkdir(parents=True)
    (skills / "pytorch-fsdp" / "SKILL.md").write_text(
        "---\nname: pytorch-fsdp\ndescription: active official copy\n---\nbody\n",
        encoding="utf-8",
    )
    backup = skills / ".restore-backups" / "official-optional-20260929T000000Z" / "pytorch-fsdp"
    backup.mkdir(parents=True)
    (backup / "SKILL.md").write_text(
        "---\nname: pytorch-fsdp\ndescription: backup copy\n---\nbody\n",
        encoding="utf-8",
    )


def test_restore_backup_is_excluded_from_skill_paths(tmp_path):
    backup_md = (
        tmp_path / ".restore-backups" / "official-optional-x" / "pytorch-fsdp" / "SKILL.md"
    )
    assert is_excluded_skill_path(backup_md, root=tmp_path)
    assert not is_excluded_skill_path(tmp_path / "pytorch-fsdp" / "SKILL.md", root=tmp_path)


def test_restore_backup_does_not_shadow_active_skill(tmp_path):
    _seed(tmp_path)
    found = [p for p in iter_skill_index_files(tmp_path, "SKILL.md")]
    assert found == [tmp_path / "pytorch-fsdp" / "SKILL.md"], found


def test_restored_skill_resolves_by_name_without_collision(tmp_path):
    _seed(tmp_path)
    names = [p.parent.name for p in iter_skill_index_files(tmp_path, "SKILL.md")]
    assert names.count("pytorch-fsdp") == 1
