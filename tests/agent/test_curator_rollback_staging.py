"""Failed recovery must retain the staged files omitted from safety snapshots."""

from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", ["staging", "carry"])
@pytest.mark.parametrize("metadata_kind", ["directory", "pointer"])
def test_failed_rollback_keeps_unrestored_metadata(
    tmp_path, monkeypatch, failure, metadata_kind
):
    home = tmp_path / "home"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent import curator_backup as curator

    for name in ("alpha", "beta"):
        skill = skills / name
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            f"# {name}\nSnapshot content\n", encoding="utf-8"
        )
    target = curator.snapshot_skills()
    assert target is not None
    expected = {}
    for name in ("alpha", "beta"):
        skill = skills / name
        metadata = skill / ".git"
        if metadata_kind == "directory":
            metadata.mkdir()
            metadata /= "local-history"
        expected[name] = f"unpublished metadata for {name}".encode()
        metadata.write_bytes(expected[name])
        (skill / "SKILL.md").write_text("Current content\n", encoding="utf-8")

    real_move = curator.shutil.move
    staged_names = []

    def fail_move(source, destination, *args, **kwargs):
        source, destination = Path(source), Path(destination)
        if failure == "carry" and source.name == ".git":
            raise PermissionError("injected metadata move failure")
        if failure == "staging":
            if source.parent == skills:
                if staged_names:
                    raise PermissionError("injected staging failure")
                result = real_move(source, destination, *args, **kwargs)
                staged_names.append(source.name)
                return result
            if source.parent.name.startswith(".rollback-staging-"):
                raise PermissionError("injected recovery move failure")
        return real_move(source, destination, *args, **kwargs)

    monkeypatch.setattr(curator.shutil, "move", fail_move)
    ok, message, restored = curator.rollback(target.name)

    assert not ok
    assert restored is None
    staging = list((skills / ".curator_backups").glob(".rollback-staging-*"))
    assert len(staging) == 1
    assert str(staging[0]) in message
    for name, content in expected.items():
        relative = Path(name) / ".git"
        if metadata_kind == "directory":
            relative /= "local-history"
        copies = [root / relative for root in (skills, staging[0])]
        assert any(path.is_file() and path.read_bytes() == content for path in copies)


def test_successful_rollback_restores_metadata_and_removes_staging(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    skill = home / "skills" / "alpha"
    skill.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent import curator_backup as curator

    instructions = skill / "SKILL.md"
    instructions.write_text("Snapshot content\n", encoding="utf-8")
    target = curator.snapshot_skills()
    assert target is not None
    instructions.write_text("Current content\n", encoding="utf-8")
    (skill / ".git").write_text("gitdir: /local/worktree\n", encoding="utf-8")

    ok, _message, restored = curator.rollback(target.name)

    assert ok and restored == target
    assert instructions.read_text(encoding="utf-8-sig") == "Snapshot content\n"
    assert (skill / ".git").read_text(
        encoding="utf-8-sig"
    ) == "gitdir: /local/worktree\n"
    assert not list((skill.parent / ".curator_backups").glob(".rollback-staging-*"))
