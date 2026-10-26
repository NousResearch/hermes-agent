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
    staging = list((skills / ".curator_backups").glob(".rollback-*"))
    assert len(staging) == 1
    assert str(staging[0]) in message
    # The next curator pass (or a retry's safety snapshot) must not prune the retained copy.
    assert curator.snapshot_skills() is not None
    for name, content in expected.items():
        relative = Path(name) / ".git"
        if metadata_kind == "directory":
            relative /= "local-history"
        copies = [root / relative for root in (skills, staging[0])]
        assert any(path.is_file() and path.read_bytes() == content for path in copies)


@pytest.mark.parametrize("mode", ["rename_fails", "collision"])
def test_retain_staging_never_raises(tmp_path, monkeypatch, mode):
    """A failed rollback returns a failure, never raises, even when retaining the staging dir fails or collides."""
    home = tmp_path / "home"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent import curator_backup as curator

    skill = skills / "alpha"
    skill.mkdir()
    (skill / "SKILL.md").write_text("snap\n", encoding="utf-8")
    target = curator.snapshot_skills()
    (skill / ".git").write_bytes(b"meta")

    real_move = curator.shutil.move

    def fail_move(src, dst, *a, **k):
        if Path(src).name == ".git":
            raise PermissionError("injected carry failure")
        return real_move(src, dst, *a, **k)

    monkeypatch.setattr(curator.shutil, "move", fail_move)
    backups = skills / ".curator_backups"
    if mode == "rename_fails":
        def boom(self, *a, **k):
            raise PermissionError("injected rename")
        monkeypatch.setattr(Path, "rename", boom)
    else:
        monkeypatch.setattr(curator, "_utc_id", lambda: "2020-01-01T00-00-00Z")
        (backups / ".rollback-unrestored-2020-01-01T00-00-00Z").mkdir(parents=True)

    ok, message, restored = curator.rollback(target.name)  # must not raise
    assert not ok and restored is None
    assert "could not restore alpha/.git" in message
    if mode == "rename_fails":
        assert "rename failed" in message and ".rollback-staging-" in message
    else:
        kept = backups / ".rollback-unrestored-2020-01-01T00-00-00Z-1"
        assert kept.is_dir() and str(kept) in message
