import errno
import os
import tarfile
from pathlib import Path

import pytest

from hermes_cli import profiles


def _archive(path: Path, root: str = "incoming") -> Path:
    payload = path.parent / "config.yaml"
    payload.write_text("model: test\n", encoding="utf-8")
    with tarfile.open(path, "w:gz") as tf:
        tf.add(payload, arcname=f"{root}/config.yaml")
    return path


@pytest.fixture()
def profiles_root(tmp_path, monkeypatch):
    root = tmp_path / "profiles"
    root.mkdir()
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: root)
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: root / name)
    monkeypatch.setattr(profiles, "validate_profile_name", lambda name: None)
    return root


def _members(archive: Path) -> set:
    with tarfile.open(archive, "r:gz") as tf:
        return set(tf.getnames())


def test_export_rejects_file_hardlinked_outside_profile_root(tmp_path, profiles_root):
    source = profiles_root / "source"
    (source / "memories").mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("private", encoding="utf-8")
    os.link(outside, source / "memories" / "MEMORY.md")

    with pytest.raises(ValueError, match="hardlinked outside the profile root"):
        profiles.export_profile("source", str(tmp_path / "export.tar.gz"))


def test_export_allows_hardlinked_auth_store_and_inside_profile_hardlinks(tmp_path, profiles_root):
    """A profile auth.json hardlinked to the root store is a supported shared-grant setup and is
    never archived; two names for one inode inside the profile escape nothing."""
    source = profiles_root / "source"
    (source / "skills" / "demo").mkdir(parents=True)
    root_auth = tmp_path / "auth.json"
    root_auth.write_text('{"grants": {}}', encoding="utf-8")
    os.link(root_auth, source / "auth.json")
    (source / "skills" / "demo" / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    os.link(source / "skills" / "demo" / "SKILL.md", source / "skills" / "demo" / "COPY.md")

    members = _members(profiles.export_profile("source", str(tmp_path / "export.tar.gz")))

    assert "source/skills/demo/SKILL.md" in members
    assert "source/skills/demo/COPY.md" in members
    assert "source/auth.json" not in members


def test_default_export_ignores_hardlinks_in_unexported_profiles(tmp_path, monkeypatch):
    """``profiles/`` is outside the default export's root allow-list, so a hardlink inside some
    other profile must not fail ``hermes profile export default``."""
    home = tmp_path / ".hermes"
    (home / "profiles" / "other").mkdir(parents=True)
    (home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    os.link(outside, home / "profiles" / "other" / "notes.txt")
    monkeypatch.setattr(profiles, "_existing_profile_dir", lambda name: ("default", home))

    members = _members(profiles.export_profile("default", str(tmp_path / "default.tar.gz")))

    assert "default/config.yaml" in members
    assert not any(m.startswith("default/profiles") for m in members)


def test_import_stages_beside_destination_and_publishes_with_rename(tmp_path, profiles_root, monkeypatch):
    archive = _archive(tmp_path / "incoming.tar.gz")
    real_rename = os.rename
    calls = []

    def rename_on_destination_filesystem(src, dst):
        calls.append((Path(src), Path(dst)))
        assert Path(src).parent.parent == profiles_root
        real_rename(src, dst)

    monkeypatch.setattr(profiles.os, "rename", rename_on_destination_filesystem)
    imported = profiles.import_profile(str(archive), name="published")

    assert imported == profiles_root / "published"
    assert (imported / "config.yaml").read_text(encoding="utf-8") == "model: test\n"
    assert calls and calls[-1][1] == imported
    assert not any(p.name.startswith(".published.import-") for p in profiles_root.iterdir())


def test_import_never_falls_back_to_copy_on_cross_device_rename(tmp_path, profiles_root, monkeypatch):
    """``shutil.move`` would answer EXDEV with a non-atomic copytree; publication must fail."""
    archive = _archive(tmp_path / "incoming.tar.gz")

    real_rename = os.rename

    def cross_device(src, dst):
        if Path(dst) == profiles_root / "published":
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_rename(src, dst)

    monkeypatch.setattr(profiles.os, "rename", cross_device)
    with pytest.raises(OSError) as excinfo:
        profiles.import_profile(str(archive), name="published")

    assert excinfo.value.errno == errno.EXDEV
    assert not (profiles_root / "published").exists()
    assert not any(p.name.startswith(".published.import-") for p in profiles_root.iterdir())


def test_import_refuses_a_raced_destination_without_nesting_into_it(tmp_path, profiles_root, monkeypatch):
    """A profile published by a concurrent import between the exists() check and the rename must
    be reported as existing, not replaced and not nested into (``shutil.move`` semantics)."""
    archive = _archive(tmp_path / "incoming.tar.gz")
    real_rename = os.rename

    def racing_rename(src, dst):
        if Path(dst) == profiles_root / "published":
            Path(dst).mkdir()
            (Path(dst) / "config.yaml").write_text("model: winner\n", encoding="utf-8")
        real_rename(src, dst)

    monkeypatch.setattr(profiles.os, "rename", racing_rename)
    with pytest.raises(FileExistsError, match="already exists"):
        profiles.import_profile(str(archive), name="published")

    winner = profiles_root / "published"
    assert sorted(p.name for p in winner.iterdir()) == ["config.yaml"]
    assert (winner / "config.yaml").read_text(encoding="utf-8") == "model: winner\n"
