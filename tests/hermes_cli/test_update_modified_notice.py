"""Skill sync keeps user edits and tells the updater's user how to inspect them."""

import shutil
from pathlib import Path

import pytest

from tools import skills_sync
from hermes_cli.update_cmd_maint import _print_bundled_skills_sync_report, _profile_skill_sync_status


def test_kept_skill_edits_have_an_actionable_update_report(tmp_path, monkeypatch, capsys):
    bundled = tmp_path / "bundled"
    source = bundled / "example/SKILL.md"
    source.parent.mkdir(parents=True)
    source.write_text("---\nname: example\ndescription: fixture\n---\nOriginal\n", encoding="utf-8")
    installed = tmp_path / "skills"
    monkeypatch.setattr(skills_sync, "SKILLS_DIR", installed)
    monkeypatch.setattr(skills_sync, "MANIFEST_FILE", installed / ".manifest.json")
    monkeypatch.setattr(skills_sync, "_get_bundled_dir", lambda: bundled)
    monkeypatch.setattr(skills_sync, "_get_optional_dir", lambda: tmp_path / "optional")
    monkeypatch.setattr("agent.skill_utils.get_external_skills_dirs", list)

    _print_bundled_skills_sync_report()
    user_skill = installed / "example/SKILL.md"
    assert user_skill.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert "list-modified" not in capsys.readouterr().out
    user_skill.write_text("My local instructions\n", encoding="utf-8")
    source.write_text(source.read_text(encoding="utf-8") + "Upstream revision\n", encoding="utf-8")

    _print_bundled_skills_sync_report()
    out = capsys.readouterr().out
    assert "1 user-modified (kept)" in out
    assert "hermes skills list-modified" in out
    assert user_skill.read_text(encoding="utf-8") == "My local instructions\n"


def test_update_reports_retained_copies_that_no_longer_receive_bundled_updates(tmp_path, monkeypatch, capsys):
    home = tmp_path / "profile"
    home.mkdir()
    bundled = tmp_path / "bundled"
    optional = tmp_path / "optional"
    optional.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_sync, "_get_bundled_dir", lambda: bundled)
    monkeypatch.setattr(skills_sync, "_get_optional_dir", lambda: optional)
    for name in ("optional-local-edit", "optional-pristine", "dropped-local", "dropped-archived", "still-bundled"):
        source = bundled / name / "SKILL.md"
        source.parent.mkdir(parents=True)
        source.write_text(f"---\nname: {name}\ndescription: fixture\n---\nOriginal\n", encoding="utf-8")

    assert "dropped-local" in skills_sync.sync_skills(quiet=True)["copied"]
    installed = home / "skills"
    edited = installed / "optional-local-edit" / "SKILL.md"
    edited.write_text(edited.read_text(encoding="utf-8") + "Local edits\n", encoding="utf-8")
    archive = installed / ".archive"
    archive.mkdir()
    (installed / "dropped-archived").rename(archive / "dropped-archived")
    for name in ("optional-local-edit", "optional-pristine"):
        (bundled / name).rename(optional / name)
    for name in ("dropped-local", "dropped-archived"):
        (bundled / name).rename(tmp_path / name)
    manifest = installed / ".bundled_manifest"
    manifest_before = manifest.read_bytes()

    _print_bundled_skills_sync_report()
    out = capsys.readouterr().out
    assert "1 moved to optional, local copy kept (no bundled updates): optional-local-edit" in out
    assert "#skills-that-leave-the-bundle" in out
    # The fix depends on the profile and on Hub installs, so the notice names no mutating command.
    assert "repair-official" not in out
    assert "1 removed from the bundle, local copy kept (no bundled updates): dropped-local" in out
    assert "Skills are up to date" not in out
    for name in ("optional-pristine", "still-bundled", "dropped-archived"):
        assert name not in out
    # The unedited optional copy is adopted by the official-provenance backfill, so it is Hub-managed.
    assert skills_sync._read_hub_install_paths() == {"optional-pristine"}
    profile_status = _profile_skill_sync_status(skills_sync.sync_skills(quiet=True))
    assert "!1 moved to optional" in profile_status and "!1 removed from the bundle" in profile_status
    assert edited.read_text(encoding="utf-8").endswith("Local edits\n")
    assert (installed / "dropped-local" / "SKILL.md").is_file()
    assert (archive / "dropped-archived" / "SKILL.md").is_file()
    assert manifest.read_bytes() == manifest_before

    # Deleting a dropped copy, as the docs suggest, stops the report and drops its manifest entry.
    shutil.rmtree(installed / "dropped-local")
    result = skills_sync.sync_skills(quiet=True)
    assert all(name != "dropped-local" for name, _ in result["retired_kept"])
    assert "dropped-local" in result["cleaned"]


def _retired_fixture(tmp_path, monkeypatch, folder, name):
    """Install one bundled skill in a temp profile, edit it, then drop it from the bundle."""
    home, bundled, optional = tmp_path / "profile", tmp_path / "bundled", tmp_path / "optional"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_sync, "_get_bundled_dir", lambda: bundled)
    monkeypatch.setattr(skills_sync, "_get_optional_dir", lambda: optional)
    source = bundled / "cat" / folder / "SKILL.md"
    source.parent.mkdir(parents=True)
    source.write_text(f"---\nname: {name}\ndescription: fixture\n---\nOriginal\n", encoding="utf-8")
    skills_sync.sync_skills(quiet=True)
    installed = home / "skills" / "cat" / folder / "SKILL.md"
    installed.write_text(installed.read_text(encoding="utf-8") + "Local edits\n", encoding="utf-8")
    return home, source, optional


def test_skill_renamed_on_move_to_optional_is_not_reported_as_removed(tmp_path, monkeypatch, capsys):
    _, source, optional = _retired_fixture(tmp_path, monkeypatch, "renamed-slug", "old-public-name")
    moved = optional / "cat" / "renamed-slug" / "SKILL.md"
    moved.parent.mkdir(parents=True)
    moved.write_text("---\nname: new-public-name\ndescription: fixture\n---\nOfficial\n", encoding="utf-8")
    shutil.rmtree(source.parent)

    _print_bundled_skills_sync_report()
    out = capsys.readouterr().out
    assert "1 moved to optional, local copy kept (no bundled updates): old-public-name" in out
    assert "removed from the bundle" not in out


def _write_hub_lock(home, monkeypatch, content):
    """Write the profile's Hub lock; ``None`` writes one whose read is denied."""
    lock = home / "skills" / ".hub" / "lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b"{}" if content is None else content)
    if content is None:  # mode bits don't stop root, so deny every touch of the lock path
        for method in ("exists", "read_text"):  # Path.exists() raises PermissionError on 3.12
            real = getattr(Path, method)

            def denied(self, *args, _real=real, **kwargs):
                if self == lock:
                    raise PermissionError(13, "Permission denied", str(self))
                return _real(self, *args, **kwargs)

            monkeypatch.setattr(Path, method, denied)


@pytest.mark.parametrize("lock", [b"[1]", b"{not json", b"\xff\xfe{}", b'{"installed": []}', None],
                         ids=["list", "malformed", "undecodable", "installed-list", "access-denied"])
def test_unreadable_hub_lock_skips_the_notice_without_breaking_sync(tmp_path, monkeypatch, lock):
    home, source, _ = _retired_fixture(tmp_path, monkeypatch, "dropped", "dropped")
    shutil.rmtree(source.parent)
    _write_hub_lock(home, monkeypatch, lock)

    result = skills_sync.sync_skills(quiet=True)
    assert result["retired_kept"] == []
    assert "dropped" not in result["cleaned"]
    assert _profile_skill_sync_status(result) == "up to date"


@pytest.mark.parametrize("lock", [b"", b'{"installed": {"other": {"install_path": "elsewhere/other"}}}'],
                         ids=["no-lock", "lock-without-this-copy"])
def test_hub_lock_that_does_not_own_the_copy_still_reports_it(tmp_path, monkeypatch, lock):
    home, source, _ = _retired_fixture(tmp_path, monkeypatch, "dropped", "dropped")
    shutil.rmtree(source.parent)
    if lock:
        _write_hub_lock(home, monkeypatch, lock)

    assert skills_sync.sync_skills(quiet=True)["retired_kept"] == [["dropped", "cat/dropped"]]
