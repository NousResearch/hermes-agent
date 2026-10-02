"""Profile export's ``extra_files`` (the Desktop's ``desktop.json`` overlay) never write through a symlink.

``export_profile`` stages the profile with ``copytree(symlinks=True)``, so a symlinked source file or
directory is still a symlink in staging. An extra file staged at that path must land in the archive, not
in the link's target (the source profile's real file, or a dotfiles checkout the link points into).
"""

import tarfile
from pathlib import Path

import pytest

from hermes_cli.profiles import create_profile, export_profile


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.mark.require_symlinks
def test_extra_files_land_in_archive_without_writing_through_symlinks(profile_env, tmp_path):
    profile_dir = create_profile("coder", no_alias=True)
    dotfiles = tmp_path / "dotfiles"
    shared = dotfiles / "shared"
    shared.mkdir(parents=True)
    linked_file = dotfiles / "desktop.json"
    linked_file.write_text('{"theme": "mine"}', encoding="utf-8")
    (profile_dir / "desktop.json").symlink_to(linked_file)
    (profile_dir / "shared").symlink_to(shared, target_is_directory=True)
    extra = {"desktop.json": '{"theme": "overlay"}', "shared/notes.json": '{"n": 1}'}

    archive = export_profile("coder", str(tmp_path / "coder.tar.gz"), extra_files=extra)

    with tarfile.open(archive, "r:gz") as tf:
        archived = {
            member.name.removeprefix("coder/"): tf.extractfile(member).read().decode("utf-8")
            for member in tf.getmembers()
            if member.isfile() and member.name.removeprefix("coder/") in extra
        }
    assert archived == extra
    assert linked_file.read_text(encoding="utf-8") == '{"theme": "mine"}'
    assert list(shared.iterdir()) == []
