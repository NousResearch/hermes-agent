"""Import, distribution install and rename each publish a directory under a name.

A profile published under a previously deleted name is a live profile: ``profile delete`` leaves a
tombstone at ``profiles/.deleted/<name>``, and a tombstone left for that name made the new profile
unlisted, unresolvable through ``-p``, and refused by ``create`` as existing.

Import and rename publish the way create does, so a profile published under that name in the meantime
wins cleanly: the loser raises the usual "already exists" and leaves no trace.
"""

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import profiles, profiles_initialization
from hermes_cli.profile_distribution import DistributionManifest, install_distribution, write_manifest


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return tmp_path


def _create_and_delete(name: str) -> None:
    profiles.create_profile(name, no_alias=True)
    with patch("hermes_cli.profiles._cleanup_gateway_service"), \
         patch("hermes_cli.profiles._live_default_multiplexer", return_value=False):
        profiles.delete_profile(name, yes=True)


def _publish_by_import(tmp_path: Path, name: str) -> None:
    profiles.create_profile("source", no_alias=True)
    archive = profiles.export_profile("source", str(tmp_path / "source.tar.gz"))
    profiles.import_profile(str(archive), name=name)


def _publish_by_install(tmp_path: Path, name: str) -> None:
    source = tmp_path / "distribution"
    source.mkdir()
    (source / "SOUL.md").write_text("Distribution.\n")
    (source / "config.yaml").write_text("model:\n  model: m\n")
    write_manifest(source, DistributionManifest(name=name, version="0.1.0"))
    install_distribution(str(source), name=name)


def _publish_by_rename(tmp_path: Path, name: str) -> None:
    profiles.create_profile("source", no_alias=True)
    with patch("hermes_cli.profiles._cleanup_gateway_service"):
        profiles.rename_profile("source", name)


@pytest.mark.parametrize("publish", [_publish_by_import, _publish_by_install, _publish_by_rename],
                         ids=["import", "install", "rename"])
def test_profile_published_under_a_deleted_name_is_live(profile_env, publish):
    _create_and_delete("reused")

    publish(profile_env, "reused")

    assert "reused" in [p.name for p in profiles.list_profiles()]
    assert profiles.resolve_profile_env("reused") == str(profiles.get_profile_dir("reused"))


def _archive(tmp_path: Path) -> Path:
    profiles.create_profile("source", no_alias=True)
    (profiles.get_profile_dir("source") / "MARK_IMPORTED").write_text("i")
    return Path(profiles.export_profile("source", str(tmp_path / "source.tar.gz")))


def _publish_winner(name: str) -> None:
    winner = profiles.get_profile_dir(name)
    winner.mkdir(parents=True)
    (winner / "MARK_WINNER").write_text("w")


def _leftovers() -> list:
    return sorted(p.name for p in profiles.get_profile_dir("x").parent.iterdir() if ".staging-" in p.name)


def test_an_import_racing_a_publish_of_its_name_refuses_and_leaves_the_winner_alone(profile_env, monkeypatch):
    archive = _archive(profile_env)
    real = profiles_initialization.prepare_imported_tree

    def publish_meanwhile(staged):
        real(staged)
        _publish_winner("racer")

    monkeypatch.setattr(profiles_initialization, "prepare_imported_tree", publish_meanwhile)
    with pytest.raises(FileExistsError, match="already exists"):
        profiles.import_profile(str(archive), name="racer")
    assert sorted(p.name for p in profiles.get_profile_dir("racer").iterdir()) == ["MARK_WINNER"]
    assert _leftovers() == []


@pytest.mark.platforms("windows")
def test_a_losing_import_removes_the_read_only_files_it_extracted(profile_env, monkeypatch):
    """Extraction re-applies each member's mode, so a read-only file (a git object) lands read-only, and
    Windows refuses to unlink a read-only file: a plain rmtree leaves the staged tree behind."""
    profiles.create_profile("source", no_alias=True)
    pack = profiles.get_profile_dir("source") / "skills" / "repo" / "p.pack"
    pack.parent.mkdir(parents=True)
    pack.write_text("p")
    pack.chmod(0o444)
    archive = Path(profiles.export_profile("source", str(profile_env / "source.tar.gz")))
    real = profiles_initialization.prepare_imported_tree

    def publish_meanwhile(staged):
        real(staged)
        _publish_winner("racer")

    monkeypatch.setattr(profiles_initialization, "prepare_imported_tree", publish_meanwhile)
    with pytest.raises(FileExistsError, match="already exists"):
        profiles.import_profile(str(archive), name="racer")
    assert _leftovers() == []


def test_an_import_waits_for_a_create_of_the_same_name_in_this_process(profile_env, monkeypatch):
    """An in-process create and import of one name serialize on the create lock: the import waits."""
    from hermes_cli import profile_provision

    archive = _archive(profile_env)
    real_lock, asked, done = profile_provision._create_lock, threading.Event(), threading.Event()

    def observed_lock(name):
        asked.set()  # the import reached the lock
        return real_lock(name)

    monkeypatch.setattr(profile_provision, "_create_lock", observed_lock)
    with real_lock("racer"):
        worker = threading.Thread(target=lambda: (profiles.import_profile(str(archive), name="racer"), done.set()))
        worker.start()
        assert asked.wait(10)
        assert not done.wait(2) and _leftovers() == []  # held at the lock: nothing staged yet
    worker.join(10)
    assert done.is_set() and (profiles.get_profile_dir("racer") / "MARK_IMPORTED").exists()


@pytest.mark.platforms("posix")
def test_an_import_stages_privately_until_its_files_are_tightened(profile_env, monkeypatch):
    archive = _archive(profile_env)
    real, modes = profiles_initialization.prepare_imported_tree, []

    def record(staged):
        modes.append(staged.parent.stat().st_mode & 0o777)
        assert staged.parent.parent == profiles.get_profile_dir("private").parent  # beside the target: one rename
        real(staged)

    monkeypatch.setattr(profiles_initialization, "prepare_imported_tree", record)
    import os
    old_umask = os.umask(0o022)  # the process umask: a mkdir without mode=0o700 would give 0o755
    try:
        profiles.import_profile(str(archive), name="private")
    finally:
        os.umask(old_umask)
    assert modes == [0o700] and _leftovers() == []  # the emptied staging root is removed too


def test_a_rename_racing_a_publish_of_the_new_name_refuses_and_keeps_both(profile_env, monkeypatch):
    profiles.create_profile("alpha", no_alias=True)
    (profiles.get_profile_dir("alpha") / "MARK_ALPHA").write_text("a")
    real = profiles._stop_bot_desktop

    def publish_meanwhile(home):
        real(home)
        _publish_winner("beta")

    monkeypatch.setattr(profiles, "_stop_bot_desktop", publish_meanwhile)
    with patch("hermes_cli.profiles._cleanup_gateway_service", return_value=False), \
         patch("hermes_cli.profiles._live_default_multiplexer", return_value=False), \
         pytest.raises(FileExistsError, match="already exists"):
        profiles.rename_profile("alpha", "beta")
    assert (profiles.get_profile_dir("alpha") / "MARK_ALPHA").exists()
    assert sorted(p.name for p in profiles.get_profile_dir("beta").iterdir()) == ["MARK_WINNER"]
