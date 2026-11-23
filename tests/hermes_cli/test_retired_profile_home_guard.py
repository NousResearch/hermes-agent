"""A writer still bound to a profile home that ``profile delete`` or ``profile rename`` retired cannot
re-create it in this process, through ``Path.mkdir``, ``os.makedirs`` or ``shutil.copytree``.

Three writers are shipped code (pairing, the SQLite quarantine lock, the sandbox dir) that rebuilt the
home on main; the others spell the mkdir in ways a static check missed. ``hermes_constants`` lists
what the guard does not cover.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import profiles
from hermes_constants import named_profile_is_deleted

MOVES = ("delete", "rename")


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _retire(name: str, move: str) -> Path:
    with patch("hermes_cli.profiles._cleanup_gateway_service"), patch("hermes_cli.profiles._stop_profile_backends"), \
         patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
        home = profiles.create_profile(name, no_alias=True, no_skills=True)
        if move == "delete":
            profiles.delete_profile(name, yes=True)
        else:
            profiles.rename_profile(name, f"{name}-renamed")
    assert not home.exists()
    return home


def _pairing(home: Path, monkeypatch) -> None:
    from gateway.pairing import PairingStore
    PairingStore(profile=home.name)


def _quarantine_lock(home: Path, monkeypatch) -> None:
    from persistence.sqlite._state.dbfile import quarantine_cross_process_lock
    with quarantine_cross_process_lock(home / "state.db"):
        pass


def _sandbox_dir(home: Path, monkeypatch) -> None:
    from tools.environments import base
    monkeypatch.setenv("HERMES_HOME", str(home))
    base.get_sandbox_dir()


def _makedirs_fstring(home: Path, monkeypatch) -> None:
    os.makedirs(f"{home}/cache/images", exist_ok=True)


def _positional_parents(home: Path, monkeypatch) -> None:
    (home / "logs").mkdir(0o700, True, True)


def _copytree_into(home: Path, monkeypatch) -> None:
    src = home.parent.parent / "skel"
    (src / "skills").mkdir(parents=True)
    shutil.copytree(src, home / "restored-skel")


WRITERS = {
    "PairingStore": _pairing,
    "quarantine_cross_process_lock": _quarantine_lock,
    "get_sandbox_dir": _sandbox_dir,
    "os.makedirs f-string": _makedirs_fstring,
    "Path.mkdir positional parents": _positional_parents,
    "shutil.copytree": _copytree_into,
}


@pytest.mark.parametrize("move", MOVES)
@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_a_writer_bound_to_a_retired_home_cannot_rebuild_it(root, monkeypatch, writer, move):
    home = _retire("worker", move)
    with pytest.raises(FileNotFoundError, match="was deleted or renamed"):
        WRITERS[writer](home, monkeypatch)
    assert not home.exists()


@pytest.mark.parametrize("move", MOVES)
def test_a_writer_thread_cannot_rebuild_it_either(root, monkeypatch, move):
    import threading
    home = _retire("worker", move)
    errors = []
    thread = threading.Thread(target=lambda: _try(lambda: (home / "logs").mkdir(parents=True), errors))
    thread.start()
    thread.join()
    assert errors and not home.exists()


def _try(fn, errors):
    try:
        fn()
    except FileNotFoundError as exc:
        errors.append(exc)


def test_the_refusal_names_the_command_that_brings_the_name_back(root):
    home = _retire("worker", "delete")
    with pytest.raises(FileNotFoundError, match="hermes profile create worker"):
        (home / "cache").mkdir(parents=True)


@pytest.mark.parametrize("move", MOVES)
def test_create_brings_a_retired_name_back(root, move):
    home = _retire("worker", move)
    with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
        assert profiles.create_profile("worker", no_alias=True, no_skills=True) == home
    assert not named_profile_is_deleted(home)
    (home / "cache" / "x").mkdir(parents=True)  # a live profile's writers are untouched


def test_import_brings_a_deleted_name_back(root, tmp_path):
    import tarfile
    home = _retire("worker", "delete")
    tree = tmp_path / "export" / "worker"
    tree.mkdir(parents=True)
    (tree / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    archive = tmp_path / "worker.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tree, arcname="worker")
    assert profiles.import_profile(str(archive), name="worker") == home
    assert home.is_dir() and not named_profile_is_deleted(home)


def test_a_writer_racing_the_rename_cannot_rebuild_the_old_home(root, monkeypatch):
    """The move and the writer's next mkdir interleave: the old name is retired before the move lands."""
    from hermes_cli import profiles_initialization
    real = profiles_initialization.publish_staging
    seen = []

    def move_then_write(src, dst, exists_error):
        real(src, dst, exists_error)
        try:
            (src / "logs").mkdir(parents=True)
        except FileNotFoundError as exc:
            seen.append(exc)

    monkeypatch.setattr(profiles_initialization, "publish_staging", move_then_write)
    home = _retire("worker", "rename")
    assert seen and not home.exists()


def _backup(tmp_path, files):
    import zipfile
    backup = tmp_path / "backup.zip"
    with zipfile.ZipFile(backup, "w") as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    return backup


def _import(backup, monkeypatch):
    from argparse import Namespace

    import hermes_cli.gateway as gateway_mod
    import hermes_cli.gateway_setup_service as service_setup
    from hermes_cli.backup import run_import
    monkeypatch.setattr(service_setup, "ensure_gateway_service", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)
    return run_import(Namespace(zipfile=str(backup), force=True))


def test_a_rename_onto_a_retired_name_makes_it_live(root):
    _retire("after", "delete")
    with patch("hermes_cli.profiles._cleanup_gateway_service"), patch("hermes_cli.profiles._stop_profile_backends"), \
         patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
        profiles.create_profile("before", no_alias=True, no_skills=True)
        new = profiles.rename_profile("before", "after")
    assert profiles.profile_exists("after") and not named_profile_is_deleted(new)


def test_distribution_install_brings_a_deleted_name_back(root, tmp_path):
    from hermes_cli.profile_distribution import DistributionManifest, install_distribution, write_manifest
    home = _retire("worker", "delete")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "config.yaml").write_text("model:\n  default: dist\n", encoding="utf-8")
    write_manifest(staged, DistributionManifest(name="worker"))
    assert install_distribution(str(staged), name="worker").target_dir == home
    assert profiles.profile_exists("worker")


@pytest.mark.parametrize("move", MOVES)
def test_a_restore_into_a_retired_home_completes_as_before_the_guard(root, tmp_path, monkeypatch, move):
    """The restore, its reservation and nested members write into the retired home as on main."""
    home = _retire("worker", move)
    _import(_backup(tmp_path, {"config.yaml": "model: {}\n", "profiles/worker/config.yaml": "model: {}\n",
                               "profiles/worker/memories/MEMORY.md": "m\n"}), monkeypatch)
    assert (home / "config.yaml").is_file() and (home / "memories" / "MEMORY.md").is_file()


def test_a_restore_whose_target_is_the_retired_home_itself_completes(root, tmp_path, monkeypatch):
    """``HERMES_HOME`` names the deleted profile: the restore makes its own target as on main."""
    home = _retire("worker", "delete")
    monkeypatch.setenv("HERMES_HOME", str(home))
    _import(_backup(tmp_path, {"config.yaml": "model: {}\n"}), monkeypatch)
    assert (home / "config.yaml").is_file()


def test_a_writer_thread_during_a_restore_is_still_refused(root):
    import threading

    from hermes_constants import restoring_profile_homes
    home = _retire("worker", "delete")
    errors = []
    with restoring_profile_homes():
        thread = threading.Thread(target=lambda: _try(lambda: (home / "logs").mkdir(parents=True), errors))
        thread.start()
        thread.join()
    assert errors and not home.exists()


def test_an_existing_home_still_gets_eexist_from_an_exist_ok_mkdir(root):
    """An unfinished delete keeps its home and its record; the owner reservation re-mkdirs it exist_ok."""
    from hermes_constants import mark_named_profile_deleted
    with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
        home = profiles.create_profile("worker", no_alias=True, no_skills=True)
    mark_named_profile_deleted(home, incomplete=True)
    home.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileExistsError):
        home.mkdir()


def test_the_guard_is_installed_once_however_often_the_module_is_reloaded(tmp_path):
    """Test files reload hermes_constants; a reload must not stack a second hook. Five reloads, then
    one mkdir: the module-level hook body runs once, so exactly one hook is installed."""
    script = (
        "import importlib, sys, os, hermes_constants as hc\n"
        "for _ in range(5): importlib.reload(hc)\n"
        "seen = []\n"
        "real_split = os.path.split\n"
        "os.path.split = lambda p: seen.append(p) or real_split(p)\n"
        "os.mkdir(sys.argv[1])\n"
        "os.path.split = real_split\n"
        "print(len(seen))\n")
    out = subprocess.run([sys.executable, "-c", script, str(tmp_path / "probe")], capture_output=True, text=True,
                         check=True, cwd=Path(__file__).resolve().parents[2])
    assert out.stdout.strip() == "1"
