"""A SessionDB open or repair for a deleted named profile never recreates its home.

Moved out of ``tests/tui_gateway/test_profiles_list_delete_race.py``, which keeps the gateway's
own cases.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_state
import hermes_state_repair
from hermes_cli import profile_lifecycle, profiles
from hermes_constants import named_profile_is_deleted
from hermes_state import SessionDB


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(profiles, "_get_wrapper_dir", lambda: fake_home / ".local" / "bin")
    return hermes_home


def test_invalid_db_quarantine_cannot_recreate_profile_deleted_after_precheck(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "state.db"
    db_path.write_bytes(b"not a SQLite database")
    real_header_check = hermes_state.has_invalid_sqlite_header_preopen
    deleted = False

    def delete_during_header_check(path, *args, **kwargs):
        nonlocal deleted
        if Path(path) == db_path and not deleted:
            assert real_header_check(path, *args, **kwargs) is True
            profile_lifecycle.mark_profile_deleting(profile_dir)
            shutil.rmtree(profile_dir)
            deleted = True
            return True
        return real_header_check(path, *args, **kwargs)

    monkeypatch.setattr(
        hermes_state, "has_invalid_sqlite_header_preopen", delete_during_header_check
    )

    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        SessionDB(db_path=db_path)

    assert deleted, "the startup header probe must exercise the deletion race"
    assert not profile_dir.exists()
    assert named_profile_is_deleted(profile_dir) is True


def test_schema_repair_lock_cannot_recreate_profile_deleted_after_precheck(
    home: Path,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "state.db"
    db_path.write_bytes(b"malformed")
    profile_lifecycle.mark_profile_deleting(profile_dir)
    shutil.rmtree(profile_dir)

    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        with hermes_state_repair._cross_process_repair_lock(db_path):
            pytest.fail("repair lock must not publish for a deleted profile")

    assert not profile_dir.exists()
    assert named_profile_is_deleted(profile_dir) is True


def test_unrelated_profiles_named_directory_keeps_normal_sessiondb_semantics(home: Path) -> None:
    db_path = home.parent / "project" / "profiles" / "worker" / "state.db"

    db = SessionDB(db_path=db_path)
    db.close()

    assert db_path.is_file()


def test_named_profile_sessiondb_never_mkdir_after_validation_race(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    db_path = profile_dir / "state.db"
    profile_dir.mkdir(parents=True)
    SessionDB(db_path=db_path).close()

    def remove_after_validation(_home: Path) -> None:
        shutil.rmtree(profile_dir)

    monkeypatch.setattr(
        hermes_state,
        "assert_named_profile_home_available",
        remove_after_validation,
    )

    with pytest.raises((FileNotFoundError, sqlite3.OperationalError)):
        SessionDB(db_path=db_path)
    assert not profile_dir.exists()


def test_deletion_tombstone_blocks_stale_sessiondb_open_in_another_process(home: Path) -> None:
    profile_dir = home / "profiles" / "worker"
    db_path = profile_dir / "state.db"
    profile_dir.mkdir(parents=True)
    SessionDB(db_path=db_path).close()
    profiles.delete_profile("worker", yes=True)

    script = (
        "import os; from pathlib import Path; from hermes_state import SessionDB; "
        "path=Path(os.environ['TARGET_DB']); "
        "\ntry:\n SessionDB(db_path=path)\nexcept FileNotFoundError:\n raise SystemExit(0)\n"
        "raise SystemExit(1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "HERMES_HOME": str(home), "TARGET_DB": str(db_path)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not profile_dir.exists()
