"""Marker watchdog: only a provably-dead, long-stale, unlocked marker is archived."""
import os
import time

import pytest

import hermes_constants
from hermes_cli.update_lock import (
    STALE_MARKER_QUARANTINE_AGE_SECONDS,
    quarantine_stale_marker,
    update_marker_path,
)

DEAD_PID = 2**30


@pytest.fixture
def lock_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    return home


def _dead_marker(age_seconds: float) -> bytes:
    return f"{DEAD_PID}\n{int(time.time() - age_seconds)}\nrun:test-run\n".encode()


def test_absent_marker_quarantines_nothing(lock_home):
    assert quarantine_stale_marker() is None


def test_malformed_marker_is_never_touched(lock_home):
    update_marker_path().write_bytes(b"not a marker at all\n")
    assert quarantine_stale_marker() is None
    assert update_marker_path().is_file()


def test_young_dead_marker_is_never_touched(lock_home):
    update_marker_path().write_bytes(_dead_marker(60))
    assert quarantine_stale_marker() is None
    assert update_marker_path().is_file()


def test_live_marker_is_never_touched(lock_home):
    from hermes_cli.update_lock import process_create_time

    ct = process_create_time(os.getpid())
    assert ct is not None
    lock_home_marker = update_marker_path()
    lock_home_marker.write_bytes(f"{os.getpid()}\n{int(time.time())}\nct:{ct:.3f}\n".encode())
    assert quarantine_stale_marker() is None
    assert lock_home_marker.is_file()


def test_old_dead_marker_is_archived_beside_the_original(lock_home):
    update_marker_path().write_bytes(_dead_marker(STALE_MARKER_QUARANTINE_AGE_SECONDS + 60))
    target = quarantine_stale_marker()
    assert target is not None
    assert not update_marker_path().exists()
    assert target.is_file()
    assert target.name.startswith(update_marker_path().name + ".quarantined-")


def test_dry_run_reports_without_moving(lock_home):
    update_marker_path().write_bytes(_dead_marker(STALE_MARKER_QUARANTINE_AGE_SECONDS + 60))
    target = quarantine_stale_marker(dry_run=True)
    assert target is not None
    assert update_marker_path().is_file()
