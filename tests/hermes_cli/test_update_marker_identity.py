"""Shared update marker identity against a scratch home."""
import os
import sys
import time
import pytest
from hermes_cli import update_marker_identity as gate

@pytest.mark.parametrize("alive,created,status", [
    (False, None, "stale"), (True, 99.9, "matching"),
    (True, 100.9, "matching"), (True, 101, "stale"),
    (True, None, "unknown"),
])
def test_pid_identity_contract(alive, created, status):
    assert gate.pid_identity_status(123, 100, pid_alive=lambda pid: alive,
                                    pid_create_time=lambda pid: created) == status


def test_proven_owner_never_expires(tmp_path):
    marker = tmp_path / gate.MARKER_NAME
    marker.write_text("123\n100\n")
    assert gate.live_update_marker_owner(marker, now=10000, pid_alive=lambda pid: True,
                                         pid_create_time=lambda pid: 99.9) is not None
    assert marker.exists()


def test_unprovable_owner_expires_at_the_age_ceiling(tmp_path):
    """Upstream main unlinks a marker past 20 min. The PR kept unknown owners
    blocking forever, so one recycled PID disabled the bridge permanently."""
    marker = tmp_path / gate.MARKER_NAME
    marker.write_text("123\n100\n")
    inside = 100 + gate.UPDATE_MARKER_MAX_AGE_SECONDS - 60
    beyond = 100 + gate.UPDATE_MARKER_MAX_AGE_SECONDS + 60
    unknown = dict(pid_alive=lambda pid: True, pid_create_time=lambda pid: None)
    assert gate.live_update_marker_owner(marker, now=inside, **unknown) is not None
    assert gate.live_update_marker_owner(marker, now=beyond, **unknown) is None
    assert marker.exists(), "readers never mutate; the lock owner reclaims"


def test_posix_creation_time_is_provable_without_third_party_packages(tmp_path):
    """B7: _pid_create_time returned None unconditionally off Windows, so every
    live PID stayed `unknown` on macOS/Linux and no marker ever aged out."""
    created = gate.pid_create_time(os.getpid())
    if os.name != "nt" and not sys.platform.startswith(("linux", "darwin")):
        pytest.skip(f"no stdlib creation-time source for {sys.platform}")
    assert created is not None, f"{sys.platform} must prove its own creation time"
    assert 0 < created <= time.time() + 1
    assert gate.pid_identity_status(os.getpid(), int(time.time()) + 5) == "matching"


def test_dead_owner_reader_does_not_delete_marker(tmp_path):
    marker = tmp_path / gate.MARKER_NAME
    marker.write_text("123\n100\n")
    assert gate.live_update_marker_owner(marker, now=101, pid_alive=lambda pid: False) is None
    assert marker.exists()


def test_tagged_bridge_keeps_gate_closed_during_claim_gap(tmp_path):
    marker = tmp_path / gate.MARKER_NAME
    marker.write_text("123\n100\nhandoff-bridge\n")
    assert gate.live_update_marker_owner(marker, now=110, pid_alive=lambda pid: False) is not None
    assert gate.live_update_marker_owner(marker, now=131, pid_alive=lambda pid: False) is None
