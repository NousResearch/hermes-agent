"""Clock-independent Linux ownership; related upstream issue #105714."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import active_sessions as registry

pytestmark = pytest.mark.platforms("linux")


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))


def acquire(owner="first", config=None):
    return registry.try_acquire_active_session(
        session_id="canonical", surface="desktop", config=config or {},
        metadata={"live_session_id": owner}, track_liveness=True,
    )


def test_clock_step_keeps_snapshot_and_excludes_competing_writer(monkeypatch):
    lease, error = acquire()
    assert lease is not None and error is None
    entries = registry._read_entries(registry._state_path(), strict=True)
    recorded = entries[0]["process_start_time"]
    epoch = registry._process_start_time(os.getpid())
    monkeypatch.setattr(registry, "_OWN_START", None)
    monkeypatch.setattr(registry, "_process_start_time", lambda pid: epoch + 7)

    assert registry.active_session_registry_snapshot(strict=True) == entries
    competitor, refusal = acquire("second")
    assert competitor is None
    assert refusal.reason == registry.SESSION_NOT_OWNED
    assert registry._read_entries(registry._state_path())[0]["process_start_time"] == recorded
    replacement, error = acquire()
    assert replacement is not None and error is None
    replacement.release()
    assert registry.active_session_registry_snapshot(strict=True) == []


def test_new_linux_lease_uses_explicit_boot_identity():
    lease, error = acquire()
    assert lease is not None and error is None
    marker = registry._read_entries(registry._state_path())[0]["process_start_time"]
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ticks = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
    assert marker == f"linux-v1:{boot}:{ticks}"


def test_recycled_pid_and_previous_boot_are_pruned():
    lease, error = acquire()
    assert lease is not None and error is None
    entries = registry._read_entries(registry._state_path())
    marker = entries[0]["process_start_time"]
    prefix, boot, ticks = marker.split(":")
    for stale in [f"{prefix}:{boot}:{int(ticks) + 1}",
                  f"{prefix}:00000000-0000-0000-0000-000000000000:{ticks}"]:
        entries[0]["process_start_time"] = stale
        registry._write_entries(registry._state_path(), entries)
        assert registry.active_session_registry_snapshot(strict=True) == []


def test_unreadable_identity_preserves_registry_and_refuses_claim(monkeypatch):
    lease, error = acquire()
    assert lease is not None and error is None
    before = registry._state_path().read_bytes()
    monkeypatch.setattr(registry, "_linux_process_identity", lambda pid: None)
    with pytest.raises(registry.ActiveSessionRegistryError):
        registry.active_session_registry_snapshot(strict=True)
    with pytest.raises(registry.ActiveSessionRegistryError):
        acquire("second")
    assert registry._state_path().read_bytes() == before


@pytest.mark.parametrize("marker", ["linux-v1:bad:12", "linux-v1:", "linux-v2:unknown:1"])
def test_malformed_identity_is_not_silently_pruned(marker):
    lease, error = acquire()
    assert lease is not None and error is None
    entries = registry._read_entries(registry._state_path())
    entries[0]["process_start_time"] = marker
    registry._write_entries(registry._state_path(), entries)
    before = registry._state_path().read_bytes()
    with pytest.raises(registry.ActiveSessionRegistryError):
        registry.active_session_registry_snapshot(strict=True)
    assert registry._state_path().read_bytes() == before


def test_legacy_epoch_entries_keep_the_existing_comparison(monkeypatch):
    monkeypatch.setattr(registry, "_own_start_time", lambda: 1000.0)
    assert registry._pid_liveness(os.getpid(), 1000.0) is True
    assert registry._pid_liveness(os.getpid(), 999.0) is False


def test_process_name_with_spaces_and_parentheses(monkeypatch):
    original = Path.read_text
    stat_path = Path(f"/proc/{os.getpid()}/stat")
    stat = original(stat_path)
    expected = registry._linux_process_identity(os.getpid())
    unusual = stat[:stat.index("(")] + "(worker ) with (spaces))" + stat[stat.rindex(")") + 1:]

    def read(path, *args, **kwargs):
        return unusual if path == stat_path else original(path, *args, **kwargs)

    with patch.object(Path, "read_text", read):
        assert registry._linux_process_identity(os.getpid()) == expected


@pytest.mark.parametrize("fault", [PermissionError("denied"), "malformed stat", "123 (name) R"])
def test_failed_proc_probe_never_falls_back_to_epoch(fault, monkeypatch):
    original = Path.read_text
    stat_path = Path(f"/proc/{os.getpid()}/stat")

    def read(path, *args, **kwargs):
        if path == stat_path:
            if isinstance(fault, Exception):
                raise fault
            return fault
        return original(path, *args, **kwargs)

    monkeypatch.setattr(registry, "_process_start_time", lambda pid: pytest.fail("epoch fallback"))
    with patch.object(Path, "read_text", read):
        assert registry._linux_process_identity(os.getpid()) is None
        with pytest.raises(registry.ActiveSessionRegistryError):
            acquire()


def test_unknown_owner_keeps_capacity_and_untracked_snapshot(monkeypatch):
    lease, error = acquire()
    assert lease is not None and error is None
    entries = registry._read_entries(registry._state_path())
    entries[0].pop("track_liveness")
    registry._write_entries(registry._state_path(), entries)
    original = registry._linux_process_identity
    owner_pid = os.getpid() + 100000
    entries[0]["pid"] = owner_pid
    registry._write_entries(registry._state_path(), entries)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
    monkeypatch.setattr(registry, "_linux_process_identity",
                        lambda pid: None if pid == owner_pid else original(pid))
    assert registry.active_session_registry_snapshot() == entries
    with pytest.raises(registry.ActiveSessionRegistryError, match="liveness is unknown"):
        acquire("second")
    other, refusal = registry.try_acquire_active_session(
        session_id="other", surface="desktop", config={"max_concurrent_sessions": 1},
    )
    assert other is None and refusal.reason == registry.MAX_CONCURRENT_SESSIONS


def test_real_child_lease_is_retained_across_epoch_probe_drift(tmp_path, monkeypatch):
    import subprocess
    import sys

    code = """
import json, sys
from hermes_cli.active_sessions import try_acquire_active_session
lease, error = try_acquire_active_session(session_id='canonical', surface='desktop', config={}, track_liveness=True)
assert lease is not None, error
print(json.dumps({'pid': __import__('os').getpid()}), flush=True)
sys.stdin.readline()
"""
    child = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        pid = json.loads(child.stdout.readline())["pid"]
        epoch = registry._process_start_time(pid)
        monkeypatch.setattr(registry, "_process_start_time", lambda other: epoch + 7)
        snapshot = registry.active_session_registry_snapshot(strict=True)
        assert len(snapshot) == 1 and snapshot[0]["pid"] == pid
        assert acquire("competitor")[0] is None
    finally:
        child.communicate("stop\n", timeout=10)
    assert child.returncode == 0
    assert registry.active_session_registry_snapshot(strict=True) == []
