"""A reboot can hand the dead gateway's PID to a process that looks alive but is not ours.

Windows recycles PIDs aggressively: after a reboot the number stored in ``gateway.pid``,
``gateway.lock`` and the machine-global ``telegram-bot-token-*.lock`` may belong to an unrelated
protected process. ``OpenProcess`` then fails with ACCESS_DENIED — which ``win32_pid_exists``
reports as "alive" — while psutil can read neither its start time nor its command line. Every
identity probe comes back empty at once, so the record must be judged by the boot it was written
on; otherwise the ghost holds the token lock forever (exit 78, parked, no retry) and the pid record
reports "another gateway instance is already running" (exit 1).

Every record below is stamped in 2023, so it predates the boot of any host running these tests —
except where a test replaces the stamp to model a same-boot holder.
"""

import os

import pytest

from gateway import status
from gateway.status_scoped_locks import _scoped_lock_record_is_stale

# Centisecond epoch fingerprint (the Windows ``create_time * 100`` shape): 2023-11-14 23:13:20Z.
_PREVIOUS_START_TIME = 1_700_003_600 * 100
_PREVIOUS_STAMP = "2023-11-14T23:13:20+00:00"


@pytest.fixture
def ghost(monkeypatch):
    """A PID that answers "alive" while yielding no identity at all (the post-reboot ghost)."""
    pid = os.getpid()
    monkeypatch.setattr(status, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(status, "_get_process_start_time", lambda _pid: None)
    monkeypatch.setattr(status, "_read_process_cmdline", lambda _pid: None)
    return pid


def _record(**overrides):
    record = {
        "pid": os.getpid(),
        "kind": "hermes-gateway",
        "argv": ["/home/u/.hermes/hermes-agent/hermes_cli/main.py", "gateway", "run"],
        "start_time": _PREVIOUS_START_TIME,
        "hermes_home": "",
        "scope": "telegram-bot-token",
        "metadata": {"platform": "telegram"},
        "updated_at": _PREVIOUS_STAMP,
    }
    record.update(overrides)
    return record


def test_pid_record_from_a_previous_boot_is_not_live(ghost):
    assert status._live_pid_from_record(_record()) is None


def test_scoped_lock_from_a_previous_boot_is_reclaimed(ghost):
    assert _scoped_lock_record_is_stale(_record(), ghost)


def test_scoped_lock_from_a_previous_boot_is_reclaimed_by_its_stamp(ghost):
    """``start_time`` as a POSIX tick count is boot-relative: the record stamp still dates it."""
    assert _scoped_lock_record_is_stale(_record(start_time=4321), ghost)


def test_unidentifiable_owner_within_this_boot_stays_reserved(ghost):
    """A same-boot holder we merely cannot inspect keeps its lock: nothing proves it is gone."""
    record = _record(start_time=None, updated_at=None)
    assert not _scoped_lock_record_is_stale(record, ghost)
    assert status._live_pid_from_record(record) is not None


def test_identifiable_holder_is_never_reclaimed_by_its_timestamps(monkeypatch):
    """A clock step can make a live holder look older than boot; a readable identity outvotes it."""
    pid = os.getpid()
    monkeypatch.setattr(status, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(status, "_get_process_start_time", lambda _pid: _PREVIOUS_START_TIME)
    monkeypatch.setattr(status, "_read_process_cmdline", lambda _pid: "hermes gateway run")
    assert not _scoped_lock_record_is_stale(_record(metadata={}), pid)


def test_same_boot_naive_utc_stamp_stays_reserved_in_non_utc_timezone(ghost, monkeypatch):
    """An offset-free persisted UTC stamp must not be interpreted as local time."""
    import time
    from datetime import UTC, datetime
    from gateway import status_boot_epoch

    if not hasattr(time, "tzset"):
        pytest.skip("requires process-local POSIX timezone override")

    boot = 1_700_000_000
    monkeypatch.setattr(status_boot_epoch, "current_boot_epoch", lambda: boot)
    stamp = datetime.fromtimestamp(boot + 600, UTC).replace(tzinfo=None).isoformat()
    record = _record(start_time=None, updated_at=stamp)

    saved_tz = os.environ.get("TZ")
    try:
        monkeypatch.setenv("TZ", "JST-9")
        time.tzset()
        assert not _scoped_lock_record_is_stale(record, ghost)
        assert status._live_pid_from_record(record) == ghost
    finally:
        if saved_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", saved_tz)
        time.tzset()
