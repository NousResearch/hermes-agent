import inspect
import os
import sys

import pytest

from tui_gateway import slash_worker


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, getppid=lambda: 999999) is True


def test_is_orphaned_false_when_direct_parent_is_unchanged():
    original_ppid = 1234
    assert slash_worker._is_orphaned(original_ppid, getppid=lambda: original_ppid) is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only OpenProcess probe")
def test_win32_probe_live_parent_is_not_orphaned():
    # Our own PID is alive by definition; the probe must not flag it.
    assert slash_worker._is_orphaned(os.getpid()) is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only OpenProcess probe")
def test_win32_probe_dead_pid_is_orphaned():
    # A PID Windows can never have: OpenProcess returns NULL and the probe
    # treats a missing handle as orphaned. Must not spuriously pass on a
    # real (possibly recycled) parent PID, hence the max-range sentinel.
    assert slash_worker._is_orphaned(0x7FFFFFFF) is True


def test_parent_death_watchdog_contract_has_no_create_time_plumbing():
    assert list(inspect.signature(slash_worker._is_orphaned).parameters) == [
        "original_ppid",
        "getppid",
    ]
    assert list(inspect.signature(slash_worker._start_parent_death_watchdog).parameters) == [
        "original_ppid",
    ]
