"""Start-time fingerprints agree across processes of different ages on macOS (#138216).

psutil's public ``create_time()`` on macOS adds ``INIT_BOOT_TIME - boot_time()``, where
``INIT_BOOT_TIME`` is captured when the reading process first imported psutil. ``kern.boottime``
steps by seconds over the host's uptime, so a gateway that imported psutil days ago read every
other process's start time shifted by the accumulated drift, while the CLI that recorded the
fingerprint read it unshifted. Past the 2 s tolerance the cron reaper judged a live
``hermes cron run`` owner dead and marked its execution ``unknown`` mid-run.

A long-running reader is simulated the way the report measured it: pin psutil's
``INIT_BOOT_TIME`` to 7 s before the current boot time, as if kern.boottime had stepped 7 s
since this process imported psutil.
"""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway import status, status_start_time

_DRIFT_S = 7.0


@pytest.fixture()
def live_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield child
    finally:
        child.kill()
        child.wait(timeout=10)


@pytest.fixture()
def long_running_reader(monkeypatch):
    """This process reads start times like a gateway whose psutil import predates a 7 s drift."""
    import psutil
    from psutil import _psosx

    monkeypatch.setattr(_psosx, "INIT_BOOT_TIME", psutil.boot_time() - _DRIFT_S)


@pytest.mark.platforms("macos")
def test_fingerprint_does_not_depend_on_when_the_reader_imported_psutil(live_child, long_running_reader):
    import psutil

    # What the public reading gives this "old" reader: shifted by the drift.
    shifted = round(psutil.Process(live_child.pid).create_time() * 100)
    fingerprint = status.get_process_start_time(live_child.pid)
    # What a freshly started process (no psutil correction yet) reads for the same child.
    fresh = round(psutil.Process(live_child.pid)._proc.create_time(monotonic=True) * 100)

    assert abs(shifted - fresh) >= _DRIFT_S * 100 - 1  # the drift is real in this setup
    assert fingerprint == fresh
    assert status.start_time_fingerprints_match(fresh, fingerprint)


@pytest.mark.platforms("macos")
def test_cron_reaper_keeps_a_live_external_owner_after_boot_time_drift(live_child, long_running_reader):
    """The report's failure: the CLI recorded its start time fresh, the gateway re-reads it later."""
    from cron import executions
    import psutil

    recorded_by_cli = round(psutil.Process(live_child.pid)._proc.create_time(monotonic=True) * 100)

    assert executions._owner_is_live(live_child.pid, recorded_by_cli) is True


def test_backend_without_the_kernel_reading_keeps_the_public_creation_time():
    """psutil backends whose ``create_time`` takes no ``monotonic`` argument (Windows ``fast_only``,
    SunOS, AIX) apply no reader-dependent correction; the public value is the fingerprint source."""
    windows_like = SimpleNamespace(_proc=SimpleNamespace(create_time=lambda fast_only=False: 111.0),
                                   create_time=lambda: 1700000000.25)
    no_backend = SimpleNamespace(create_time=lambda: 1700000000.5)

    assert status_start_time.psutil_create_time(windows_like) == 1700000000.25
    assert status_start_time.psutil_create_time(no_backend) == 1700000000.5
