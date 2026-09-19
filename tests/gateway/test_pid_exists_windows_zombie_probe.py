"""Windows guard on the psutil zombie probe in ``_pid_exists``.

``_pid_exists`` historically ran ``psutil.Process(pid).status()`` on every
call to detect POSIX zombies.  Windows has no zombie state, and the call
costs ~7 ms there — paid once per registry entry while holding the
active-session file lock on every poller pass, which starved pollers at
>=7 leases (#115578).  On Windows liveness must come straight from
``psutil.pid_exists()`` without the status() syscall; on POSIX the zombie
contract (zombies report dead, #42126) must not change.
"""

import sys
from unittest.mock import MagicMock

from gateway import status


class _FakeProcess:
    """Counts status() reads so tests can prove the probe never runs."""

    def __init__(self, prober):
        self._prober = prober

    def status(self):
        self._prober.status_calls += 1
        return self._prober.status_result


class _StatusProber:
    def __init__(self, *, status_result="sleeping"):
        self.status_calls = 0
        self.status_result = status_result

    def __call__(self, pid):
        return _FakeProcess(self)


def _install_psutil(monkeypatch, prober, *, pid_exists):
    fake = MagicMock()
    fake.STATUS_ZOMBIE = "zombie"
    fake.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    fake.Process.side_effect = prober
    fake.pid_exists.return_value = pid_exists
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake


def test_windows_skips_zombie_status_probe(monkeypatch):
    # status() would answer "zombie" if called: the guard, not the data,
    # is what keeps Windows off the ~7 ms syscall.
    prober = _StatusProber(status_result="zombie")
    fake = _install_psutil(monkeypatch, prober, pid_exists=True)
    monkeypatch.setattr(status, "_IS_WINDOWS", True)

    assert status._pid_exists(4242) is True
    assert prober.status_calls == 0
    fake.pid_exists.assert_called_once_with(4242)


def test_windows_dead_pid_still_comes_from_pid_exists(monkeypatch):
    prober = _StatusProber()
    _install_psutil(monkeypatch, prober, pid_exists=False)
    monkeypatch.setattr(status, "_IS_WINDOWS", True)

    assert status._pid_exists(4242) is False
    assert prober.status_calls == 0


def test_posix_still_reports_zombie_dead(monkeypatch):
    prober = _StatusProber(status_result="zombie")
    _install_psutil(monkeypatch, prober, pid_exists=True)
    monkeypatch.setattr(status, "_IS_WINDOWS", False)

    assert status._pid_exists(4242) is False
    assert prober.status_calls == 1


def test_posix_no_such_process_reports_dead(monkeypatch):
    fake = MagicMock()
    fake.STATUS_ZOMBIE = "zombie"
    fake.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    fake.Process.side_effect = fake.NoSuchProcess("gone")
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setattr(status, "_IS_WINDOWS", False)

    assert status._pid_exists(4242) is False
