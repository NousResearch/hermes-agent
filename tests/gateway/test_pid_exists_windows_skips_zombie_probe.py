"""_pid_exists must not run the POSIX zombie probe on Windows (#115578).

``psutil.Process(pid).status()`` costs ~7 ms per call and zombies do not
exist on Windows. It runs once per registry entry inside the exclusive
session-registry file lock, so at a handful of leases pollers starve on
the unfair ``msvcrt.locking(LK_LOCK)`` retry cadence. On Windows the
authoritative checks (``psutil.pid_exists()`` / ctypes OpenProcess) apply.
"""

import os
import sys
import types
from unittest.mock import MagicMock

from gateway import status


class _OsWithName(types.ModuleType):
    """Proxy of the real os module reporting a fake os.name.

    Patching ``os.name`` globally breaks ``pathlib`` (``Path`` dispatches
    on it) and with it pytest itself; swapping only ``gateway.status``'s
    reference keeps the rest of the process on the real platform.
    """

    def __init__(self, name):
        super().__init__("os")
        self.__dict__["_real_os"] = os
        self.__dict__["name"] = name

    def __getattr__(self, attr):
        return getattr(self.__dict__["_real_os"], attr)


def _set_os_name(monkeypatch, name):
    monkeypatch.setattr(status, "os", _OsWithName(name))


def _stub_psutil(monkeypatch, *, ps_status="running", exists=True):
    fake = MagicMock()
    fake.STATUS_ZOMBIE = "zombie"
    fake.NoSuchProcess = type("NoSuchProcess", (Exception,), {})

    constructed = []

    class FakeProcess:
        def __init__(self, pid):
            constructed.append(pid)
            self.pid = pid

        def status(self):
            return ps_status

    fake.Process = FakeProcess
    fake.pid_exists = MagicMock(return_value=exists)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake, constructed


def test_windows_skips_zombie_status_probe(monkeypatch):
    """os.name == nt: no Process().status() probe; pid_exists() decides."""
    fake, constructed = _stub_psutil(monkeypatch, exists=True)
    _set_os_name(monkeypatch, "nt")

    assert status._pid_exists(123456) is True

    assert constructed == []
    fake.pid_exists.assert_called_once_with(123456)


def test_windows_missing_pid_reports_dead_without_probe(monkeypatch):
    fake, constructed = _stub_psutil(monkeypatch, exists=False)
    _set_os_name(monkeypatch, "nt")

    assert status._pid_exists(123456) is False

    assert constructed == []


def test_posix_zombie_still_reports_dead(monkeypatch):
    """POSIX behavior is unchanged: zombies read as dead."""
    fake, constructed = _stub_psutil(monkeypatch, ps_status="zombie", exists=True)
    _set_os_name(monkeypatch, "posix")

    assert status._pid_exists(123456) is False

    assert constructed == [123456]


def test_posix_live_reports_alive(monkeypatch):
    fake, constructed = _stub_psutil(monkeypatch, ps_status="running", exists=True)
    _set_os_name(monkeypatch, "posix")

    assert status._pid_exists(123456) is True

    assert constructed == [123456]
    fake.pid_exists.assert_called_once_with(123456)