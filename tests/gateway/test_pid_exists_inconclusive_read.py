"""An inconclusive ``/proc`` read is not a death verdict (issue #110352).

``_pid_exists`` is the "is this pid alive" primitive the gateway and the kanban
dispatcher both read. psutil maps a **failed** ``/proc/<pid>/stat`` read onto
``NoSuchProcess``/``ZombieProcess`` (psutil #2418), so its ``NoSuchProcess`` is
not evidence of death: on 2026-09-13 a single dispatcher sweep booked nine
host-local kanban workers dead off it at 21:57:40Z, and all nine were alive,
heartbeating, and kept working for 12+ minutes afterwards. The authority for the
psutil branch is therefore ``_pid_confirmed_dead``: /proc must positively say
the pid is gone or a zombie, and an unreadable entry counts as alive.

These are pure unit tests: no board, no gateway.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

import gateway.status as gstatus


@pytest.mark.skipif(sys.platform != "linux", reason="/proc probe is Linux-only")
def test_inconclusive_psutil_read_of_a_live_pid_reports_alive(monkeypatch):
    """The incident, at the primitive: a failed read must not read as death."""
    import psutil

    pid = os.getpid()

    class _FailingProcess:
        def __init__(self, _pid):
            pass

        def status(self):
            # What psutil raises when it cannot read /proc/<pid>/stat.
            raise psutil.ZombieProcess(pid)

    monkeypatch.setattr(psutil, "Process", _FailingProcess)
    assert gstatus._pid_exists(pid) is True, (
        "a pid whose /proc read failed is not a dead pid"
    )


@pytest.mark.skipif(sys.platform != "linux", reason="/proc probe is Linux-only")
def test_inconclusive_read_of_a_gone_pid_still_reports_dead(monkeypatch):
    """Confirming against /proc keeps real death detectable."""
    import psutil

    absent = 991107
    while os.path.exists(f"/proc/{absent}"):
        absent += 1

    class _FailingProcess:
        def __init__(self, _pid):
            pass

        def status(self):
            raise psutil.NoSuchProcess(absent)

    monkeypatch.setattr(psutil, "Process", _FailingProcess)
    assert gstatus._pid_exists(absent) is False


@pytest.mark.skipif(sys.platform != "linux", reason="/proc probe is Linux-only")
def test_confirmed_dead_is_tri_state(monkeypatch):
    """gone/zombie True; live and unreadable False (never a death verdict)."""
    live_status = "Name:\tpython\nState:\tS (sleeping)\nPPid:\t1\n"

    class _FakeFile:
        def __init__(self, text):
            self._text = text

        def __enter__(self):
            import io

            return io.StringIO(self._text)

        def __exit__(self, *_exc):
            return False

    def _fake_open(text=None, exc=None):
        def _open(*_args, **_kwargs):
            if exc is not None:
                raise exc
            return _FakeFile(text)

        return _open

    monkeypatch.setattr(gstatus, "open", _fake_open(exc=FileNotFoundError()), raising=False)
    assert gstatus._pid_confirmed_dead(991108) is True

    monkeypatch.setattr(gstatus, "open", _fake_open(exc=PermissionError("denied")), raising=False)
    assert gstatus._pid_confirmed_dead(991108) is False, (
        "an unreadable entry is not evidence of death"
    )

    monkeypatch.setattr(gstatus, "open", _fake_open(text=live_status), raising=False)
    assert gstatus._pid_confirmed_dead(991108) is False

    monkeypatch.setattr(gstatus, "open", _fake_open(text="Name:\tx\nState:\tZ (zombie)\n"), raising=False)
    assert gstatus._pid_confirmed_dead(991108) is True, (
        "a zombie is dead for every caller that acts on this answer"
    )

    monkeypatch.delattr(gstatus, "open", raising=False)
    assert gstatus._pid_confirmed_dead(os.getpid()) is False
