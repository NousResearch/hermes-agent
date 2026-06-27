import psutil
import pytest

from tui_gateway import slash_worker


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, 1.0, getppid=lambda: 999999) is True


def test_is_orphaned_true_when_parent_create_time_mismatch():
    # Same ppid but a different create_time means the PID was reused.
    me = psutil.Process()
    assert slash_worker._is_orphaned(me.pid, 0.0, getppid=lambda: me.pid) is True


def test_is_orphaned_false_when_parent_alive_and_matches():
    me = psutil.Process()
    assert (
        slash_worker._is_orphaned(me.pid, me.create_time(), getppid=lambda: me.pid) is False
    )


def test_is_orphaned_true_when_parent_is_zombie(monkeypatch):
    me = psutil.Process()

    class ZombieParent:
        def status(self):
            return psutil.STATUS_ZOMBIE

        def create_time(self):
            return me.create_time()

    monkeypatch.setattr(slash_worker.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(slash_worker.psutil, "Process", lambda pid: ZombieParent())
    assert slash_worker._is_orphaned(me.pid, me.create_time(), getppid=lambda: me.pid) is True
