"""Tests for the systemd ExecStopPost cgroup reaper (issue #37454)."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from gateway import cgroup_cleanup


class TestOwnCgroupPath:
    def test_parses_v2_cgroup_path(self, tmp_path, monkeypatch):
        proc_self = tmp_path / "cgroup"
        proc_self.write_text("0::/user.slice/user-1000.slice/hermes-gateway.service\n")
        monkeypatch.setattr(
            cgroup_cleanup,
            "Path",
            lambda p: proc_self if p == "/proc/self/cgroup" else Path(p),
        )

        assert cgroup_cleanup._own_cgroup_path() == "/user.slice/user-1000.slice/hermes-gateway.service"


class TestReapCgroup:


    def test_noop_when_procs_file_missing(self, tmp_path, monkeypatch):
        cgroup_path = "/missing.slice/hermes-gateway.service"
        monkeypatch.setattr(
            cgroup_cleanup,
            "Path",
            lambda p: tmp_path / "does-not-exist" if "cgroup.procs" in p else Path(p),
        )

        def _explode(*_a, **_kw):
            pytest.fail("os.kill must not be called when cgroup.procs is unreadable")

        monkeypatch.setattr(cgroup_cleanup.os, "kill", _explode)
        assert cgroup_cleanup.reap_cgroup(cgroup_path) == 0


class TestMain:

    def test_main_refuses_when_parent_not_systemd(self, tmp_path, monkeypatch):
        # Parent comm is a shell (any non-systemd parent): main() must refuse
        # and must never reach os.kill — the no-arg reaper shares the parent's
        # live cgroup, so reaping there SIGKILLs that process.
        comm = tmp_path / "comm"
        comm.write_text("bash\n")
        monkeypatch.setattr(cgroup_cleanup.os, "getppid", lambda: 4242)
        monkeypatch.setattr(
            cgroup_cleanup,
            "Path",
            lambda p: comm if p == "/proc/4242/comm" else Path(p),
        )

        def _explode(*_a, **_kw):
            pytest.fail("os.kill must not be called when the parent is not systemd")

        monkeypatch.setattr(cgroup_cleanup.os, "kill", _explode)
        assert cgroup_cleanup.main() == 1

    def test_main_refuses_when_comm_unreadable(self, monkeypatch):
        # Parent pid vanished (or /proc unreadable): fail closed — refuse.
        monkeypatch.setattr(cgroup_cleanup.os, "getppid", lambda: 999999)

        def _missing(*_a, **_kw):
            raise OSError("no such process")

        monkeypatch.setattr(
            Path,
            "read_text",
            _missing,
            raising=True,
        )

        def _explode(*_a, **_kw):
            pytest.fail("os.kill must not be called when the parent is undeterminable")

        monkeypatch.setattr(cgroup_cleanup.os, "kill", _explode)
        assert cgroup_cleanup.main() == 1

    def test_main_proceeds_when_parent_is_systemd(self, tmp_path, monkeypatch):
        # Simulates the ExecStopPost context: a real test child cannot have
        # systemd as its parent, so the manager's comm is stood in with a fake
        # /proc/<ppid>/comm. reap_cgroup is stubbed so the call never signals.
        comm = tmp_path / "comm"
        comm.write_text("systemd\n")
        monkeypatch.setattr(cgroup_cleanup.os, "getppid", lambda: 4242)
        monkeypatch.setattr(
            cgroup_cleanup,
            "Path",
            lambda p: comm if p == "/proc/4242/comm" else Path(p),
        )
        calls: list[tuple] = []
        monkeypatch.setattr(
            cgroup_cleanup,
            "reap_cgroup",
            lambda *a, **kw: calls.append((a, kw)) or 0,
        )
        assert cgroup_cleanup.main() == 0
        assert len(calls) == 1
