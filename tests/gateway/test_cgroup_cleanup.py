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

    def test_returns_none_when_proc_missing(self, monkeypatch):
        def _raise(_path):
            raise FileNotFoundError

        monkeypatch.setattr(cgroup_cleanup.Path, "read_text", lambda self, *a, **k: _raise(self))
        assert cgroup_cleanup._own_cgroup_path() is None


class TestReapCgroup:
    def test_skips_own_pid_and_kills_the_rest(self, tmp_path, monkeypatch):
        own = os.getpid()
        cgroup_path = "/test.slice/hermes-gateway.service"
        procs_file = tmp_path / "cgroup.procs"
        procs_file.write_text(f"{own}\n1001\n1002\n\n")

        def _fake_path(p):
            if p == f"/sys/fs/cgroup{cgroup_path}/cgroup.procs":
                return procs_file
            return Path(p)

        monkeypatch.setattr(cgroup_cleanup, "Path", _fake_path)

        killed_pids: list[tuple[int, int]] = []
        monkeypatch.setattr(cgroup_cleanup.os, "kill", lambda pid, sig: killed_pids.append((pid, sig)))

        count = cgroup_cleanup.reap_cgroup(cgroup_path)

        assert count == 2
        assert (own, signal.SIGKILL) not in killed_pids
        assert (1001, signal.SIGKILL) in killed_pids
        assert (1002, signal.SIGKILL) in killed_pids

    def test_preserves_running_kanban_worker_tree(self, tmp_path, monkeypatch):
        cgroup_path = "/test.slice/hermes-gateway.service"
        procs_file = tmp_path / "cgroup.procs"
        procs_file.write_text("1001\n1002\n1003\n")
        monkeypatch.setattr(
            cgroup_cleanup,
            "Path",
            lambda p: procs_file if p.endswith("cgroup.procs") else Path(p),
        )
        monkeypatch.setattr(cgroup_cleanup, "_running_kanban_worker_pids", lambda: {1001})
        monkeypatch.setattr(cgroup_cleanup, "_descendant_pids", lambda roots, pids: {1002})
        killed = []
        monkeypatch.setattr(cgroup_cleanup.os, "kill", lambda pid, sig: killed.append(pid))

        assert cgroup_cleanup.reap_cgroup(cgroup_path) == 1
        assert killed == [1003]

    def test_tolerates_already_exited_pids(self, tmp_path, monkeypatch):
        cgroup_path = "/test.slice/hermes-gateway.service"
        procs_file = tmp_path / "cgroup.procs"
        procs_file.write_text("1001\n1002\n")

        monkeypatch.setattr(
            cgroup_cleanup,
            "Path",
            lambda p: procs_file if p.endswith("cgroup.procs") else Path(p),
        )

        def _kill(pid, _sig):
            if pid == 1001:
                raise ProcessLookupError
            if pid == 1002:
                raise PermissionError

        monkeypatch.setattr(cgroup_cleanup.os, "kill", _kill)

        assert cgroup_cleanup.reap_cgroup(cgroup_path) == 0

    def test_noop_when_cgroup_path_unknown(self, monkeypatch):
        monkeypatch.setattr(cgroup_cleanup, "_own_cgroup_path", lambda: None)

        def _explode(*_a, **_kw):
            pytest.fail("os.kill must not be called when cgroup path is unknown")

        monkeypatch.setattr(cgroup_cleanup.os, "kill", _explode)
        assert cgroup_cleanup.reap_cgroup() == 0

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
