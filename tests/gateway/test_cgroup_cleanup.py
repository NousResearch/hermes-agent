"""Tests for the systemd ExecStopPost cgroup reaper (issue #37454)."""

from __future__ import annotations

import os
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

    def test_main_refuses_when_pid1_parent_is_not_systemd(self, tmp_path, monkeypatch):
        # Regression: a container where the gateway itself is PID 1 (or init
        # is tini/launchd) must NOT be authorized by "ppid == 1". PID 1 has to
        # present as systemd in /proc/1/comm like any other parent.
        comm = tmp_path / "comm"
        comm.write_text("tini\n")
        monkeypatch.setattr(cgroup_cleanup.os, "getppid", lambda: 1)
        monkeypatch.setattr(
            cgroup_cleanup,
            "Path",
            lambda p: comm if p == "/proc/1/comm" else Path(p),
        )

        def _explode(*_a, **_kw):
            pytest.fail("os.kill must not be called for a non-systemd PID 1 parent")

        monkeypatch.setattr(cgroup_cleanup.os, "kill", _explode)
        assert cgroup_cleanup._parent_is_systemd() is False
        assert cgroup_cleanup.main() == 1


class TestLiveGatewayGuard:

    @pytest.mark.parametrize("verb", ["run", "restart"])
    def test_reap_refuses_when_live_gateway_in_cgroup(self, monkeypatch, verb):
        # Regression: a live gateway PID still in cgroup.procs (gateway is PID 1
        # in a plain container, a targeted reap of a running service, or an
        # in-process `gateway restart` on a host without a service manager)
        # must abort the reap before any signal.
        import gateway.status

        cgroup_path = "/some.slice/some-gateway.service"
        gateway_cmdline = f"/opt/hermes/venv/bin/python -m hermes_cli.main gateway {verb}"
        monkeypatch.setattr(
            cgroup_cleanup, "_read_cgroup_pids", lambda _p: [777, os.getpid()]
        )
        monkeypatch.setattr(
            gateway.status,
            "_read_process_cmdline",
            lambda pid: gateway_cmdline if pid == 777 else None,
        )

        def _explode(*_a, **_kw):
            pytest.fail("os.kill must not signal a cgroup holding a live gateway")

        monkeypatch.setattr(cgroup_cleanup.os, "kill", _explode)
        assert cgroup_cleanup.reap_cgroup(cgroup_path) == 0

    def test_reap_proceeds_when_only_orphans_in_cgroup(self, monkeypatch):
        # Allow-path contract: orphans with non-gateway command lines must
        # still be reaped — the guard must not destroy the feature it secures.
        import gateway.status

        monkeypatch.setattr(
            cgroup_cleanup, "_read_cgroup_pids", lambda _p: [777, os.getpid()]
        )
        monkeypatch.setattr(
            gateway.status,
            "_read_process_cmdline",
            lambda pid: "bash -c adb forward tcp:8888 tcp:8889" if pid == 777 else None,
        )
        killed: list[int] = []
        monkeypatch.setattr(
            cgroup_cleanup.os, "kill", lambda pid, sig: killed.append(pid)
        )
        assert cgroup_cleanup.reap_cgroup("/some.slice/some-gateway.service") == 1
        assert killed == [777]
