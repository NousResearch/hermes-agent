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
        assert cgroup_cleanup.main() == 1


class TestLiveGatewayGuard:
    @pytest.mark.parametrize("verb", ["run", "restart"])
    def test_reap_refuses_when_live_gateway_in_cgroup(self, monkeypatch, verb):
        # Regression: a live gateway PID still in cgroup.procs (gateway is PID 1
        # in a plain container, a targeted reap of a running service, or an
        # in-process `gateway restart` on a host without a service manager)
        # must abort the reap before any signal — even under a systemd parent —
        # and main() must report the refusal with a non-zero exit.
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
        assert cgroup_cleanup.reap_cgroup(cgroup_path) is None
        monkeypatch.setattr(cgroup_cleanup, "_parent_is_systemd", lambda: True)
        monkeypatch.setattr(cgroup_cleanup, "_own_cgroup_path", lambda: cgroup_path)
        assert cgroup_cleanup.main() == 1

        # Allow-path: once the gateway is gone, a non-gateway orphan in the
        # same cgroup must still be reaped — the guard must not destroy the
        # feature it secures.
        monkeypatch.setattr(
            gateway.status,
            "_read_process_cmdline",
            lambda pid: "bash -c adb forward tcp:8888 tcp:8889" if pid == 777 else None,
        )
        killed: list[int] = []
        monkeypatch.setattr(cgroup_cleanup.os, "kill", lambda pid, sig: killed.append(pid))
        assert cgroup_cleanup.reap_cgroup(cgroup_path) == 1
        assert killed == [777]

class TestForegroundScopeSweep:
    """The ExecStopPost parity for foreground scopes (#70716): a SIGKILLed gateway
    leaves its long-lived command in ``hermes-fg-<pid>-*.scope``, which neither the
    unit's KillMode nor the cgroup reap reaches, and the next gateway has a new PID."""

    def test_sweep_reads_the_dead_pid_from_the_record_and_does_not_block(self, monkeypatch):
        import gateway.status

        monkeypatch.setattr(gateway.status, "_read_pid_record", lambda path=None: {"pid": 4242})
        seen: dict = {}
        monkeypatch.setattr(
            "tools.environments.local.stop_foreground_scopes",
            lambda pid=None, **kw: seen.update(pid=pid, **kw),
        )

        assert cgroup_cleanup.reap_foreground_scopes() is True
        # KillMode=mixed never reaches the scope, so $MAINPID would have to — and it is
        # unset in ExecStopPost. The record is the substitute; the PID stays in the glob.
        assert seen == {"pid": 4242, "no_block": True}

    def test_sweep_accepts_an_explicit_pid(self, monkeypatch):
        seen: list = []
        monkeypatch.setattr(
            "tools.environments.local.stop_foreground_scopes",
            lambda pid=None, **kw: seen.append((pid, kw)),
        )

        assert cgroup_cleanup.reap_foreground_scopes(777) is True
        assert seen == [(777, {"no_block": True})]

    def test_no_sweep_without_a_record(self, monkeypatch):
        # A graceful stop unlinks the record before exiting, and it also stops its own
        # scopes in-process — so a missing record must not turn into a blind sweep.
        import gateway.status

        monkeypatch.setattr(gateway.status, "_read_pid_record", lambda path=None: None)
        called: list = []
        monkeypatch.setattr(
            "tools.environments.local.stop_foreground_scopes", lambda *a, **kw: called.append(a)
        )

        assert cgroup_cleanup.reap_foreground_scopes() is False
        assert called == []

    def test_main_sweeps_only_after_a_permitted_reap(self, monkeypatch):
        monkeypatch.setattr(cgroup_cleanup, "_parent_is_systemd", lambda: True)
        swept: list = []
        monkeypatch.setattr(cgroup_cleanup, "reap_foreground_scopes", lambda *a, **kw: swept.append(a))

        # Refusal path (a live gateway is in the cgroup): its scopes must survive.
        monkeypatch.setattr(cgroup_cleanup, "reap_cgroup", lambda *_a: None)
        assert cgroup_cleanup.main() == 1
        assert swept == []

        monkeypatch.setattr(cgroup_cleanup, "reap_cgroup", lambda *_a: 0)
        assert cgroup_cleanup.main() == 0
        assert swept == [()]
