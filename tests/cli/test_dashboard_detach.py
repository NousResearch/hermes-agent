"""Tests for `hermes dashboard --detach` (#40587).

Covers the new detach path:

* ``_find_stale_dashboard_pids`` merges the PID file into the cmdline scan
  so ``--status`` / ``--stop`` find detached servers reliably.
* ``_detach_and_run`` constructs the grandchild argv correctly (host,
  port, --no-open / --insecure / --pid-file are passed through).
* ``cmd_dashboard`` rejects an occupied bind address before forking so
  the user gets a clean error instead of a 15-second timeout.

The double-fork itself is NOT exercised end-to-end here — running
``os.fork()`` from inside pytest's worker would inherit file
descriptors, leak the test's signal handlers, and (on macOS) confuse
Coverage.py.  Instead we mock ``os.fork`` and ``os.execvp`` to a
no-op and assert on the argv that would have been exec'd.  The
real-world detach path is covered by manual smoke tests in the PR
description.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────


def _write_pid_file(home: Path, pid: int) -> Path:
    """Materialise ~/.hermes/run/dashboard.pid with *pid* inside."""
    pid_dir = home / "run"
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / "dashboard.pid"
    pid_file.write_text(str(pid))
    return pid_file


def _patch_pid_exists(monkeypatch, *, alive: set[int] | None = None) -> None:
    """Stub gateway.status._pid_exists so tests don't need real processes.

    By default no PID is alive; pass ``alive={pid, ...}`` to whitelist
    specific PIDs as "running" (e.g. the test's own).
    """
    alive = alive or set()

    def _fake(pid: int) -> bool:
        return pid in alive

    # _find_stale_dashboard_pids imports this lazily inside the function
    # body, so patching the module is sufficient.
    monkeypatch.setattr(
        "gateway.status._pid_exists", _fake, raising=False
    )
    # Also patch at the import site just in case the lazy import was
    # already resolved (matters for tests run in the same process).
    import gateway.status as _gs

    monkeypatch.setattr(_gs, "_pid_exists", _fake)


def _import_main():
    """Lazy import of hermes_cli.main (it's a heavy module)."""
    import hermes_cli.main as m

    return m


# ── _find_stale_dashboard_pids: PID file integration ──────────────────────


class TestPidFileMerging:
    """The PID file is a precise pointer and is preferred over the
    cmdline scan when present."""

    def test_pid_file_with_alive_pid_is_returned(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Don't run a real cmdline scan — there's no dashboard running
        # in the test env, so pgrep would return [] anyway.  Stub it
        # defensively.
        _patch_pid_exists(monkeypatch, alive={4242})
        _write_pid_file(tmp_path, 4242)

        m = _import_main()
        pids = m._find_stale_dashboard_pids()
        assert 4242 in pids

    def test_pid_file_pointing_at_dead_pid_is_skipped(
        self, tmp_path, monkeypatch
    ):
        """A stale PID file (process gone) must not cause us to try
        to kill a non-existent process."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Default: no PIDs are "alive" — including the one in the file.
        _patch_pid_exists(monkeypatch)
        _write_pid_file(tmp_path, 9999)

        m = _import_main()
        pids = m._find_stale_dashboard_pids()
        assert 9999 not in pids

    def test_exclude_pids_filters_pid_file(
        self, tmp_path, monkeypatch
    ):
        """``exclude_pids`` (used by the desktop backend to protect its
        own child) must apply to PID file hits too, not just cmdline
        hits."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _patch_pid_exists(monkeypatch, alive={5000})
        _write_pid_file(tmp_path, 5000)

        m = _import_main()
        pids = m._find_stale_dashboard_pids(exclude_pids={5000})
        assert pids == []

    def test_self_pid_excluded(self, tmp_path, monkeypatch):
        """We never include our own PID, regardless of where it came from."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        my_pid = os.getpid()
        _patch_pid_exists(monkeypatch, alive={my_pid})
        _write_pid_file(tmp_path, my_pid)

        m = _import_main()
        pids = m._find_stale_dashboard_pids()
        assert my_pid not in pids

    def test_garbage_pid_file_does_not_crash(
        self, tmp_path, monkeypatch
    ):
        """A truncated or non-numeric PID file must not raise — the
        cmdline scan is the fallback."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_file = _write_pid_file(tmp_path, 1234)  # write valid first
        pid_file.write_text("not-a-pid\n")
        _patch_pid_exists(monkeypatch)

        m = _import_main()
        # Should not raise; result is whatever the cmdline scan returns
        # (empty in a test env).
        pids = m._find_stale_dashboard_pids()
        assert isinstance(pids, list)


# ── _detach_and_run: argv construction ─────────────────────────────────────


class TestDetachArgv:
    """_detach_and_run should pass the right flags to the grandchild
    so it re-binds the same host/port and writes the same PID file."""

    def _capture_grandchild_argv(self, monkeypatch, **kwargs):
        """Run _detach_and_run with fork/exec mocked, return what was
        passed to ``os.execvp``.

        We simulate the GRANDCHILD path: the first fork returns 0
        (we are the child), setsid succeeds, the second fork also
        returns 0 (we are the grandchild), at which point execvp is
        called with the argv we want to assert on.
        """
        captured: dict = {}
        fork_call_count = {"n": 0}

        def _fake_fork():
            fork_call_count["n"] += 1
            # 1st call: pretend we're the first-fork child (return 0).
            # 2nd call: pretend we're the grandchild (return 0).
            return 0

        def _fake_setsid():
            return 0

        def _fake_execvp(file, args):
            captured["file"] = file
            captured["argv"] = list(args)
            # Raise so we never reach the post-exec code or the
            # port-wait loop in the launcher path.
            raise SystemExit(0)

        monkeypatch.setattr("os.fork", _fake_fork)
        monkeypatch.setattr("os.setsid", _fake_setsid)
        monkeypatch.setattr("os.execvp", _fake_execvp)
        # Stub open() for /dev/null so we don't depend on host config.
        monkeypatch.setattr("os.open", lambda *a, **kw: 99)
        monkeypatch.setattr("os.dup2", lambda *a, **kw: None)
        monkeypatch.setattr("os.close", lambda *a, **kw: None)
        monkeypatch.setattr("os._exit", lambda code: None)

        from hermes_cli.web_server import _detach_and_run

        with mock.patch("sys.exit") as exit_mock:
            with mock.patch("builtins.print") as print_mock:
                try:
                    _detach_and_run(**kwargs)
                except SystemExit:
                    # _fake_execvp raises; let it bubble.
                    pass

        return captured

    def test_minimal_argv_passes_host_port(self, monkeypatch, tmp_path):
        """Default flags: --host and --port, no extras."""
        captured = self._capture_grandchild_argv(
            monkeypatch,
            host="127.0.0.1",
            port=9119,
            open_browser=True,
            allow_public=False,
            pid_file=None,
        )
        argv = captured.get("argv", [])
        assert "--host" in argv
        assert "127.0.0.1" in argv
        assert "--port" in argv
        assert "9120" not in argv  # sanity: defaults are 127.0.0.1:9119
        assert "--no-open" not in argv
        assert "--insecure" not in argv
        assert "--pid-file" not in argv

    def test_full_argv_with_pid_file(self, monkeypatch, tmp_path):
        """All optional flags propagate to the grandchild."""
        pid_file = tmp_path / "custom.pid"
        captured = self._capture_grandchild_argv(
            monkeypatch,
            host="0.0.0.0",
            port=9120,
            open_browser=False,
            allow_public=True,
            pid_file=str(pid_file),
        )
        argv = captured.get("argv", [])
        assert "0.0.0.0" in argv
        assert "9120" in argv
        assert "--no-open" in argv
        assert "--insecure" in argv
        assert "--pid-file" in argv
        assert str(pid_file) in argv
        # And the entry point is the dashboard subcommand.
        assert "dashboard" in argv
        assert "-m" in argv
        assert "hermes_cli.main" in argv

    def test_no_insecure_when_not_requested(self, monkeypatch, tmp_path):
        captured = self._capture_grandchild_argv(
            monkeypatch,
            host="127.0.0.1",
            port=9119,
            open_browser=True,
            allow_public=False,
            pid_file=None,
        )
        argv = captured.get("argv", [])
        assert "--insecure" not in argv
        assert "--no-open" not in argv
        assert "--pid-file" not in argv


# ── cmd_dashboard: precheck on occupied port ───────────────────────────────


class TestDetachPrecheck:
    """`hermes dashboard --detach` must reject an occupied bind address
    before forking so the user sees a fast error."""

    def test_occupied_port_exits_cleanly(self, tmp_path, monkeypatch):
        # Point HERMES_HOME somewhere harmless so the default PID-file
        # path resolution doesn't blow up.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        import argparse
        from hermes_cli.main import cmd_dashboard

        args = argparse.Namespace(
            host="127.0.0.1",
            port=9119,
            no_open=True,
            insecure=False,
            detach=True,
            pid_file=None,
            skip_build=True,  # don't try to build the SPA
            dashboard_subcommand=None,
        )

        # Stub the web UI build check so we don't need a real dist on disk.
        # The function under test is the precheck; the build is incidental.
        monkeypatch.setattr("hermes_cli.main._build_web_ui", lambda *_a, **_kw: True)

        # Pretend 9119 is in use.  Patch the module-level helper
        # rather than the global ``socket`` module — the latter trips
        # up stdlib modules like ``socketserver`` that import
        # ``socket`` at interpreter startup.
        import hermes_cli.main as m

        monkeypatch.setattr(m, "_port_is_in_use", lambda h, p, t=0.25: True)

        # Short-circuit start_server so we don't try to actually bind.
        start_called = {"count": 0}

        def _fake_start_server(**kw):
            start_called["count"] += 1

        monkeypatch.setattr(
            "hermes_cli.web_server.start_server", _fake_start_server
        )

        with mock.patch("builtins.print") as print_mock:
            try:
                cmd_dashboard(args)
            except SystemExit as exc:
                # Real (un-mocked) sys.exit raises SystemExit.  We expect
                # the precheck to call sys.exit(1); accept any non-zero
                # exit code as long as start_server was never reached.
                assert exc.code != 0
        assert start_called["count"] == 0, (
            "start_server should not be called when the precheck rejects the bind"
        )
        printed = " ".join(
            str(c.args[0]) for c in print_mock.call_args_list if c.args
        )
        assert "9119" in printed
        assert "--port" in printed or "--stop" in printed


class _FakeSocket:
    """Minimal socket stand-in for the precheck path.

    connect_ex returns 0 (= connected → port in use) or
    non-zero (= refused → port free)."""

    def __init__(self, connect_returns: int = 1):
        self._connect_returns = connect_returns

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def settimeout(self, _t):
        pass

    def connect_ex(self, _addr):
        return self._connect_returns

    def connect(self, _addr):
        if self._connect_returns == 0:
            return  # success
        raise OSError("refused")

    def close(self):
        pass
