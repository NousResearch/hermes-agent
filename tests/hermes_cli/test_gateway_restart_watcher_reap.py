"""Reap semantics of the ``--replace`` detached restart watcher (#73480).

``hermes_cli.gateway._spawn_gateway_restart_watcher`` builds a tiny Python
watcher *as a source string* and launches it detached.  The watcher polls the
outgoing gateway's PID and then respawns the gateway.

The bug: the respawn used to be a bare ``subprocess.Popen(cmd, ...)`` whose
handle was dropped, after which the watcher script simply ran off the end and
exited.  The replacement gateway therefore lost its parent the instant it
started and was reparented to PID 1.  On a systemd-supervised host that orphan
lives *outside* the unit's cgroup: systemd never tracks it, never stops it and
never reaps it, so it races the supervised gateway for the duplicate-instance
guard and keeps platform sessions (Telegram's ``getUpdates`` long poll, in the
filed incident) open indefinitely.

Because the watcher is only a string, it can be exercised end-to-end in-process
against a short-lived stub child — no gateway, no systemd, no privileges.
"""

import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

import hermes_cli.gateway as gateway

# Stand-in for the respawned gateway.  Announces itself, blocks until the test
# releases it, then records who its parent is at exit time.
REPLACEMENT_SOURCE = """
import os
import pathlib
import sys
import time

started = pathlib.Path(sys.argv[1])
release = pathlib.Path(sys.argv[2])
parent_record = pathlib.Path(sys.argv[3])

started.write_text(str(os.getpid()), encoding="utf-8")
deadline = time.monotonic() + 60
while time.monotonic() < deadline and not release.exists():
    time.sleep(0.02)
parent_record.write_text(str(os.getppid()), encoding="utf-8")
"""


def _generated_watcher_source(old_pid: int, run_argv: list[str]) -> str:
    """Return the watcher source ``_spawn_gateway_restart_watcher`` builds.

    The real function is called (so the source under test is the real one,
    including its ``.format()`` escaping); only the detached launch is
    intercepted, so the test can run the watcher itself under its own control.
    """
    captured = {}

    def _fake_popen(argv, **kwargs):
        captured["argv"] = argv
        return mock.MagicMock()

    with mock.patch.object(gateway.subprocess, "Popen", _fake_popen):
        assert gateway._spawn_gateway_restart_watcher(old_pid, list(run_argv)) is True

    watcher_argv = captured["argv"]
    # [sys.executable, "-c", <watcher source>, str(old_pid), *run_argv]
    assert watcher_argv[1] == "-c"
    assert watcher_argv[3] == str(old_pid)
    assert watcher_argv[4:] == list(run_argv)
    return watcher_argv[2]


def _reaped_pid() -> int:
    """A PID that has already exited *and been reaped*, so it no longer exists.

    The watcher's poll loop must fall through immediately rather than burn its
    120s deadline.
    """
    finished = subprocess.Popen([sys.executable, "-c", ""])
    finished.wait()
    return finished.pid


def _wait_for(path: Path, message: str, timeout: float = 20.0) -> None:
    """Block until ``path`` appears.

    The watcher breaks out of its poll loop as soon as it sees the (already
    reaped) old PID is gone, so the replacement appears in well under a
    second; 20s is a large margin for a loaded runner while still surfacing a
    genuine regression quickly.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(message)


class TestRestartWatcherReapsReplacement:
    """The generated watcher must stay in the replacement's ``waitpid`` chain."""

    def test_generated_watcher_source_is_valid_python(self):
        """The watcher body is interpolated through ``str.format()``.

        Every literal brace in it has to be doubled; a single missed ``{``
        yields either a ``KeyError`` at build time or syntactically broken
        source that only fails once a user's gateway tries to restart.
        """
        source = _generated_watcher_source(
            _reaped_pid(), [sys.executable, "-c", "pass"]
        )
        compile(source, "<restart-watcher>", "exec")

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="asserts POSIX waitpid-chain/reparenting semantics",
    )
    def test_watcher_waits_for_and_reaps_a_short_lived_replacement(self, tmp_path):
        """Regression guard for #73480.

        Before the fix the watcher dropped the ``Popen`` handle and exited as
        soon as the replacement was spawned, so the replacement was orphaned to
        PID 1.  The watcher must instead remain the replacement's parent for
        its whole lifetime and collect its exit status.
        """
        started = tmp_path / "replacement-started"
        release = tmp_path / "replacement-release"
        parent_record = tmp_path / "replacement-parent"

        run_argv = [
            sys.executable,
            "-c",
            REPLACEMENT_SOURCE,
            str(started),
            str(release),
            str(parent_record),
        ]
        old_pid = _reaped_pid()
        source = _generated_watcher_source(old_pid, run_argv)

        watcher = subprocess.Popen(
            [sys.executable, "-c", source, str(old_pid), *run_argv],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for(started, "the watcher never spawned the replacement")

            # The replacement is alive; the watcher must not have exited.
            with pytest.raises(subprocess.TimeoutExpired):
                watcher.wait(timeout=5)

            release.write_text("go", encoding="utf-8")

            # wait() returning is exactly "the child was reaped by this
            # process" — the watcher only gets here once it has collected the
            # replacement's exit status.
            assert watcher.wait(timeout=30) == 0
        finally:
            if not release.exists():
                release.write_text("go", encoding="utf-8")
            if watcher.poll() is None:
                watcher.kill()
                watcher.wait(timeout=30)

        replacement_pid = int(started.read_text(encoding="utf-8").strip())
        assert replacement_pid > 0
        assert replacement_pid != watcher.pid

        # The parent the replacement saw at exit is the watcher itself, not
        # PID 1 — i.e. it never left the waitpid chain.
        assert parent_record.read_text(encoding="utf-8").strip() == str(watcher.pid)
