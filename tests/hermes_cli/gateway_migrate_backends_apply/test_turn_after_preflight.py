"""A chat turn that starts after the preflight, with the real turn-marker reader and writer.

The old backend is a real child process; when the apply installs the gateway (after the preflight read no
turn) the test tells it to start a turn, and it records the marker with ``tui_gateway.turn_marker`` (pid and
start time are its own). The apply reads the marker again right before the stop: without --force-interrupt
the move rolls back (exit 4) and the old backend is never stopped; an unreadable marker file at that point
counts as unknown activity and stops the move too; with --force-interrupt the move goes ahead.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli.gateway_migrate_backends import BackendItem
from hermes_cli.gateway_migrate_backends_host import SystemHost
from tests.hermes_cli.test_gateway_migrate_backends_apply import FakeHost, _args
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("linux", "macos")

_REPO = Path(__file__).resolve().parents[3]
_BACKEND = ("import sys, pathlib; from tests.waiting import wait_until; from tui_gateway import turn_marker\n"
            "home, go = sys.argv[1], pathlib.Path(sys.argv[2])\n"
            "wait_until(go.exists, timeout=60)\n"
            "turn_marker.record_turn_start(home, 'session-1', 'a reply is being written')\n"
            "(go.parent / 'turn-started').write_text('')\n"
            "wait_until((go.parent / 'stop').exists, timeout=120)\n")


@pytest.fixture
def backend(home, tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", _BACKEND, str(home), str(tmp_path / "go")], cwd=_REPO,
                            env={**os.environ, "PYTHONPATH": str(_REPO), "HERMES_HOME": str(home)})
    yield proc, tmp_path
    (tmp_path / "stop").write_text("", encoding="utf-8")
    proc.wait(timeout=60)


class RealTurnsHost(FakeHost):
    """The engine tests' fake host, but the old backend is the child process and its turns are read by
    the real marker reader."""

    def __init__(self, home, proc, signals, *, after=None):
        super().__init__(home, supervisor="manual")
        self.proc, self.signals, self.after = proc, signals, after

    def backends(self):
        return [BackendItem(**{**item.__dict__, "pid": self.proc.pid}) if item.pid else item
                for item in super().backends()]

    running_turns = SystemHost.running_turns

    def same_process(self, pid, start_time):
        return pid == self.proc.pid and self.old["active"]

    def install_gateway(self):
        super().install_gateway()
        (self.signals / "go").write_text("", encoding="utf-8")
        wait_until((self.signals / "turn-started").exists, timeout=60, what="the turn to start")
        if self.after:
            self.after()


def test_a_real_turn_started_after_preflight_rolls_back_with_the_old_backend_untouched(home, backend, capsys):
    proc, signals = backend
    host = RealTurnsHost(home, proc, signals)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK
    out = capsys.readouterr().out
    assert "a turn started on the old backend during migration" in out
    assert "old_stop" not in host.calls and host.old["active"] is True
    assert proc.poll() is None  # the old backend process still runs


def test_unreadable_turn_activity_after_preflight_also_stops_the_move(home, backend, capsys):
    """The marker file cannot be read when the apply checks again: unknown is not idle."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root reads a mode-000 file")
    from tui_gateway import turn_marker
    proc, signals = backend
    marker = turn_marker._marker_path(home)
    host = RealTurnsHost(home, proc, signals, after=lambda: marker.chmod(0o000))
    try:
        code = engine.cmd_apply(host, _args())
    finally:
        if marker.exists():
            marker.chmod(0o600)

    out = capsys.readouterr().out
    assert code == engine.EXIT_ROLLED_BACK, out
    assert "turn activity cannot be read" in out
    assert "old_stop" not in host.calls and host.old["active"] is True and proc.poll() is None


def test_with_force_interrupt_a_turn_started_after_preflight_does_not_stop_the_move(home, backend):
    proc, signals = backend
    host = RealTurnsHost(home, proc, signals)

    assert engine.cmd_apply(host, _args(force_interrupt=True)) == engine.EXIT_OK
    assert "old_stop" in host.calls
