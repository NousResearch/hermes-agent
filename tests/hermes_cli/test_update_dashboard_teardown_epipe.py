"""#132088: the post-update dashboard teardown can kill the update's own stdout reader.

``hermes update`` may run inside a terminal hosted by the serve backend that
``_refresh_dashboard_after_update`` stops. Once that backend dies, every remaining
write raises EPIPE; the teardown must absorb it, record the cleanup step as done,
and leave the fleet matrix / reconciliation / receipt finalize that follow free to
run — instead of aborting mid-report, being misrecorded as a failed cleanup step,
or dying on the dead pipe.

The undisrupted paths (unrecovered PIDs returned for the receipt, kill kwargs)
are covered by ``test_update_sqlite_remediation.py``.
"""

import os
import signal
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd, update_cmd_maint

IS_WINDOWS = sys.platform == "win32"


def _patch_teardown(monkeypatch, kill_behavior):
    calls = {"steps": []}

    def fake_kill(**kwargs):
        return kill_behavior()

    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(
        _kill_stale_dashboard_processes=fake_kill))
    monkeypatch.setattr(
        "hermes_cli.update_cmd._record_update_step",
        lambda step, ok, detail="": calls["steps"].append((step, ok, detail)),
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: "/hermes-test-home")
    return calls


def test_teardown_survives_its_own_dead_stdout(monkeypatch):
    """The stopped backend hosted this update's stdout: the real EPIPE on the next
    write is absorbed, the streams are diverted so later writes succeed, the step
    is recorded as done (not as a failed cleanup), and the verdict stays empty —
    the survivor probe, not the dead pipe, decides what is down."""
    read_fd, write_fd = os.pipe()
    os.close(read_fd)  # the reader (hosted backend) is gone -> EPIPE on every write
    monkeypatch.setattr(sys, "stdout", os.fdopen(write_fd, "w", encoding="utf-8"))
    # A plain writable fd for stderr too, so the diversion's dup2 lands on fds this
    # test owns instead of pytest's capture descriptors.
    monkeypatch.setattr(sys, "stderr",
                        os.fdopen(os.open(os.devnull, os.O_WRONLY), "w", encoding="utf-8"))

    def kill_behavior():
        # First write after the kill: a real flush into the reader-less pipe.
        print("stopping dashboard process(es)", flush=True)
        raise AssertionError("unreachable with a dead reader")

    calls = _patch_teardown(monkeypatch, kill_behavior)

    assert update_cmd_maint._refresh_dashboard_after_update(already_restarted_units=set()) == set()
    assert calls["steps"] == [
        ("dashboard_cleanup", True, "stopped backend hosted this update's stdout")
    ]

    # The streams were diverted: later teardown writes complete instead of raising,
    # so the fleet matrix / reconciliation / receipt finalize that follow can run.
    print("post-teardown write must not raise either", flush=True)


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX signals")
def test_stop_phase_runs_with_sigpipe_ignored(monkeypatch):
    """The phase runs with SIGPIPE ignored: CPython starts with it ignored, but an
    exec/wrapper chain can hand us the default disposition — the phase must restore
    the ignore so it cannot die on the pipe whose reader it is about to kill."""
    observed = {}
    saved = signal.getsignal(signal.SIGPIPE)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # the hostile-but-legal inherited state

    def kill_behavior():
        observed["sigpipe"] = signal.getsignal(signal.SIGPIPE)
        return {"unrecovered": []}

    try:
        _patch_teardown(monkeypatch, kill_behavior)
        assert update_cmd_maint._refresh_dashboard_after_update(
            already_restarted_units=set()) == set()
    finally:
        signal.signal(signal.SIGPIPE, saved)

    assert observed["sigpipe"] is signal.SIG_IGN


def test_real_cleanup_failures_are_still_isolated(monkeypatch, capsys):
    """Unchanged path: a genuine failure is still isolated and recorded as a failed
    step — the new EPIPE branch must not swallow real errors."""

    def kill_behavior():
        raise RuntimeError("scan broke")

    calls = _patch_teardown(monkeypatch, kill_behavior)

    assert update_cmd_maint._refresh_dashboard_after_update(already_restarted_units=set()) == set()
    assert calls["steps"] == [("dashboard_cleanup", False, "RuntimeError: scan broke")]
    assert "Could not refresh running dashboard/serve process(es)" in capsys.readouterr().out
