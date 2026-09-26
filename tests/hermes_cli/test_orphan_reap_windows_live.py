"""LIVE Windows E2E for the Desktop boot orphan reap (#122533).

Runs ONLY on a real Windows host (the on-demand ``windows-venv-e2e.yml``
lane). No mocked psutil, no faked cmdlines: real processes with real argv
shapes are spawned and the real scan/classify/reap code is driven against
the live Windows process table.

The bug class (#122533): the Desktop-owned ``serve`` lifespan sweeps
``_reap_unsupervised_gateway_orphans()`` at boot. A gateway only claims
``gateway.pid``/``gateway.lock`` after imports and runner setup, so for its
first seconds it is scan-visible but record-visible to nobody — the argv
sweep cannot tell it from a corpse, marks it with a ``planned-stop``
marker, and its own planned-stop watcher consumes that self-targeting
marker as a deliberate stop. The fix gives the Desktop boot call site a
startup grace (``min_age_s``) so a still-booting gateway is spared.

Each test pins the CORRECT behavior, so on unfixed main the buggy ones
fail — that failure on the Windows runner is the empirical premise-check.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.platforms("windows"),
    # The spawned lookalikes carry a "gateway run" argv tail as inert data
    # (that tail is exactly what the real matcher classifies on); every
    # child is killed by the test.
    pytest.mark.spawns_gateway_lookalike,
    # This suite's whole point is the REAL reap path delivering a REAL stop
    # to a REAL process outside the test's own subtree, so the live-system
    # guard must be bypassed deliberately. It is safe here because every
    # process the sweep can touch is a lookalike this file spawned, and each
    # test reaps only while the host has no gateway running.
    pytest.mark.live_system_guard_bypass,
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The canonical argv tail the gateway matcher accepts, as inert data to a
# sleeper child but fully visible to the Windows process scan.
_LOOKALIKE_TAIL = ["hermes_cli.main", "gateway", "run"]


def _spawn_lookalike() -> subprocess.Popen:
    """A real process whose cmdline matches the gateway argv matcher."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)", *_LOOKALIKE_TAIL],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.8)  # let the process table settle
    assert proc.poll() is None, "sleeper died at spawn"
    return proc


def _spawn_orphan():
    """A gateway-lookalike whose launcher parent is GONE.

    This is the shape a booting Desktop-launched gateway has: a short-lived
    launcher that spawned it and exited, so the parent chain breaks before
    reaching a real supervisor and the candidate classifies as UNSUPERVISED.
    """
    launcher_src = (
        "import subprocess, sys, time;"
        f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)',"
        f" {', '.join(repr(a) for a in _LOOKALIKE_TAIL)}], creationflags=0x08000000);"
        "time.sleep(0.3)"
    )
    launcher = subprocess.Popen(
        [sys.executable, "-c", launcher_src],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    launcher.wait(timeout=30)
    time.sleep(1.0)  # let the dead PPID settle in the process table
    return _find_lookalike()


def _find_lookalike():
    """Re-attach to the orphaned lookalike by scanning for its argv tail."""
    import psutil

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
        except Exception:  # noqa: BLE001 — a process can vanish mid-scan
            continue
        if "time.sleep(300)" in cmd and all(tail in cmd for tail in _LOOKALIKE_TAIL):
            return proc
    raise AssertionError("orphaned lookalike not found in the live process table")


def _kill(*procs) -> None:
    import psutil

    for proc in procs:
        try:
            pid = proc if isinstance(proc, int) else proc.pid
            if psutil.pid_exists(pid):
                psutil.Process(pid).terminate()
                psutil.Process(pid).wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass


def _marker_path():
    from gateway.status import _get_planned_stop_marker_path

    return _get_planned_stop_marker_path()


def _marker_names(pid: int) -> bool:
    marker = _marker_path()
    if not marker.exists():
        return False
    return f'"target_pid":{pid}' in marker.read_text(encoding="utf-8").replace(" ", "")


def _clear_marker() -> None:
    marker = _marker_path()
    if marker.exists():
        marker.unlink()


class TestDesktopBootReapGrace:
    def test_boot_sweep_spares_a_booting_gateway(self):
        """LEG A: the Desktop boot sweep must NOT reap a gateway that is
        still starting up — the reported outage, pinned as correct behavior.

        On unfixed main this fails: the sweep marks and kills the candidate.
        """
        from hermes_cli.dashboard_procs import _REAP_MIN_AGE_SECONDS
        from hermes_cli.gateway import (
            _reaper_candidate_is_supervisor_owned,
            _reap_unsupervised_gateway_orphans,
            _scan_gateway_pids,
        )

        proc = _spawn_orphan()
        pid = proc.pid
        try:
            # Premise: the live process table sees it and the classifier agrees.
            assert pid in _scan_gateway_pids(set()), "live scan missed the orphaned lookalike"
            assert not _reaper_candidate_is_supervisor_owned(pid), (
                "expected an UNSUPERVISED orphan (dead launcher parent)"
            )
            _clear_marker()
            # On UNFIXED main the reaper has no min_age_s parameter at all, so
            # this falls back to the bare call — and the test then fails for the
            # real reason: the booting gateway was reaped. That keeps the
            # RED proof behavioral (the process died) instead of a TypeError.
            grace: dict[str, float] = (
                {"min_age_s": _REAP_MIN_AGE_SECONDS}
                if "min_age_s" in inspect.signature(_reap_unsupervised_gateway_orphans).parameters
                else {}
            )
            reaped = _reap_unsupervised_gateway_orphans(**grace)
            assert not reaped, f"boot sweep reaped a still-booting gateway: {reaped}"
            assert not _marker_names(pid), "boot sweep wrote a planned-stop marker for it"
            import psutil

            assert psutil.pid_exists(pid), "boot sweep terminated a still-booting gateway"
        finally:
            _clear_marker()
            _kill(proc)

    def test_no_grace_sweep_still_reaps_the_orphan(self):
        """The no-grace default must keep reaping — the startup grace is
        opt-in and must not atrophy the immediate stop/restart reap."""
        from hermes_cli.gateway import _reap_unsupervised_gateway_orphans

        proc = _spawn_orphan()
        pid = proc.pid
        try:
            _clear_marker()
            reaped = _reap_unsupervised_gateway_orphans()
            assert reaped, "the no-grace sweep stopped reaping unsupervised orphans"
            assert _marker_names(pid), "no planned-stop marker was written for the reaped orphan"
        finally:
            _clear_marker()
            _kill(proc)
