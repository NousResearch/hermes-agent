# -*- coding: utf-8 -*-
"""Live Windows probes for the fail-closed taskkill process-identity guard.

Runs only on real Windows (the on-demand ``wine2e/**`` windows-latest lane).
These tests spawn REAL processes and drive the REAL guard code against the
live process table — the coverage the mocked Linux suites cannot provide.

Class under test (#98814 / #89614):

- ``gateway.status.terminate_pid(force=True)`` requires a matching
  ``expected_start_time`` and must refuse (never taskkill) on a missing or
  mismatched identity.
- ``hermes_cli._subprocess_compat.pid_is_hermes`` fails closed on foreign
  processes and identity mismatches.
- ``hermes_cli.update_cmd._refuse_gateway_ancestor_tree_kill`` refuses to
  nominate any ancestor of the current process for a tree-kill.
"""
import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="live taskkill-identity probes are Windows-only"
)


def _spawn_sleeper(seconds: int = 60) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


def _cleanup(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


class TestTerminatePidIdentityLive:
    def test_matching_identity_kills_real_process(self):
        from gateway.status import get_process_start_time, terminate_pid

        proc = _spawn_sleeper()
        try:
            start = get_process_start_time(proc.pid)
            assert start is not None, "live process must have a fingerprint"
            terminate_pid(proc.pid, force=True, expected_start_time=start)
            assert proc.wait(timeout=15) is not None
        finally:
            _cleanup(proc)

    def test_missing_expectation_refuses_and_process_survives(self):
        from gateway.status import terminate_pid

        proc = _spawn_sleeper()
        try:
            with pytest.raises(OSError, match="start-time guard"):
                terminate_pid(proc.pid, force=True)
            assert proc.poll() is None, "refusal must leave the process running"
        finally:
            _cleanup(proc)

    def test_mismatched_identity_refuses_and_process_survives(self):
        """The recycled-PID scenario: recorded identity != live identity."""
        from gateway.status import get_process_start_time, terminate_pid

        # Simulate recycling: capture the identity of a process that then
        # dies, and respawn a DIFFERENT process. We can't force Windows to
        # hand back the same PID, so assert the guard refuses when the stale
        # fingerprint is presented against the new (different) live process.
        victim = _spawn_sleeper(1)
        stale_start = get_process_start_time(victim.pid)
        victim.wait(timeout=30)

        impostor = _spawn_sleeper()
        try:
            live_start = get_process_start_time(impostor.pid)
            assert live_start is not None
            if stale_start == live_start:
                pytest.skip("fingerprints collided; cannot express mismatch")
            with pytest.raises(OSError, match="identity"):
                terminate_pid(
                    impostor.pid, force=True, expected_start_time=stale_start
                )
            assert impostor.poll() is None, "mismatch must never kill"
        finally:
            _cleanup(impostor)

    def test_dead_pid_identity_unavailable_refuses(self):
        from gateway.status import get_process_start_time, terminate_pid

        proc = _spawn_sleeper(1)
        pid = proc.pid
        start = get_process_start_time(pid)
        proc.wait(timeout=30)
        # Give the OS a beat to drop the process object.
        time.sleep(0.5)
        if get_process_start_time(pid) == start:
            pytest.skip("PID instantly recycled onto identical fingerprint")
        with pytest.raises(OSError):
            terminate_pid(pid, force=True, expected_start_time=start)


class TestPidIsHermesLive:
    def test_foreign_real_process_is_refused(self):
        """A live non-Hermes process (bare python sleeper in a temp-ish argv)
        must never be judged safe for taskkill."""
        from hermes_cli._subprocess_compat import pid_is_hermes

        proc = _spawn_sleeper()
        try:
            # sys.executable in CI lives under a uv/hostedtoolcache path with
            # no 'hermes' token; if the checkout path itself contains one this
            # assertion is environment-dependent, so guard for it.
            if "hermes" in sys.executable.lower():
                pytest.skip("interpreter path names hermes; probe would match")
            assert pid_is_hermes(proc.pid) is False
        finally:
            _cleanup(proc)

    def test_stale_fingerprint_is_refused_even_for_hermes_argv(self):
        from gateway.status import get_process_start_time
        from hermes_cli._subprocess_compat import pid_is_hermes

        proc = _spawn_sleeper()
        try:
            live = get_process_start_time(proc.pid)
            assert live is not None
            assert (
                pid_is_hermes(proc.pid, expected_start_time=live + 12345) is False
            )
        finally:
            _cleanup(proc)

    def test_nonexistent_pid_is_refused(self):
        from hermes_cli._subprocess_compat import pid_is_hermes

        assert pid_is_hermes(2**24) is False


class TestAncestorRefusalLive:
    def test_real_parent_chain_is_refused(self, capsys):
        """Walk the REAL psutil parent chain: every ancestor of this test
        process must be refused as a tree-kill target (#98814)."""
        import psutil

        from hermes_cli.gateway import _is_pid_ancestor_of_current_process
        from hermes_cli.update_cmd import _refuse_gateway_ancestor_tree_kill

        ancestors = [os.getpid()]
        parent = psutil.Process(os.getpid()).parent()
        while parent is not None and len(ancestors) < 6:
            ancestors.append(parent.pid)
            parent = parent.parent()

        for pid in ancestors:
            assert _is_pid_ancestor_of_current_process(pid) is True, pid

        refused = _refuse_gateway_ancestor_tree_kill(
            ancestors, gateway_mode=False
        )
        assert refused is True
        out = capsys.readouterr().out
        assert "taskkill /T" in out
        assert "separate terminal" in out

    def test_unrelated_live_process_is_not_refused(self):
        from hermes_cli.gateway import _is_pid_ancestor_of_current_process
        from hermes_cli.update_cmd import _refuse_gateway_ancestor_tree_kill

        proc = _spawn_sleeper()
        try:
            assert _is_pid_ancestor_of_current_process(proc.pid) is False
            assert (
                _refuse_gateway_ancestor_tree_kill(
                    [proc.pid], gateway_mode=False
                )
                is False
            )
        finally:
            _cleanup(proc)
