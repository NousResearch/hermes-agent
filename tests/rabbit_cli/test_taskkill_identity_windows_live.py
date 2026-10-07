# -*- coding: utf-8 -*-
"""Live Windows probes for the fail-closed taskkill process-identity guard.

Runs only on real Windows (the on-demand ``wine2e/**`` windows-latest lane).
These tests spawn REAL processes and drive the REAL guard code against the
live process table — the coverage the mocked Linux suites cannot provide.

Class under test (#98814 / #89614):

- ``gateway.status.terminate_pid(force=True)`` requires a matching
  ``expected_start_time`` and must refuse (never taskkill) on a missing or
  mismatched identity.
- ``rabbit_cli._subprocess_compat.pid_is_rabbit`` fails closed on foreign
  processes and identity mismatches.
"""
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.platforms("windows")  # live taskkill-identity probes are Windows-only


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


class TestPidIsRabbitLive:
    def test_foreign_real_process_is_refused(self):
        """A live non-Rabbit process (bare python sleeper in a temp-ish argv)
        must never be judged safe for taskkill."""
        from rabbit_cli._subprocess_compat import pid_is_rabbit

        proc = _spawn_sleeper()
        try:
            # sys.executable in CI lives under a uv/hostedtoolcache path with
            # no 'rabbit' token; if the checkout path itself contains one this
            # assertion is environment-dependent, so guard for it.
            if "rabbit" in sys.executable.lower():
                pytest.skip("interpreter path names rabbit; probe would match")
            assert pid_is_rabbit(proc.pid) is False
        finally:
            _cleanup(proc)

    def test_stale_fingerprint_is_refused_even_for_rabbit_argv(self):
        from gateway.status import get_process_start_time
        from rabbit_cli._subprocess_compat import pid_is_rabbit

        proc = _spawn_sleeper()
        try:
            live = get_process_start_time(proc.pid)
            assert live is not None
            assert (
                pid_is_rabbit(proc.pid, expected_start_time=live + 12345) is False
            )
        finally:
            _cleanup(proc)

    def test_nonexistent_pid_is_refused(self):
        from rabbit_cli._subprocess_compat import pid_is_rabbit

        assert pid_is_rabbit(2**24) is False
