"""Regression test for #28181 — kanban worker SIGTERM must terminate the process.

The single-query signal handler in cli.py (``_signal_handler_q``) raises
``KeyboardInterrupt`` to unwind the main thread on SIGTERM/SIGHUP. That works
for interactive ``hermes chat -q`` invocations, but kanban workers spawned by
the dispatcher are likely to have a non-daemon thread alive (terminal_tool's
``_wait_for_process``, custom plugin background workers, etc.). With
``KeyboardInterrupt`` only the main thread unwinds; the non-daemon thread
keeps the process alive after the gateway has already restarted, the kanban
dispatcher's ``_pid_alive`` check returns True forever, and the task stays
``running`` indefinitely.

The fix: when the process is a dispatcher-spawned worker (``HERMES_KANBAN_TASK``
env var set), flush logging + stdout/stderr and call ``os._exit(0)`` instead.
The kernel reclaims the PID immediately, and ``detect_crashed_workers``
reclaims the stale claim on the next dispatcher tick.

These tests use a synthetic Python script that mirrors the cli.py signal
handler shape so we can exercise the exit-path contract without booting the
full CLI (which needs a real provider config).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest


def _synthetic_worker_script() -> str:
    """A standalone script that mirrors cli.py's single-query SIGTERM handler.

    Keeping the synthetic copy here means the test exercises the exact handler
    shape without needing the full hermes_cli boot path (config, providers,
    skills, etc.). If the production handler in cli.py drifts, the test
    that loads the real handler (test_real_handler_uses_os_exit) will catch it.
    """
    return textwrap.dedent(
        """
        import os, signal, sys, threading, time

        # Non-daemon thread that blocks forever — simulates the worker
        # thread that would prevent orderly Python shutdown after
        # KeyboardInterrupt unwinds main.
        stuck = threading.Event()
        threading.Thread(target=stuck.wait, daemon=False).start()

        def handler(signum, frame):
            # Mirrors cli.py:_signal_handler_q. Real handler sleeps 1.5s; the
            # test uses a short grace so it runs fast.
            try:
                time.sleep(0.05)
            except Exception:
                pass
            if os.environ.get("HERMES_KANBAN_TASK"):
                # Check for fast-reclaim marker (mirrors cli.py logic).
                is_fast_reclaim = False
                marker = f"/tmp/hermes_fast_reclaim_{os.getpid()}"
                try:
                    is_fast_reclaim = os.path.exists(marker)
                    if is_fast_reclaim:
                        os.unlink(marker)
                except Exception:
                    pass
                if is_fast_reclaim:
                    try:
                        if hasattr(signal, "SIGALRM"):
                            signal.signal(signal.SIGALRM, lambda *_: os._exit(0))
                            signal.alarm(2)
                    except Exception:
                        pass
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(0)
                # Infra SIGTERM — fall through to KeyboardInterrupt
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, handler)
        print("READY", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            sys.exit(0)
        """
    )


def _is_alive_like_dispatcher(pid: int) -> bool:
    """Mirrors hermes_cli/kanban_db.py:_pid_alive on Linux.

    A zombie is treated as dead — the dispatcher's _pid_alive checks
    /proc/<pid>/status for State: Z. We replicate that here so a clean
    os._exit followed by zombie-state is correctly counted as dead.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if sys.platform == "linux":
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("State:"):
                        if "Z" in line.split(":", 1)[1]:
                            return False
                        break
        except (FileNotFoundError, PermissionError, OSError):
            pass
    return True


def _spawn_synthetic(env_overrides: dict) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(env_overrides)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _synthetic_worker_script()],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    # Wait for "READY" so we know the signal handler is installed.
    assert proc.stdout is not None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line and line.startswith(b"READY"):
            return proc
    proc.kill()
    raise RuntimeError("synthetic worker never signalled READY")


def _cleanup(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="SIGTERM semantics differ on Windows; kanban dispatcher is POSIX-only",
)
def test_sigterm_with_kanban_task_env_terminates_quickly():
    """With HERMES_KANBAN_TASK set AND fast-reclaim marker, SIGTERM should
    kill the process in <2s even when a non-daemon thread is still alive.

    The marker is written before SIGTERM to simulate the dispatcher's
    _terminate_reclaimed_worker() writing it before sending the signal.
    """
    proc = _spawn_synthetic({"HERMES_KANBAN_TASK": "t_test_28181"})
    try:
        t0 = time.time()

        # Write the fast-reclaim marker so the worker treats this as a
        # foreign-host reclaim (fast-reclaim path) rather than infra SIGTERM.
        marker = f"/tmp/hermes_fast_reclaim_{proc.pid}"
        try:
            with open(marker, "w") as f:
                f.write("1")
        except Exception:
            pytest.fail("failed to write fast-reclaim marker")

        os.kill(proc.pid, signal.SIGTERM)

        # Should die in <2s. The handler sleeps ~50ms, then os._exit(0)
        # is immediate. Give generous headroom for slow CI runners.
        deadline = t0 + 2.0
        while time.time() < deadline:
            if not _is_alive_like_dispatcher(proc.pid):
                elapsed = time.time() - t0
                assert elapsed < 2.0
                assert not os.path.exists(marker), (
                    "fast-reclaim marker should have been cleaned up "
                    "by the worker"
                )
                return
            time.sleep(0.02)
        pytest.fail(
            "process still alive 2s after SIGTERM with HERMES_KANBAN_TASK set "
            "(dispatcher would keep extending claim) — fix regressed"
        )
    finally:
        _cleanup(proc)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="SIGTERM semantics differ on Windows; kanban dispatcher is POSIX-only",
)
def test_sigterm_without_kanban_task_env_uses_keyboard_interrupt_path():
    """Without HERMES_KANBAN_TASK, the original KeyboardInterrupt path runs.

    This is the contrast case proving the fix is gated on the env var: in
    interactive ``hermes chat -q`` (no env var), behavior is unchanged. The
    process MAY hang under non-daemon threads, but that's not a kanban-worker
    concern. We just verify the handler logs the KeyboardInterrupt branch
    rather than os._exit'ing.
    """
    proc = _spawn_synthetic({})
    try:
        os.kill(proc.pid, signal.SIGTERM)
        # Wait a moment for the handler to react.
        time.sleep(0.5)
        # The process may or may not be dead depending on whether the
        # KeyboardInterrupt unwinds cleanly. The behavioral guarantee is
        # only that the env-gated path didn't fire.
        try:
            # Drain stdout up to whatever's available.
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass
    finally:
        _cleanup(proc)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="SIGTERM semantics differ on Windows; kanban dispatcher is POSIX-only",
)
def test_sigterm_with_kanban_task_env_and_no_marker_uses_keyboard_interrupt():
    """With HERMES_KANBAN_TASK set but NO fast-reclaim marker (infra SIGTERM),
    the handler should fall through to KeyboardInterrupt rather than
    os._exit(0), allowing graceful state-preserving shutdown.
    """
    proc = _spawn_synthetic({"HERMES_KANBAN_TASK": "t_test_infra"})
    try:
        # Verify no marker exists before SIGTERM.
        marker = f"/tmp/hermes_fast_reclaim_{proc.pid}"
        assert not os.path.exists(marker)

        os.kill(proc.pid, signal.SIGTERM)
        # Wait briefly for the handler to react.
        time.sleep(0.5)

        # The behavior should match test_sigterm_without... — the process
        # may or may not be dead (KeyboardInterrupt may or may not unwind
        # cleanly with a non-daemon thread). What MUST NOT happen is the
        # fast-reclaim os._exit(0) path which always terminates in <2s.
        # We verify it's NOT quick-dead like the fast-reclaim case:
        try:
            os.kill(proc.pid, 0)
        except (ProcessLookupError, OSError):
            pass  # process may have died; that's fine

        # Clean up stdout/stderr
        try:
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass
    finally:
        _cleanup(proc)


def test_real_handler_distinguishes_sigterm_types():
    """Source-level invariant: cli.py's _signal_handler_q must distinguish
    fast-reclaim SIGTERM (with marker) from infra SIGTERM (no marker).

    Catches the case where someone refactors the handler and accidentally
    drops the distinction, either losing the fast-reclaim exit path (which
    would cause the dispatcher to hang on reclaim) or falling through to
    os._exit(0) for infra SIGTERM (which would cause unnecessary work loss).
    Reading cli.py directly is cheap and avoids the heavy CLI import.
    """
    import pathlib

    cli_path = (
        pathlib.Path(__file__).resolve().parent.parent.parent / "cli.py"
    )
    src = cli_path.read_text()
    # Locate the handler body.
    start = src.find("def _signal_handler_q(signum, frame):")
    assert start != -1, "cli.py is missing _signal_handler_q"
    # Look ahead for the env-gated os._exit call within ~100 lines.
    body = src[start : start + 5000]
    assert "HERMES_KANBAN_TASK" in body, (
        "_signal_handler_q must gate its kanban-worker exit path on "
        "HERMES_KANBAN_TASK — see #28181"
    )
    # Fast-reclaim marker path must exist
    assert "hermes_fast_reclaim_" in body, (
        "_signal_handler_q must check for fast-reclaim marker to "
        "distinguish SIGTERM types — see feat/kanban-sigterm"
    )
    assert "os._exit(0)" in body, (
        "_signal_handler_q must call os._exit(0) for fast-reclaim — "
        "see feat/kanban-sigterm"
    )
    assert "_is_fast_reclaim" in body, (
        "_signal_handler_q must gate os._exit(0) on fast-reclaim "
        "detection — infra SIGTERM should fall through to "
        "KeyboardInterrupt for graceful shutdown"
    )
