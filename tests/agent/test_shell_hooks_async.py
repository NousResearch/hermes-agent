"""Tests for async shell hooks.

Covers the async-hook bridge added on top of PR #13143:
  * config-parse validation tiers (accepted / warned / rejected)
  * fire-and-forget callback returns immediately
  * the bounded semaphore drops firings on saturation
  * shutdown path terminates running subprocesses within the grace
    window and SIGKILLs survivors
  * CLI SIGINT handler terminates tracked subprocesses
  * Popen-refactor regression gate (in test_shell_hooks.py)
  * async pre_tool_call block directives are logged and ignored
  * claude-code runtime ``{"async": true}`` marker on a sync hook is
    warned and ignored
  * thread-safety of _live_futures under concurrent firings

All tests share an autouse teardown fixture that drains the async pool
state so module-level singletons never leak across cases.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agent import shell_hooks


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_async_state():
    """Drop every singleton / set the async pool tracks so tests with
    different pool sizes, graces, or signal dispositions don't share
    state."""
    yield
    shell_hooks._reset_async_pool()
    shell_hooks.reset_for_tests()


@pytest.fixture
def stub_async_config(monkeypatch):
    """Helper: drive _async_config() without round-tripping the real
    config loader."""
    def _apply(pool_size: int = 10, grace: int = 5) -> None:
        monkeypatch.setattr(
            shell_hooks, "_async_config", lambda: (pool_size, grace),
        )
    return _apply


def _write_script(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    path.chmod(0o755)
    return path


# ── config-parse validation tiers ────────────────────────────────────────


class TestAsyncConfigValidation:
    def test_accepted_event_registers_as_async(self):
        specs = shell_hooks._parse_hooks_block({
            "post_tool_call": [
                {"command": "/tmp/a.sh", "async": True},
            ],
        })
        assert len(specs) == 1
        assert specs[0].is_async is True

    def test_rejected_event_skipped(self, caplog):
        with caplog.at_level(
            logging.ERROR, logger=shell_hooks.logger.name,
        ):
            specs = shell_hooks._parse_hooks_block({
                "pre_llm_call": [{"command": "/tmp/x.sh", "async": True}],
            })
        assert specs == []
        assert any(
            "cannot use async: true" in r.getMessage()
            and "pre_llm_call" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.parametrize(
        "event", ["transform_tool_result", "transform_terminal_output"],
    )
    def test_transform_events_rejected(self, event, caplog):
        with caplog.at_level(
            logging.ERROR, logger=shell_hooks.logger.name,
        ):
            specs = shell_hooks._parse_hooks_block({
                event: [{"command": "/tmp/x.sh", "async": True}],
            })
        assert specs == []

    def test_pre_tool_call_async_warns_but_accepts(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger=shell_hooks.logger.name,
        ):
            specs = shell_hooks._parse_hooks_block({
                "pre_tool_call": [
                    {"command": "/tmp/audit.sh", "async": True},
                ],
            })
        assert len(specs) == 1 and specs[0].is_async is True
        assert any(
            "block directives will be ignored" in r.getMessage()
            for r in caplog.records
        )

    def test_non_bool_async_defaults_to_sync(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger=shell_hooks.logger.name,
        ):
            specs = shell_hooks._parse_hooks_block({
                "post_tool_call": [
                    {"command": "/tmp/x.sh", "async": "sure"},
                ],
            })
        assert len(specs) == 1 and specs[0].is_async is False


# ── fire-and-forget behaviour ────────────────────────────────────────────


class TestAsyncReturnsImmediately:
    def test_async_hook_returns_immediately(
        self, tmp_path, stub_async_config,
    ):
        stub_async_config(pool_size=4, grace=5)
        sentinel = tmp_path / "done.marker"
        # Script sleeps, then writes the sentinel.
        script = _write_script(
            tmp_path, "slow.sh",
            f"#!/usr/bin/env bash\nsleep 1\ntouch {sentinel}\nprintf '{{}}\\n'\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="post_tool_call", command=str(script),
            timeout=30, is_async=True,
        )
        cb = shell_hooks._make_callback(spec)

        t0 = time.monotonic()
        assert cb(tool_name="terminal") is None
        elapsed = time.monotonic() - t0
        assert elapsed < 0.25, (
            f"async callback blocked {elapsed:.3f}s — should be ~0"
        )
        # Future is pending
        with shell_hooks._async_tracking_lock:
            assert len(shell_hooks._live_futures) >= 1

        # Drain: wait for the future to resolve + sentinel to exist.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with shell_hooks._async_tracking_lock:
                remaining = len(shell_hooks._live_futures)
            if remaining == 0 and sentinel.exists():
                break
            time.sleep(0.05)
        assert sentinel.exists(), "background subprocess did not complete"


class TestAsyncExecutesInBackground:
    def test_body_runs_and_sentinel_appears(
        self, tmp_path, stub_async_config,
    ):
        stub_async_config(pool_size=4, grace=5)
        sentinel = tmp_path / "run.log"
        script = _write_script(
            tmp_path, "log.sh",
            f"#!/usr/bin/env bash\necho $$ > {sentinel}\nprintf '{{}}\\n'\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="on_session_end", command=str(script),
            timeout=10, is_async=True,
        )
        cb = shell_hooks._make_callback(spec)
        cb(session_id="s-1")

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not sentinel.exists():
            time.sleep(0.05)
        assert sentinel.exists()


# ── saturation: bounded semaphore drops firings ──────────────────────────


class TestPoolSaturation:
    def test_saturation_drops_firings(
        self, tmp_path, stub_async_config, caplog,
    ):
        stub_async_config(pool_size=1, grace=5)

        first_marker = tmp_path / "first.marker"
        second_marker = tmp_path / "second.marker"

        first_script = _write_script(
            tmp_path, "first.sh",
            f"#!/usr/bin/env bash\nsleep 2\ntouch {first_marker}\n"
            f"printf '{{}}\\n'\n",
        )
        second_script = _write_script(
            tmp_path, "second.sh",
            f"#!/usr/bin/env bash\ntouch {second_marker}\nprintf '{{}}\\n'\n",
        )

        spec_a = shell_hooks.ShellHookSpec(
            event="post_tool_call", command=str(first_script),
            timeout=30, is_async=True,
        )
        spec_b = shell_hooks.ShellHookSpec(
            event="post_tool_call", command=str(second_script),
            timeout=30, is_async=True,
        )
        cb_a = shell_hooks._make_callback(spec_a)
        cb_b = shell_hooks._make_callback(spec_b)

        # First firing grabs the semaphore.
        cb_a(tool_name="terminal")
        # Second firing must be dropped (semaphore is at 0).
        with caplog.at_level(
            logging.WARNING, logger=shell_hooks.logger.name,
        ):
            cb_b(tool_name="terminal")
        assert any(
            "pool saturated" in r.getMessage() for r in caplog.records
        )
        # Second subprocess was never spawned.
        assert not second_marker.exists()

        # Wait for the first to clear.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not first_marker.exists():
            time.sleep(0.05)
        assert first_marker.exists()

        # Third firing after release succeeds.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with shell_hooks._async_tracking_lock:
                remaining = len(shell_hooks._live_futures)
            if remaining == 0:
                break
            time.sleep(0.05)
        cb_b(tool_name="terminal")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not second_marker.exists():
            time.sleep(0.05)
        assert second_marker.exists()


# ── async pre_tool_call block directives are ignored ─────────────────────


class TestAsyncBlockIsIgnored:
    def test_block_logged_warn_and_discarded(
        self, tmp_path, stub_async_config, caplog,
    ):
        stub_async_config(pool_size=2, grace=5)
        script = _write_script(
            tmp_path, "block.sh",
            "#!/usr/bin/env bash\n"
            'printf \'{"action": "block", "message": "late"}\\n\'\n',
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command=str(script),
            timeout=10, is_async=True,
        )
        cb = shell_hooks._make_callback(spec)
        with caplog.at_level(
            logging.WARNING, logger=shell_hooks.logger.name,
        ):
            # Async callback returns None (no block directive surfaces).
            assert cb(tool_name="terminal") is None
            # Wait for worker body to run and log.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with shell_hooks._async_tracking_lock:
                    remaining = len(shell_hooks._live_futures)
                if remaining == 0:
                    break
                time.sleep(0.05)
        assert any(
            "block directive but firing is async" in r.getMessage()
            and "late" in r.getMessage()
            for r in caplog.records
        ), [r.getMessage() for r in caplog.records]


# ── shutdown: terminates running subprocesses ─────────────────────────────


class TestShutdownTerminatesSubprocesses:
    def test_shutdown_fires_sigterm_and_returns_within_grace(
        self, tmp_path, stub_async_config,
    ):
        stub_async_config(pool_size=2, grace=1)
        script = _write_script(
            tmp_path, "sleepy.sh",
            "#!/usr/bin/env bash\nsleep 30\nprintf '{}\\n'\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="post_tool_call", command=str(script),
            timeout=60, is_async=True,
        )
        cb = shell_hooks._make_callback(spec)
        cb(tool_name="terminal")

        # Let the subprocess actually spawn.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with shell_hooks._async_tracking_lock:
                n_procs = len(shell_hooks._live_procs)
            if n_procs >= 1:
                break
            time.sleep(0.05)
        assert n_procs >= 1

        t0 = time.monotonic()
        shell_hooks.shutdown_async_hooks(grace_seconds=1)
        elapsed = time.monotonic() - t0

        assert elapsed < 5, f"shutdown took {elapsed:.2f}s"
        with shell_hooks._async_tracking_lock:
            assert len(shell_hooks._live_procs) == 0

    def test_shutdown_returns_promptly_when_children_exit_fast(
        self, tmp_path, stub_async_config,
    ):
        """shutdown issues SIGTERM, then waits up to ``grace_seconds``.
        If the subprocess honors SIGTERM and exits quickly, shutdown
        must return long before the grace window elapses — not burn
        the full grace every time."""
        stub_async_config(pool_size=2, grace=10)
        script = _write_script(
            tmp_path, "sleepy.sh",
            "#!/usr/bin/env bash\nsleep 30\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="post_tool_call", command=str(script),
            timeout=60, is_async=True,
        )
        cb = shell_hooks._make_callback(spec)
        cb(tool_name="terminal")

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with shell_hooks._async_tracking_lock:
                n_procs = len(shell_hooks._live_procs)
            if n_procs >= 1:
                break
            time.sleep(0.05)
        assert n_procs >= 1

        t0 = time.monotonic()
        shell_hooks.shutdown_async_hooks(grace_seconds=10)
        elapsed = time.monotonic() - t0

        # bash sleep honors SIGTERM immediately; shutdown must come back
        # well under the 10s grace window.
        assert elapsed < 3, f"shutdown took {elapsed:.2f}s (grace=10s)"

    def test_shutdown_escalates_even_when_shutting_down_flag_preset(
        self, tmp_path, stub_async_config,
    ):
        """Regression: the signal-handler path flips
        ``_async_shutting_down`` *before* atexit runs
        ``shutdown_async_hooks``.  Gating idempotency on that flag would
        skip the SIGKILL cascade for subprocesses that ignore SIGTERM.
        """
        stub_async_config(pool_size=2, grace=1)
        script = _write_script(
            tmp_path, "trap_term.sh",
            "#!/usr/bin/env bash\n"
            "trap '' TERM\n"
            "sleep 30\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="post_tool_call", command=str(script),
            timeout=60, is_async=True,
        )
        cb = shell_hooks._make_callback(spec)
        cb(tool_name="terminal")

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with shell_hooks._async_tracking_lock:
                n_procs = len(shell_hooks._live_procs)
            if n_procs >= 1:
                break
            time.sleep(0.05)
        assert n_procs >= 1
        with shell_hooks._async_tracking_lock:
            tracked_proc = next(iter(shell_hooks._live_procs))

        # Mimic what the signal handler does before atexit fires us.
        shell_hooks._async_shutting_down = True

        t0 = time.monotonic()
        shell_hooks.shutdown_async_hooks(grace_seconds=1)
        elapsed = time.monotonic() - t0

        assert elapsed < 5, f"shutdown took {elapsed:.2f}s"
        # SIGTERM-ignoring subprocess must have been SIGKILLed by the
        # escalation (rc == -SIGKILL on POSIX); the early-return bug
        # would have left it alive.
        assert tracked_proc.poll() is not None
        with shell_hooks._async_tracking_lock:
            assert len(shell_hooks._live_procs) == 0


# ── Claude-Code runtime async marker on sync hook ────────────────────────


class TestClaudeCodeRuntimeAsyncMarker:
    def test_marker_on_sync_hook_warns_and_is_ignored(
        self, tmp_path, caplog,
    ):
        script = _write_script(
            tmp_path, "cc_async.sh",
            "#!/usr/bin/env bash\n"
            'printf \'{"async": true}\\n\'\n',
        )
        spec = shell_hooks.ShellHookSpec(
            event="post_tool_call", command=str(script), timeout=5,
        )
        cb = shell_hooks._make_callback(spec)
        with caplog.at_level(
            logging.WARNING, logger=shell_hooks.logger.name,
        ):
            assert cb(tool_name="terminal") is None
        assert any(
            "requested async at runtime" in r.getMessage()
            for r in caplog.records
        )


# ── thread-safety under load ─────────────────────────────────────────────


class TestTrackingThreadSafety:
    def test_concurrent_firings_do_not_corrupt_tracking(
        self, tmp_path, stub_async_config,
    ):
        stub_async_config(pool_size=8, grace=5)
        script = _write_script(
            tmp_path, "quick.sh",
            "#!/usr/bin/env bash\nprintf '{}\\n'\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="post_tool_call", command=str(script),
            timeout=10, is_async=True,
        )
        cb = shell_hooks._make_callback(spec)

        N_THREADS = 10
        FIRINGS_PER_THREAD = 5
        errors = []

        def _worker():
            try:
                for _ in range(FIRINGS_PER_THREAD):
                    cb(tool_name="terminal")
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            threading.Thread(target=_worker) for _ in range(N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors

        # Drain everything.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with shell_hooks._async_tracking_lock:
                remaining = len(shell_hooks._live_futures)
            if remaining == 0:
                break
            time.sleep(0.05)
        with shell_hooks._async_tracking_lock:
            assert len(shell_hooks._live_futures) == 0
            assert len(shell_hooks._live_procs) == 0


# ── SIGINT terminates live subprocesses (CLI mode) ───────────────────────


class TestSigintTerminatesSubprocesses:
    def test_sigint_handler_terminates_tracked_procs(
        self, tmp_path, stub_async_config,
    ):
        """Simulate a SIGINT arrival while a hook subprocess is live.
        The CLI-mode handler must terminate the child and chain to the
        previous handler."""
        stub_async_config(pool_size=2, grace=1)
        script = _write_script(
            tmp_path, "hang.sh",
            "#!/usr/bin/env bash\nsleep 30\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="post_tool_call", command=str(script),
            timeout=60, is_async=True,
        )
        cb = shell_hooks._make_callback(spec)
        cb(tool_name="terminal")

        # Wait for the subprocess to spawn.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with shell_hooks._async_tracking_lock:
                n_procs = len(shell_hooks._live_procs)
            if n_procs >= 1:
                break
            time.sleep(0.05)
        assert n_procs >= 1
        with shell_hooks._async_tracking_lock:
            tracked_proc = next(iter(shell_hooks._live_procs))

        # Stage the handler state as if _maybe_install_sigint_handler
        # had run — we can't actually install the signal in a test
        # without affecting pytest, so call the handler directly.
        previous_received = []

        def _prev(sig, frm):
            previous_received.append(sig)

        shell_hooks._original_sigint_handler = _prev
        shell_hooks._sigint_handler_installed = True

        shell_hooks._async_pool_sigint_handler(signal.SIGINT, None)
        assert previous_received == [signal.SIGINT]

        # The subprocess was terminated, not allowed to run 30s.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if tracked_proc.poll() is not None:
                break
            time.sleep(0.05)
        assert tracked_proc.poll() is not None, (
            "SIGINT handler did not terminate the tracked subprocess"
        )

    def test_sigint_chains_to_default_with_keyboardinterrupt(self):
        """When the previous handler was the default disposition (None
        / SIG_DFL / SIG_IGN), the async handler raises
        KeyboardInterrupt so the normal interrupt flow still happens."""
        shell_hooks._original_sigint_handler = None
        shell_hooks._sigint_handler_installed = True
        with pytest.raises(KeyboardInterrupt):
            shell_hooks._async_pool_sigint_handler(signal.SIGINT, None)

    def test_sigint_escalates_to_sigkill_when_term_is_ignored(
        self, tmp_path, stub_async_config,
    ):
        """Regression: the signal handler must run the full
        SIGTERM → grace → SIGKILL escalation inline.  Previously it
        sent only SIGTERM and relied on atexit for the escalation —
        but atexit runs *after* non-daemon workers exit, and a worker
        stuck in ``proc.communicate(timeout=spec.timeout)`` waiting on
        a TERM-ignoring subprocess delays interpreter shutdown by up
        to 60 s even though ``grace_seconds`` is 1 s."""
        stub_async_config(pool_size=2, grace=1)
        sentinel = tmp_path / "handler_ready.sentinel"
        # A Python process can truly ignore SIGTERM (bash's
        # ``trap '' TERM`` doesn't help because bash's *external*
        # ``sleep`` child still dies from SIGTERM independently).
        # The sentinel closes a race window where SIGTERM arrives
        # before the Python-level handler is installed — in that case
        # Python's default disposition terminates the child on
        # SIGTERM and we'd never exercise the SIGKILL path.
        script = _write_script(
            tmp_path, "py_ignore_term.py",
            "#!/usr/bin/env python3\n"
            "import os, signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *a: None)\n"
            f"open({str(sentinel)!r}, 'w').close()\n"
            "sys.stdin.read()\n"
            "end = time.monotonic() + 120\n"
            "while time.monotonic() < end:\n"
            "    try: time.sleep(1)\n"
            "    except InterruptedError: pass\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="post_tool_call", command=str(script),
            timeout=60, is_async=True,
        )
        cb = shell_hooks._make_callback(spec)
        cb(tool_name="terminal")

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with shell_hooks._async_tracking_lock:
                n_procs = len(shell_hooks._live_procs)
            if n_procs >= 1 and sentinel.exists():
                break
            time.sleep(0.05)
        assert n_procs >= 1
        assert sentinel.exists(), "child never installed SIGTERM handler"
        with shell_hooks._async_tracking_lock:
            tracked_proc = next(iter(shell_hooks._live_procs))

        previous_received: list[int] = []
        shell_hooks._original_sigint_handler = (
            lambda sig, frm: previous_received.append(sig)
        )
        shell_hooks._sigint_handler_installed = True

        t0 = time.monotonic()
        shell_hooks._async_pool_sigint_handler(signal.SIGINT, None)
        elapsed = time.monotonic() - t0

        assert previous_received == [signal.SIGINT]
        # Before the fix this took ~spec.timeout (60 s) because the
        # escalation happened only in atexit, which waited for the
        # worker thread to unblock first.  After the fix the handler
        # runs the escalation inline and comes back within a small
        # multiple of ``grace_seconds`` (1 s in this test).
        assert elapsed < 5, f"handler took {elapsed:.2f}s (grace=1s)"
        # TERM-ignoring Python subprocess can only die via SIGKILL.
        assert tracked_proc.poll() is not None
        assert tracked_proc.returncode == -signal.SIGKILL, (
            f"expected -SIGKILL rc, got {tracked_proc.returncode}"
        )
