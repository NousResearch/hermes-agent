"""Regression tests for the slow-hook watchdog (#32460).

The watchdog runs in a daemon thread alongside every shell-hook
subprocess and emits a WARNING once the hook crosses the configured
threshold without returning.  Without it, a hook that hangs for the
full ``DEFAULT_TIMEOUT_SECONDS = 60`` window manifests as a silent
50-65s wall-clock gap between an LLM response and the resulting tool
execution — the symptom reported in the bug.

These tests pin three invariants:

* A hook that returns before the threshold emits **no** "still running"
  warning, so well-behaved scripts stay silent.
* A hook that runs **past** the threshold emits at least one
  "still running" warning while it is still in flight (i.e. before the
  subprocess returns), with the offending command in the message so the
  user can identify which script is hanging.
* The watchdog stops once the hook returns and does not leak repeated
  warnings indefinitely.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import shell_hooks


def _write_script(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def _reset_registration_state():
    shell_hooks.reset_for_tests()
    yield
    shell_hooks.reset_for_tests()


@pytest.fixture
def fast_watchdog(monkeypatch):
    """Shrink the watchdog thresholds so tests can exercise them in <2s.

    The production defaults (5s / 10s) are tuned for human-visible logs
    and would make every test on this file ~10s long.  We collapse both
    knobs to small but realistic values.  Threshold stays high enough
    that a single ``bash`` cold start on a loaded test host (~0.3s on
    macOS under contention) doesn't trip the watchdog — otherwise the
    "fast hook stays silent" invariant becomes flaky on shared CI.
    """
    monkeypatch.setattr(shell_hooks, "_SLOW_HOOK_THRESHOLD_SECONDS", 1.0)
    monkeypatch.setattr(shell_hooks, "_SLOW_HOOK_REPEAT_SECONDS", 0.2)


def _slow_hook_warnings(caplog) -> list[str]:
    """Return the messages logged by the watchdog (filter out other lines)."""
    return [
        r.getMessage()
        for r in caplog.records
        if "shell hook still running" in r.getMessage()
    ]


class TestSlowHookWatchdog:
    def test_fast_hook_emits_no_warning(self, tmp_path, caplog, fast_watchdog):
        """Hooks that return below the threshold stay silent.

        Even a script that prints a no-op JSON object and exits must not
        trigger the watchdog warning — otherwise the log line would fire
        on every successful tool dispatch.
        """
        script = _write_script(
            tmp_path,
            "fast.sh",
            "#!/usr/bin/env bash\nprintf '{}\\n'\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command=str(script), timeout=5,
        )

        with caplog.at_level("WARNING", logger="agent.shell_hooks"):
            result = shell_hooks._spawn(spec, "{}")

        assert result["error"] is None
        assert result["timed_out"] is False
        assert _slow_hook_warnings(caplog) == []

    def test_slow_hook_emits_warning_with_command_and_event(
        self, tmp_path, caplog, fast_watchdog
    ):
        """A hook that runs past the threshold must log a visible warning.

        The warning text must name both the event and the command so the
        operator can find the offending script without having to enable
        debug logging or guess at process ancestry.
        """
        script = _write_script(
            tmp_path,
            "slow.sh",
            "#!/usr/bin/env bash\nsleep 1.5\nprintf '{}\\n'\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command=str(script), timeout=10,
        )

        with caplog.at_level("WARNING", logger="agent.shell_hooks"):
            result = shell_hooks._spawn(spec, "{}")

        warnings = _slow_hook_warnings(caplog)
        assert warnings, "watchdog must emit at least one warning for slow hook"
        # The first warning is the threshold notice; it must name the event
        # and the command so the user can identify the offending script.
        first = warnings[0]
        assert "pre_tool_call" in first
        assert str(script) in first
        # The hook still succeeded once it finished, so the spawn result
        # is clean — the warning is purely informational.
        assert result["error"] is None
        assert result["timed_out"] is False

    def test_watchdog_stops_after_hook_returns(
        self, tmp_path, caplog, fast_watchdog
    ):
        """Once the hook returns, the watchdog must stop logging.

        Regression guard: the watchdog uses ``Event.wait()`` with a
        short interval, so a missing ``stop_event.set()`` in the
        subprocess-return path would cause the daemon thread to keep
        warning forever.  We verify the count stabilises within a
        bounded window after the hook returns.
        """
        import time

        script = _write_script(
            tmp_path,
            "slow.sh",
            "#!/usr/bin/env bash\nsleep 1.3\nprintf '{}\\n'\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command=str(script), timeout=10,
        )

        with caplog.at_level("WARNING", logger="agent.shell_hooks"):
            shell_hooks._spawn(spec, "{}")
            count_immediately_after = len(_slow_hook_warnings(caplog))
            # Give the watchdog several extra intervals to (incorrectly)
            # log more warnings if its stop signal wasn't honored.
            time.sleep(0.6)
            count_after_wait = len(_slow_hook_warnings(caplog))

        assert count_after_wait == count_immediately_after, (
            "watchdog must stop firing once the hook subprocess returns"
        )

    def test_hook_timeout_path_also_stops_watchdog(
        self, tmp_path, caplog, fast_watchdog, monkeypatch
    ):
        """When the hook hits ``spec.timeout``, the watchdog must also stop.

        The subprocess ``timeout`` branch is a separate exit path from
        the success branch.  Both must trigger the ``finally: stop``
        in :func:`_spawn`, otherwise a hung hook would leave behind a
        zombie watchdog thread that keeps logging after the timeout
        message has already fired.

        We mock ``subprocess.run`` directly rather than use a real
        ``sleep`` script: subprocess teardown on POSIX can hang
        unbounded inside ``communicate()`` waiting for grandchild
        stdout-pipe EOF (the same footgun documented in
        :func:`_wait_for_process`), which would prevent us from
        observing the watchdog state at the *moment* the timeout fires.
        Mocking lets the assertion focus on the watchdog stop signal.
        """
        import subprocess
        import time

        def _fake_run(*_args, **_kwargs):
            # Simulate a hook that exceeds spec.timeout — sleep long enough
            # that the watchdog warns at least once, then raise.
            time.sleep(1.3)
            raise subprocess.TimeoutExpired(cmd="fake-hook", timeout=1.3)

        monkeypatch.setattr(shell_hooks.subprocess, "run", _fake_run)

        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command="/bin/true", timeout=2,
        )

        with caplog.at_level("WARNING", logger="agent.shell_hooks"):
            result = shell_hooks._spawn(spec, "{}")
            # Capture the warning count right after the timeout fires…
            count_at_timeout = len(_slow_hook_warnings(caplog))
            # …then wait several extra repeat intervals and confirm the
            # watchdog stopped emitting once subprocess.run raised.
            time.sleep(0.6)
            count_after_wait = len(_slow_hook_warnings(caplog))

        assert result["timed_out"] is True
        assert count_after_wait == count_at_timeout, (
            "watchdog must stop firing once the hook subprocess raises "
            "TimeoutExpired (otherwise zombie warnings keep appearing)"
        )

    def test_spawn_failure_does_not_warn(self, tmp_path, caplog, fast_watchdog):
        """Spawn-time failures (command not found) return immediately, so
        the watchdog must not fire even with a tiny threshold.

        The watchdog only makes sense for hooks that actually started
        but won't return.  An ``argv``-validation error never reaches
        the subprocess and should look the same in the log as a
        no-hook setup.
        """
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command="", timeout=5,
        )

        with caplog.at_level("WARNING", logger="agent.shell_hooks"):
            result = shell_hooks._spawn(spec, "{}")

        assert result["error"] == "empty command"
        assert _slow_hook_warnings(caplog) == []
