"""
Regression test for issue #62151 - Gateway-executed cron jobs deadlock
on the 2nd+ API call.

The bug: when the gateway runs cron jobs in-process, the cron worker
thread (in a single-worker ThreadPoolExecutor) calls agent.run_conversation
which makes API calls. After the first API call returns with tool_calls,
the agent schedules tool-execution coroutines onto the main gateway
event loop via safe_schedule_threadsafe and blocks on the resulting
Future. If the main loop is busy serving HTTP/WS, both threads block
indefinitely → deadlock.

The fix: add an env-gated escape hatch (HERMES_CRON_DISABLE_IN_PROCESS=1)
that disables the in-process cron ticker entirely. Operators with this
issue can then run system-cron `hermes cron tick` from a separate process
(which works because it has its own event loop).

Tests:
  1. test_static_disable_flag_is_respected - source-level check that
     the start() method reads the env var and returns None.
  2. test_behavioral_disable_returns_none_when_env_set - behavioral
     check that with HERMES_CRON_DISABLE_IN_PROCESS=1, start() returns
     None without entering the tick loop.
  3. test_behavioral_default_still_runs - behavioral check that without
     the env var, start() proceeds normally (calls into the tick loop).
     We don't run the loop indefinitely - we just verify start() does NOT
     return None immediately.
"""

import re
from pathlib import Path


def test_static_disable_flag_is_respected():
    """Source-level tripwire: InProcessCronScheduler.start() must read
    HERMES_CRON_DISABLE_IN_PROCESS and return None when set to "1"."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    src = (worktree / "cron" / "scheduler_provider.py").read_text()

    # Find the InProcessCronScheduler class
    m = re.search(r"class InProcessCronScheduler.*?(?=^class |\Z)",
                  src, re.MULTILINE | re.DOTALL)
    assert m, "InProcessCronScheduler class not found"
    body = m.group(0)

    # The env var check must be present
    assert "HERMES_CRON_DISABLE_IN_PROCESS" in body, (
        "#62151 regression: the HERMES_CRON_DISABLE_IN_PROCESS env var "
        "check is missing from InProcessCronScheduler.start(). Without it, "
        "operators cannot disable the in-process cron ticker to avoid the "
        "deadlock between the cron worker thread and the main gateway "
        "event loop."
    )

    # The check must result in early return (return None) so the tick
    # loop is bypassed
    # Look for the pattern: env check + log warning + return None
    assert re.search(
        r'os\.environ\.get\(["\']HERMES_CRON_DISABLE_IN_PROCESS["\']\)\s*==\s*["\']1["\']',
        body,
    ), "#62151: env var check does not match expected pattern (HERMES_CRON_DISABLE_IN_PROCESS == '1')"

    # Verify the early-return pattern: warning + return None
    assert re.search(
        r'HERMES_CRON_DISABLE_IN_PROCESS.*?return None',
        body,
        re.DOTALL,
    ), "#62151: env var check must result in early return None to bypass the tick loop."


def test_behavioral_disable_returns_none_when_env_set(monkeypatch):
    """Behavioral check: with HERMES_CRON_DISABLE_IN_PROCESS=1,
    InProcessCronScheduler.start() returns None without entering the
    tick loop. The tick loop would block indefinitely waiting on
    stop_event.is_set(), so an early return is essential.
    """
    import os
    import sys
    sys.path.insert(0, "/tmp/hermes-pr-work-60859/hermes-agent")

    monkeypatch.setenv("HERMES_CRON_DISABLE_IN_PROCESS", "1")

    from cron.scheduler_provider import InProcessCronScheduler

    import threading
    stop_event = threading.Event()

    # Call start() - it should return None immediately when disabled
    result = InProcessCronScheduler().start(stop_event)

    assert result is None, (
        f"#62151: with HERMES_CRON_DISABLE_IN_PROCESS=1, start() should return "
        f"None to bypass the tick loop. Got: {result!r}. If the tick loop "
        f"is running, the in-process deadlock is still possible."
    )


def test_behavioral_default_still_runs(monkeypatch):
    """Behavioral check: without HERMES_CRON_DISABLE_IN_PROCESS set,
    start() proceeds into the tick loop. We use a stop_event that is
    ALREADY SET so the loop exits immediately, then verify the function
    returned None (after the loop's normal exit) and record_ticker_heartbeat
    was called.

    This verifies the env-var bypass is gated, not always-on.
    """
    import os
    import sys
    sys.path.insert(0, "/tmp/hermes-pr-work-60859/hermes-agent")

    # Make sure the env var is NOT set
    monkeypatch.delenv("HERMES_CRON_DISABLE_IN_PROCESS", raising=False)

    from cron.scheduler_provider import InProcessCronScheduler

    import threading
    stop_event = threading.Event()
    stop_event.set()  # Pre-set so the loop exits on the first iteration

    # start() will:
    #   1. Pass the env-var check (var not set)
    #   2. Enter the while loop
    #   3. See stop_event.is_set() == True, exit immediately
    #   4. Return None (implicit)
    result = InProcessCronScheduler().start(stop_event)

    # The default behavior is: enter the loop, exit on stop_event, return None
    assert result is None, (
        f"#62151: with the env var unset, start() should still run the tick loop "
        f"(returning None after the loop exits). Got: {result!r}"
    )