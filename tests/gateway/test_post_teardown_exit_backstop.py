"""Regression coverage for #108729: `hermes update` can burn its whole drain budget when
teardown itself finishes (logs "Gateway stopped") but a non-cancellable blocking call an adapter
left behind — e.g. a platform SDK thread stuck in an untimed socket read — wedges
``asyncio.run()``'s own task-cancellation sweep afterwards. The drain-scoped shutdown watchdog is
already disarmed by that point, and the process never reaches the post-``asyncio.run()``
``os._exit`` backstop because the loop never returns, so nothing is left to force the exit.

``_stop_persist_exit_state`` must arm an independent, never-disarmed backstop right where it logs
"Gateway stopped" (skipped under pytest, matching the drain-scoped watchdog it complements).
"""

from __future__ import annotations

import time
from unittest.mock import patch

from gateway.run import (
    _CRON_SHUTDOWN_DRAIN_TIMEOUT, _HOUSEKEEPING_SHUTDOWN_DRAIN_TIMEOUT, GatewayRunner,
)
from gateway.run_shutdown import GatewayShutdownMixin
from gateway.shutdown_watchdog import DEFAULT_POST_TEARDOWN_EXIT_GRACE_S


def _make_bare_runner():
    runner = object.__new__(GatewayRunner)
    runner._exit_reason = None
    return runner


def _make_ctx():
    return GatewayShutdownMixin._StopContext(deferred_count=lambda: 0, started_at=time.monotonic())


def test_stop_persist_exit_state_arms_post_teardown_backstop(monkeypatch):
    """Outside pytest, finishing teardown must arm the post-teardown exit backstop."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    runner = _make_bare_runner()
    ctx = _make_ctx()

    with patch("gateway.status.remove_pid_file"), \
         patch("gateway.status.release_gateway_runtime_lock"), \
         patch("gateway.run.GatewayRunner._update_runtime_status"), \
         patch("gateway.run_shutdown.arm_shutdown_watchdog") as arm_mock:
        GatewayRunner._stop_persist_exit_state(runner, ctx)

    assert arm_mock.call_count == 1
    (delay,), kwargs = arm_mock.call_args
    # Must fold in the concurrent _start_gateway_shutdown_tail budgets (cron + housekeeping joins),
    # not just the extra grace, or a slow-but-healthy drain there reads as the wedge we're guarding.
    expected_delay = (
        _CRON_SHUTDOWN_DRAIN_TIMEOUT + _HOUSEKEEPING_SHUTDOWN_DRAIN_TIMEOUT
        + DEFAULT_POST_TEARDOWN_EXIT_GRACE_S
    )
    assert delay == expected_delay
    assert kwargs.get("name") == "gateway-post-teardown-exit-backstop"
    # No done_event: unlike the drain-scoped watchdog, nothing should be able to disarm this one.
    assert "done_event" not in kwargs


def test_stop_persist_exit_state_skips_backstop_under_pytest(monkeypatch):
    """Under pytest (PYTEST_CURRENT_TEST set, as it normally is in this suite) the backstop must not
    arm — an os._exit thread left running past a test's lifetime would eventually kill the runner."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "gateway/test_post_teardown_exit_backstop.py::dummy")
    runner = _make_bare_runner()
    ctx = _make_ctx()

    with patch("gateway.status.remove_pid_file"), \
         patch("gateway.status.release_gateway_runtime_lock"), \
         patch("gateway.run.GatewayRunner._update_runtime_status"), \
         patch("gateway.run_shutdown.arm_shutdown_watchdog") as arm_mock:
        GatewayRunner._stop_persist_exit_state(runner, ctx)

    arm_mock.assert_not_called()
