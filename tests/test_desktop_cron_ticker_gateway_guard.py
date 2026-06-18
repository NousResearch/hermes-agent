"""Desktop dashboard cron ticker must not race a healthy gateway.

The desktop-spawned dashboard backend runs its own cron ticker so jobs fire
when no gateway exists. But when the profile's gateway IS running, the ticker
races it for ``cron/.tick.lock`` — and dashboard claims use the standalone
send path (no live adapters), bypassing threaded delivery. The ticker must
skip its tick while a gateway is alive, re-checking every iteration, and
fail open (tick anyway) if the gateway health check itself breaks.
"""

import threading
from unittest.mock import MagicMock, patch

from hermes_cli.web_server import _start_desktop_cron_ticker


class _SelfStoppingEvent(threading.Event):
    """Event whose wait() sets itself after N waits — bounds the ticker loop."""

    def __init__(self, iterations: int = 1):
        super().__init__()
        self._remaining = iterations

    def wait(self, timeout=None):  # noqa: ARG002 - signature matches Event
        self._remaining -= 1
        if self._remaining <= 0:
            self.set()
        return True


def _run_one_iteration(gateway_running, iterations: int = 1):
    """Run the ticker loop for N iterations with patched collaborators."""
    cron_tick = MagicMock()
    check = MagicMock(side_effect=gateway_running)
    with (
        patch("cron.scheduler.tick", cron_tick),
        patch("gateway.status.is_gateway_running", check),
    ):
        _start_desktop_cron_ticker(_SelfStoppingEvent(iterations), interval=0)
    return cron_tick, check


class TestDesktopTickerGatewayGuard:
    def test_skips_tick_while_gateway_running(self):
        cron_tick, check = _run_one_iteration(gateway_running=[True])
        cron_tick.assert_not_called()
        check.assert_called()

    def test_ticks_when_no_gateway_running(self):
        cron_tick, _ = _run_one_iteration(gateway_running=[False])
        cron_tick.assert_called_once()

    def test_rechecks_gateway_every_iteration(self):
        # Gateway dies between iterations: skip the first tick, fire the second.
        cron_tick, check = _run_one_iteration(
            gateway_running=[True, False], iterations=2
        )
        cron_tick.assert_called_once()
        assert check.call_count == 2

    def test_fails_open_when_gateway_check_raises(self):
        # A broken health check must not silence cron entirely — flat
        # delivery beats no delivery.
        cron_tick, _ = _run_one_iteration(
            gateway_running=RuntimeError("status unreadable")
        )
        cron_tick.assert_called_once()
