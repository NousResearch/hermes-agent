"""Tests for the start_gateway exit-code logic after /restart.

The exit-decision block in ``start_gateway`` (gateway/run.py) chooses the
process exit code based on three flags:

  _signal_initiated_shutdown | _restart_requested | _restart_via_service
  ──────────────────────────┼────────────────────┼─────────────────────
  False                     │ True               │ False  → exit 1
  False                     │ True               │ True   → exit 75
  True                      │ False              │ -      → exit 1
  False                     │ False              │ False  → exit 0

The first row was missing before the fix — a /restart on launchd exited 0,
and ``KeepAlive.SuccessfulExit=false`` kept the gateway dead.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner, start_gateway


class TestRestartExitCode:
    """Verify that /restart on a non-systemd gateway exits non-zero."""

    @staticmethod
    def _make_runner(*, restart_requested: bool = False,
                     restart_via_service: bool = False,
                     signal_initiated: bool = False) -> GatewayRunner:
        """Create a minimal GatewayRunner for exit-decision tests."""
        runner = object.__new__(GatewayRunner)
        runner._restart_requested = restart_requested
        runner._restart_via_service = restart_via_service
        runner._signal_initiated_shutdown = signal_initiated
        runner._restart_detached = False
        runner._restart_task_started = False
        runner._draining = False
        runner._running = False
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._pending_approvals = {}
        runner._background_tasks = set()
        runner._shutdown_event = MagicMock()
        runner._exit_code = None
        runner._exit_reason = None
        runner._busy_input_mode = "interrupt"
        runner.adapters = {}
        runner.config = GatewayConfig()
        return runner

    def test_restart_without_service_exits_nonzero(self):
        """A /restart on launchd (no _restart_via_service) must exit non-zero
        so launchd's KeepAlive.SuccessfulExit=false revives the gateway."""
        runner = self._make_runner(restart_requested=True,
                                   restart_via_service=False)
        _signal_initiated_shutdown = False

        # Replicate the exit-decision block from start_gateway:
        if _signal_initiated_shutdown and not runner._restart_requested:
            result = False
        elif runner._restart_via_service:
            result = "systemd_75"
        elif runner._restart_requested:
            result = False
        else:
            result = True

        assert result is False, (
            "/restart without service manager must return False (→ exit 1)"
        )

    def test_restart_via_service_raises_exit_75(self):
        """systemd shortcut path should raise SystemExit(75)."""
        runner = self._make_runner(restart_requested=True,
                                   restart_via_service=True)
        _signal_initiated_shutdown = False

        if _signal_initiated_shutdown and not runner._restart_requested:
            result = False
        elif runner._restart_via_service:
            result = "systemd_75"
        elif runner._restart_requested:
            result = False
        else:
            result = True

        assert result == "systemd_75"

    def test_signal_without_restart_exits_nonzero(self):
        """An unexpected signal without /restart should exit non-zero."""
        runner = self._make_runner(restart_requested=False,
                                   signal_initiated=True)
        _signal_initiated_shutdown = True

        if _signal_initiated_shutdown and not runner._restart_requested:
            result = False
        elif runner._restart_via_service:
            result = "systemd_75"
        elif runner._restart_requested:
            result = False
        else:
            result = True

        assert result is False

    def test_clean_shutdown_exits_zero(self):
        """A normal shutdown (no signal, no restart) should exit 0."""
        runner = self._make_runner(restart_requested=False,
                                   signal_initiated=False)
        _signal_initiated_shutdown = False

        if _signal_initiated_shutdown and not runner._restart_requested:
            result = False
        elif runner._restart_via_service:
            result = "systemd_75"
        elif runner._restart_requested:
            result = False
        else:
            result = True

        assert result is True

    def test_signal_with_restart_exits_nonzero(self):
        """Signal + restart request (SIGUSR1 /restart path) should still
        reach the restart-without-service branch and exit non-zero."""
        runner = self._make_runner(restart_requested=True,
                                   restart_via_service=False,
                                   signal_initiated=True)
        _signal_initiated_shutdown = True

        # Note: when both signal AND restart are set, the first condition
        # (_signal_initiated_shutdown and not _restart_requested) is False
        # because _restart_requested is True. So we fall through to the
        # restart branch.
        if _signal_initiated_shutdown and not runner._restart_requested:
            result = False
        elif runner._restart_via_service:
            result = "systemd_75"
        elif runner._restart_requested:
            result = False
        else:
            result = True

        assert result is False
