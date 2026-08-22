"""Desktop-backend messaging-gateway lifecycle (#84311).

Regression coverage for the Windows desktop close/reopen loop:

* On close, the desktop tree-kills its backend (`taskkill /T /F`), which
  could kill the messaging gateway when it sat in the backend's process tree.
* On reopen, the backend unconditionally reaped *any* live gateway (the
  #77276 duplicate guard) and never started a replacement — so the gateway
  was permanently gone until a manual `hermes gateway start`.

The fix: the gateway is an independent service. The desktop backend keeps a
live gateway (no reap), reaps stale-port orphans only when nothing is live,
and auto-starts a gateway the user's persisted intent says should be running
(last `gateway_state.json` was a run state AND messaging platforms are
configured).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest


def _runtime(state: str | None) -> dict | None:
    if state is None:
        return None
    return {"gateway_state": state}


class TestGatewayAutoStartReason:
    """The intent decision: last gateway state x configured platforms."""

    def test_running_state_with_platforms_auto_starts(self):
        from hermes_cli.web_server import _gateway_auto_start_reason

        with (
            patch("hermes_cli.web_server.read_runtime_status", return_value=_runtime("running")),
            patch(
                "hermes_cli.web_server._load_configured_gateway_platforms",
                return_value={"telegram"},
            ),
        ):
            assert _gateway_auto_start_reason() is not None

    def test_draining_state_with_platforms_auto_starts(self):
        from hermes_cli.web_server import _gateway_auto_start_reason

        with (
            patch("hermes_cli.web_server.read_runtime_status", return_value=_runtime("draining")),
            patch(
                "hermes_cli.web_server._load_configured_gateway_platforms",
                return_value={"telegram"},
            ),
        ):
            assert _gateway_auto_start_reason() is not None

    def test_stopped_state_never_auto_starts(self):
        from hermes_cli.web_server import _gateway_auto_start_reason

        with (
            patch("hermes_cli.web_server.read_runtime_status", return_value=_runtime("stopped")),
            patch(
                "hermes_cli.web_server._load_configured_gateway_platforms",
                return_value={"telegram"},
            ),
        ):
            assert _gateway_auto_start_reason() is None

    def test_startup_failed_state_never_auto_starts(self):
        """A failing gateway must not be hammered on every desktop open."""
        from hermes_cli.web_server import _gateway_auto_start_reason

        with (
            patch(
                "hermes_cli.web_server.read_runtime_status",
                return_value=_runtime("startup_failed"),
            ),
            patch(
                "hermes_cli.web_server._load_configured_gateway_platforms",
                return_value={"telegram"},
            ),
        ):
            assert _gateway_auto_start_reason() is None

    def test_missing_state_record_never_auto_starts(self):
        """No record = no running intent (mirrors get_runtime_status_running_pid)."""
        from hermes_cli.web_server import _gateway_auto_start_reason

        with (
            patch("hermes_cli.web_server.read_runtime_status", return_value=None),
            patch(
                "hermes_cli.web_server._load_configured_gateway_platforms",
                return_value={"telegram"},
            ),
        ):
            assert _gateway_auto_start_reason() is None

    def test_no_configured_platforms_never_auto_starts(self):
        from hermes_cli.web_server import _gateway_auto_start_reason

        with (
            patch("hermes_cli.web_server.read_runtime_status", return_value=_runtime("running")),
            patch(
                "hermes_cli.web_server._load_configured_gateway_platforms",
                return_value=set(),
            ),
        ):
            assert _gateway_auto_start_reason() is None

    def test_platform_scan_failure_never_auto_starts(self):
        from hermes_cli.web_server import _gateway_auto_start_reason

        with (
            patch("hermes_cli.web_server.read_runtime_status", return_value=_runtime("running")),
            patch(
                "hermes_cli.web_server._load_configured_gateway_platforms",
                side_effect=OSError("config unreadable"),
            ),
        ):
            assert _gateway_auto_start_reason() is None


class TestManageDesktopGatewayAtBoot:
    """The boot decision: keep live gateways, reap + auto-start only when dead."""

    def test_live_gateway_is_kept_not_reaped(self):
        """A gateway that survived the previous session must be left alone."""
        from hermes_cli.web_server import _manage_desktop_gateway_at_boot

        with (
            patch("hermes_cli.gateway.find_gateway_pids", return_value=[12345]),
            patch(
                "hermes_cli.gateway._reap_unsupervised_gateway_orphans"
            ) as mock_reap,
            patch("hermes_cli.web_server.asyncio.create_task") as mock_task,
        ):
            _manage_desktop_gateway_at_boot()

        mock_reap.assert_not_called()
        mock_task.assert_not_called()

    def test_dead_gateway_reaps_and_schedules_auto_start(self):
        from hermes_cli.web_server import _manage_desktop_gateway_at_boot

        with (
            patch("hermes_cli.gateway.find_gateway_pids", return_value=[]),
            patch(
                "hermes_cli.gateway._reap_unsupervised_gateway_orphans"
            ) as mock_reap,
            patch("hermes_cli.web_server.asyncio.create_task") as mock_task,
        ):
            _manage_desktop_gateway_at_boot()

        mock_reap.assert_called_once()
        mock_task.assert_called_once()

    def test_liveness_scan_failure_degrades_to_reap_and_auto_start(self):
        """A failed scan must not leave the gateway dead forever — fail open."""
        from hermes_cli.web_server import _manage_desktop_gateway_at_boot

        with (
            patch(
                "hermes_cli.gateway.find_gateway_pids",
                side_effect=OSError("scan failed"),
            ),
            patch(
                "hermes_cli.gateway._reap_unsupervised_gateway_orphans"
            ) as mock_reap,
            patch("hermes_cli.web_server.asyncio.create_task") as mock_task,
        ):
            _manage_desktop_gateway_at_boot()

        mock_reap.assert_called_once()
        mock_task.assert_called_once()


class TestMaybeAutoStartGateway:
    """The auto-start spawn: intent, in-flight dedupe, and failure handling."""

    def test_spawns_gateway_start_when_intent(self):
        """Dead gateway + running intent -> `hermes gateway start` action."""
        from hermes_cli.web_server import _maybe_auto_start_gateway

        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 4242
        with (
            patch(
                "hermes_cli.web_server._gateway_auto_start_reason",
                return_value="last gateway state 'running' with 1 configured platform(s)",
            ),
            patch("hermes_cli.web_server._ACTION_PROCS", {}),
            patch("hermes_cli.web_server._spawn_hermes_action", return_value=proc) as mock_spawn,
            patch(
                "hermes_cli.web_server._gateway_subcommand",
                return_value=["gateway", "start"],
            ),
        ):
            asyncio.run(_maybe_auto_start_gateway())

        mock_spawn.assert_called_once()
        args, kwargs = mock_spawn.call_args
        assert args[0] == ["gateway", "start"]
        assert args[1] == "gateway-start"

    def test_skips_spawn_when_no_intent(self):
        from hermes_cli.web_server import _maybe_auto_start_gateway

        with (
            patch("hermes_cli.web_server._gateway_auto_start_reason", return_value=None),
            patch("hermes_cli.web_server._ACTION_PROCS", {}),
            patch("hermes_cli.web_server._spawn_hermes_action") as mock_spawn,
        ):
            asyncio.run(_maybe_auto_start_gateway())

        mock_spawn.assert_not_called()

    def test_skips_spawn_when_start_action_in_flight(self):
        """A user-initiated start while auto-start is pending must not stack."""
        from hermes_cli.web_server import _maybe_auto_start_gateway

        inflight = MagicMock(spec=subprocess.Popen)
        inflight.poll.return_value = None
        with (
            patch(
                "hermes_cli.web_server._gateway_auto_start_reason",
                return_value="last gateway state 'running' with 1 configured platform(s)",
            ),
            patch("hermes_cli.web_server._ACTION_PROCS", {"gateway-start": inflight}),
            patch("hermes_cli.web_server._spawn_hermes_action") as mock_spawn,
        ):
            asyncio.run(_maybe_auto_start_gateway())

        mock_spawn.assert_not_called()

    def test_spawn_failure_is_logged_not_raised(self):
        from hermes_cli.web_server import _maybe_auto_start_gateway

        with (
            patch(
                "hermes_cli.web_server._gateway_auto_start_reason",
                return_value="last gateway state 'running' with 1 configured platform(s)",
            ),
            patch("hermes_cli.web_server._ACTION_PROCS", {}),
            patch(
                "hermes_cli.web_server._spawn_hermes_action",
                side_effect=OSError("spawn failed"),
            ),
            patch("hermes_cli.web_server._log") as mock_log,
        ):
            asyncio.run(_maybe_auto_start_gateway())

        assert mock_log.exception.called
