"""The dashboard's gateway restart hands off to a detached replacement (#154220).

``POST /api/gateway/restart`` and the config-save auto-restart spawn ``hermes gateway restart`` as a
short-lived action child. Without an installed service, a restart child that lacks ``--detach``
falls through ``start_restart_replacement`` into ``run_gateway`` and becomes the long-lived gateway
itself: a bare venv Python that, on macOS, loses Local Network access because it never goes through
``spawn_unmanaged_gateway`` / ``start_gateway_job``. The Desktop client already passes ``--detach``
(``apps/desktop/electron/local-gateway.ts``); these tests hold the dashboard to the same contract.
"""
from __future__ import annotations

import argparse
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.subcommands.gateway import build_gateway_parser


@pytest.fixture(autouse=True)
def reset_restart_state(monkeypatch):
    """Keep the module-level restart record out of neighbouring tests."""
    from hermes_cli import web_server

    monkeypatch.setattr(web_server, "_LAST_GATEWAY_RESTART", None)


def _parse_gateway_argv(argv: list[str]) -> argparse.Namespace:
    """Parse ``gateway ...`` argv with the real CLI parser (profile selectors are top-level flags)."""
    parser = argparse.ArgumentParser(prog="hermes")
    build_gateway_parser(
        parser.add_subparsers(dest="command"),
        cmd_gateway=lambda _args: None,
        cmd_proxy=lambda _args: None,
        cmd_gateway_enroll=lambda _args: None,
    )
    return parser.parse_args(argv[argv.index("gateway"):])


def _spawned_restart_argv(*, interrupt: bool) -> list[str]:
    """The argv the dashboard hands to ``_spawn_hermes_action`` for a restart."""
    from hermes_cli.web_server import _spawn_gateway_restart

    with patch("hermes_cli.web_server_gateway._spawn_hermes_action") as spawn, patch(
        "hermes_cli.web_server_gateway._ACTION_PROCS", {}
    ):
        _spawn_gateway_restart(interrupt=interrupt)
    return list(spawn.call_args.args[0])


@pytest.mark.parametrize("interrupt", [False, True], ids=["config-save", "user-commanded"])
def test_the_dashboard_restart_child_hands_off_instead_of_becoming_the_gateway(interrupt):
    from hermes_cli.gateway_supervised_restart import start_restart_replacement

    args = _parse_gateway_argv(_spawned_restart_argv(interrupt=interrupt))

    with patch(
        "hermes_cli.gateway_runtime_start.start_detached_replacement", return_value=0
    ) as detached, patch("hermes_cli.gateway.run_gateway") as foreground:
        with pytest.raises(SystemExit) as exited:
            start_restart_replacement(args, force=False)

    assert exited.value.code == 0
    detached.assert_called_once()
    foreground.assert_not_called()


def test_a_failed_detached_restart_is_still_reported():
    """The failure lookup must recognise the command the dashboard now spawns, or every failed
    restart would read as "still running" and the UI would wait for a gateway that never comes."""
    from hermes_cli import web_server
    from hermes_cli.web_server_gateway import gateway_restart_failure

    failed = MagicMock(spec=subprocess.Popen)
    failed.poll.return_value = 1
    failed.pid = 4242
    recorded = (0.0, failed, tuple(_spawned_restart_argv(interrupt=True)))
    web_server._LAST_GATEWAY_RESTART = recorded

    with patch("hermes_cli.web_server_gateway._restart_log_last_line", return_value=""):
        failure = gateway_restart_failure(None)

    assert failure == {"pid": 4242, "exit_code": 1, "reason": "exit code 1"}
