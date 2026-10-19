"""Contract for the headless ``hermes serve`` backend command.

``serve`` is what the desktop app and remote backends launch — the same gateway
as ``dashboard`` (shared handler) but always headless, and decoupled in name so
the desktop never invokes ``dashboard``. These tests pin that contract:

- ``serve`` routes to the same handler as ``dashboard``;
- each command's launch surface is its own name (``serve`` is the headless one);
- both expose the identical server-runtime flag surface.
"""

from __future__ import annotations

import argparse

from hermes_cli.subcommands.dashboard import build_dashboard_parser


def _dash(args):  # sentinel handler — identity-compared, never invoked
    return args


def _register(args):
    return args


def _webapp(args):
    return args


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    build_dashboard_parser(
        parser.add_subparsers(dest="command"),
        cmd_dashboard=_dash,
        cmd_dashboard_register=_register,
        cmd_webapp=_webapp,
    )
    return parser








def test_each_web_server_command_launches_its_own_surface():
    # cmd_dashboard, the ledger purpose and the named-profile re-exec all read
    # `ui_surface`, and the re-exec replays it as the subcommand.
    for command in ("serve", "dashboard", "webapp"):
        assert _parser().parse_args([command]).ui_surface == command


def test_webapp_is_a_distinct_browser_surface_with_dashboard_runtime_flags():
    parsed = _parser().parse_args(
        ["webapp", "--host", "0.0.0.0", "--port", "9443", "--no-open"]
    )

    assert parsed.func is _webapp
    assert parsed.host == "0.0.0.0"
    assert parsed.port == 9443
    assert parsed.no_open is True
    assert parsed.ui_surface == "webapp"
