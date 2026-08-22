"""Contract for the headless ``hermes serve`` backend command.

``serve`` is what the desktop app and remote backends launch — the same gateway
as ``dashboard`` (shared handler) but always headless, and decoupled in name so
the desktop never invokes ``dashboard``. These tests pin that contract:

- ``serve`` routes to the same handler as ``dashboard``;
- ``serve`` is headless by default, ``dashboard`` is not;
- both expose the identical server-runtime flag surface.
"""

from __future__ import annotations

import argparse

import pytest

from hermes_cli.subcommands.dashboard import build_dashboard_parser


def _dash(args):  # sentinel handler — identity-compared, never invoked
    return args


def _register(args):
    return args


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    build_dashboard_parser(
        parser.add_subparsers(dest="command"),
        cmd_dashboard=_dash,
        cmd_dashboard_register=_register,
    )
    return parser






def test_serve_supports_the_lifecycle_flags():
    for flag in ("--stop", "--status"):
        assert getattr(_parser().parse_args(["serve", flag]), flag.lstrip("-")) is True


def test_serve_is_a_headless_backend_but_dashboard_is_not():
    # `headless_backend` is the flag cmd_dashboard reads to skip the web UI
    # build; only `serve` carries it.
    assert getattr(_parser().parse_args(["serve"]), "headless_backend", False) is True
    assert getattr(_parser().parse_args(["dashboard"]), "headless_backend", False) is False


def _option_help(command: str, option: str) -> str:
    parser = _parser()
    command_parser = parser._subparsers._group_actions[0].choices[command]
    return next(
        action.help
        for action in command_parser._actions
        if option in action.option_strings
    )


@pytest.mark.parametrize("command", ["dashboard", "serve"])
def test_insecure_help_names_the_command_execution_privilege_boundary(command):
    insecure_help = _option_help(command, "--insecure")

    assert "agent and command execution" in insecure_help
    assert "Hermes process's OS privileges" in insecure_help


def test_skip_build_help_distinguishes_dashboard_from_headless_serve():
    dashboard_help = _option_help("dashboard", "--skip-build")
    serve_help = _option_help("serve", "--skip-build")

    assert "serve the existing dist directly" in dashboard_help
    assert "Accepted no-op for backward compatibility" in serve_help
    assert "serve is always headless" in serve_help
