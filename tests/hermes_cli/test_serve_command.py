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

_DASH = object()
_REGISTER = object()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    build_dashboard_parser(
        parser.add_subparsers(dest="command"),
        cmd_dashboard=_DASH,
        cmd_dashboard_register=_REGISTER,
    )
    return parser


def test_serve_routes_to_the_shared_dashboard_handler():
    args = _parser().parse_args(["serve"])
    assert args.func is _DASH


def test_serve_is_headless_by_default_but_dashboard_is_not():
    assert _parser().parse_args(["serve"]).no_open is True
    assert _parser().parse_args(["dashboard"]).no_open is False


def test_serve_accepts_the_legacy_no_open_flag_as_a_noop():
    # The desktop backend spawn (and old shells) may still pass --no-open;
    # serve must tolerate it rather than erroring on an unknown argument.
    assert _parser().parse_args(["serve", "--no-open"]).no_open is True


def test_serve_takes_the_same_runtime_flags_as_dashboard():
    argv = ["--host", "0.0.0.0", "--port", "0", "--insecure", "--skip-build", "--isolated"]
    serve = _parser().parse_args(["serve", *argv])
    dash = _parser().parse_args(["dashboard", *argv])
    for field in ("host", "port", "insecure", "skip_build", "isolated"):
        assert getattr(serve, field) == getattr(dash, field)


@pytest.mark.parametrize("flag", ["--stop", "--status"])
def test_serve_supports_the_lifecycle_flags(flag):
    assert getattr(_parser().parse_args(["serve", flag]), flag.lstrip("-")) is True
