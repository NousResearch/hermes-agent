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


def test_desktop_spawned_serve_does_not_force_headless(monkeypatch):
    """#91495: Desktop launches `hermes serve` with HERMES_DESKTOP=1.

    Forcing HERMES_SERVE_HEADLESS=1 then 404s `/` and never mounts `/api/ws`,
    so the desktop WebSocket probe rejects the session token.
    Standalone `hermes serve` (no HERMES_DESKTOP) stays strictly headless.
    """
    from hermes_cli.main import _should_force_serve_headless

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    assert _should_force_serve_headless(True) is False
    assert _should_force_serve_headless(False) is False

    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    assert _should_force_serve_headless(True) is True
    assert _should_force_serve_headless(False) is False
