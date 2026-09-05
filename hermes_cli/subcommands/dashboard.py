"""``hermes dashboard`` / ``hermes webapp`` / ``hermes serve`` parsers.

``dashboard`` is the browser web UI; ``serve`` is the same gateway, headless —
what the desktop app and remote backends run. ``serve`` also skips the web UI
build (``headless_backend=True``): pure JSON-RPC/WS clients never load the SPA.
Both share one handler (``cmd_dashboard`` → ``start_server``).
"""

from __future__ import annotations

import argparse
from typing import Callable


def _add_server_runtime_args(
    parser, *, build_hint: str = "cd web && npm run build",
    lifecycle_target: str = "all running Hermes web server processes",
) -> None:
    """Runtime flags shared by ``dashboard``, ``webapp``, and ``serve``."""
    parser.add_argument(
        "--port", type=int, default=9119, help="Port (default 9119, 0 for auto-assign by OS)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default 127.0.0.1)")
    parser.add_argument(
        "--insecure", action="store_true",
        help="DEPRECATED / NO-OP. Formerly bypassed auth on a non-loopback "
            "bind. As of the June 2026 hardening it no longer disables "
            "authentication — a public bind always requires an auth provider "
            "(password or OAuth). Bind 127.0.0.1 + tunnel to keep it local.")
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip the web UI build step and serve the existing dist directly. "
            "Useful for non-interactive contexts (Windows Scheduled Tasks, CI) "
            f"where npm may not be available. Pre-build with: {build_hint}")
    parser.add_argument(
        "--isolated", action="store_true",
        help="When launched from a named profile, run a dedicated server scoped "
            "to that profile instead of routing to the machine-level server. "
            "Default behavior is unified: profile launches attach to (or start) "
            "ONE machine-level server and preselect the profile.")
    # Internal: set by the unified-launch re-exec to preselect the launching profile.
    parser.add_argument("--open-profile", dest="open_profile", default="", help=argparse.SUPPRESS)
    # Lifecycle flags win over the start-a-server flags (they exit first). No service
    # manager / PID file: they scan the process table for `hermes dashboard|serve`
    # cmdlines and SIGTERM them — the same path `hermes update` uses.
    parser.add_argument(
        "--stop", action="store_true", help=f"Stop {lifecycle_target} and exit")
    parser.add_argument(
        "--status", action="store_true", help=f"List {lifecycle_target} and exit")


def _configure_serve_parser(parser, *, cmd_dashboard: Callable) -> None:
    """Canonical ``serve`` arguments; shared by the full tree and Desktop's lean hot-path parser."""
    _add_server_runtime_args(parser)
    # Redundant (serve is always headless) but accepted so legacy callers don't error.
    parser.add_argument("--no-open", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--ssh-session-token-file", dest="ssh_session_token_file", metavar="PATH", default=None,
        help="Read a one-shot Desktop SSH session token from PATH")
    parser.add_argument(
        "--ssh-owner-nonce", dest="ssh_owner_nonce", metavar="NONCE", default=None,
        help="Identify a Desktop-owned SSH backend process")
    parser.set_defaults(func=cmd_dashboard, no_open=True, headless_backend=True, command="serve")


def build_serve_parser(
    *, cmd_dashboard: Callable, add_help: bool = True, exit_on_error: bool = True,
) -> argparse.ArgumentParser:
    """Build the standalone parser used by the lean ``serve`` dispatch path."""
    parser = argparse.ArgumentParser(
        prog="hermes serve",
        description="Run the Hermes backend server - the JSON-RPC/WebSocket gateway the "
            "desktop app and remote clients connect to. Headless: it never opens "
            "a browser UI.",
        add_help=add_help, exit_on_error=exit_on_error)
    _configure_serve_parser(parser, cmd_dashboard=cmd_dashboard)
    return parser


def build_dashboard_parser(
    subparsers, *, cmd_dashboard: Callable, cmd_dashboard_register: Callable,
    cmd_webapp: Callable,
) -> None:
    """Attach the ``dashboard``, ``webapp``, and headless ``serve`` commands."""
    dashboard_parser = subparsers.add_parser(
        "dashboard", help="Start the web UI dashboard",
        description="Launch the Hermes Agent web dashboard for managing config, API keys, and sessions",
    )
    _add_server_runtime_args(dashboard_parser)
    dashboard_parser.add_argument(
        "--no-open", action="store_true", help="Don't open browser automatically")
    # Compat shim: desktop shells <= 0.15.x spawn `hermes dashboard --no-open --tui ...`;
    # `--tui` was removed (embedded chat always on). Accept + ignore so an old app with a
    # new CLI doesn't die on "unrecognized arguments". Drop once the app floor is > 0.16.0.
    dashboard_parser.add_argument("--tui", action="store_true", help=argparse.SUPPRESS)
    dashboard_parser.set_defaults(func=cmd_dashboard)

    # `serve`: same gateway as `dashboard`, never opens a browser. Exists so the desktop
    # app / remote backends launch a backend WITHOUT invoking `dashboard` — independent
    # surfaces that merely share this server.
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the Hermes backend server (headless; powers the desktop app and remote backends)",
        description="Run the Hermes backend server — the JSON-RPC/WebSocket gateway the "
            "desktop app and remote clients connect to. Headless: it never opens "
            "a browser UI.")
    _configure_serve_parser(serve_parser, cmd_dashboard=cmd_dashboard)

    # `register` is nested so bare `hermes dashboard` keeps launching the server.
    dashboard_subparsers = dashboard_parser.add_subparsers(dest="dashboard_subcommand")
    dashboard_register_parser = dashboard_subparsers.add_parser(
        "register",
        help="Register a self-hosted dashboard with Nous Portal (writes the OAuth client ID to .env)",
        description="Register this install as a self-hosted dashboard with your Nous "
            "Portal account. Creates an OAuth client, writes "
            "HERMES_DASHBOARD_OAUTH_CLIENT_ID into ~/.hermes/.env, and prints "
            "how to engage the login gate. Requires being logged in (hermes setup).")
    dashboard_register_parser.add_argument(
        "--name", default=None,
        help="Human-readable label for the dashboard (default: an auto-generated name)")
    dashboard_register_parser.add_argument(
        "--redirect-uri", dest="redirect_uri", default=None,
        help="Optional public HTTPS OAuth redirect URI for the dashboard, e.g. "
            "https://hermes.example.com/auth/callback. Omit for localhost-only use.")
    dashboard_register_parser.add_argument(
        "--portal-url", dest="portal_url", default=None,
        help="Override the Nous Portal base URL for registration (default: the "
            "portal you logged into). The access token must be valid at this "
            "portal. Also settable via HERMES_DASHBOARD_PORTAL_URL. Mainly for "
            "testing against a staging/preview portal.")
    dashboard_register_parser.set_defaults(func=cmd_dashboard_register)

    # =========================================================================
    # webapp command — the Desktop workspace rendered in a normal browser
    #
    # The command prepares the renderer and then hands off to the same hardened
    # server process as `dashboard`; it does not create a second backend stack.
    # =========================================================================
    webapp_parser = subparsers.add_parser(
        "webapp",
        help="Start the Hermes Desktop workspace in a browser",
        description=(
            "Launch the current Hermes Desktop workspace in a normal browser, "
            "backed by the authenticated Hermes web server."
        ),
    )
    _add_server_runtime_args(
        webapp_parser,
        build_hint="cd apps/desktop && npm run build:webapp",
        lifecycle_target="running Hermes Webapp processes",
    )
    webapp_parser.add_argument(
        "--no-open", action="store_true", help="Don't open browser automatically"
    )
    webapp_parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the browser-hosted Desktop renderer but do not start the server",
    )
    webapp_parser.add_argument(
        "--force-build",
        action="store_true",
        help="Rebuild the browser-hosted Desktop renderer even when its content stamp matches",
    )
    webapp_parser.set_defaults(
        func=cmd_webapp,
        headless_backend=False,
        webapp_surface=True,
    )
