"""``hermes dashboard`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

import argparse
from typing import Callable


def build_dashboard_parser(
    subparsers, *, cmd_dashboard: Callable, cmd_dashboard_register: Callable
) -> None:
    """Attach the ``dashboard`` subcommand (and its ``register`` action)."""
    # =========================================================================
    # dashboard command
    # =========================================================================
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Start the web UI dashboard",
        description="Launch the Hermes Agent web dashboard for managing config, API keys, and sessions",
    )
    dashboard_parser.add_argument(
        "--port", type=int, default=9119, help="Port (default 9119, 0 for auto-assign by OS)"
    )
    dashboard_parser.add_argument(
        "--host", default="127.0.0.1", help="Host (default 127.0.0.1)"
    )
    dashboard_parser.add_argument(
        "--no-open", action="store_true", help="Don't open browser automatically"
    )
    dashboard_parser.add_argument(
        "--insecure",
        action="store_true",
        help=(
            "DEPRECATED / NO-OP. Formerly bypassed dashboard auth on a "
            "non-loopback bind. As of the June 2026 hardening it no longer "
            "disables authentication — a public bind always requires an auth "
            "provider (password or OAuth). Bind 127.0.0.1 + tunnel to keep it "
            "local."
        ),
    )
    dashboard_parser.add_argument(
        "--skip-build",
        action="store_true",
        help=(
            "Skip the web UI build step and serve the existing dist directly. "
            "Useful for non-interactive contexts (Windows Scheduled Tasks, CI) "
            "where npm may not be available. Pre-build with: cd web && npm run build"
        ),
    )
    dashboard_parser.add_argument(
        "--isolated",
        action="store_true",
        help=(
            "When launched from a named profile (e.g. `worker dashboard`), run "
            "a dedicated dashboard server scoped to that profile instead of "
            "routing to the machine dashboard. Default behavior is unified: "
            "profile launches attach to (or start) ONE machine-level dashboard "
            "and preselect the profile in the UI's profile switcher."
        ),
    )
    # Internal flag set by the unified-launch re-exec (cmd_dashboard) to
    # preselect the launching profile in the SPA switcher. Hidden from
    # --help: users get this behavior automatically via `<profile> dashboard`.
    dashboard_parser.add_argument(
        "--open-profile",
        dest="open_profile",
        default="",
        help=argparse.SUPPRESS,
    )
    # Lifecycle flags — mutually exclusive with each other and with the
    # start-a-server flags above (if both are passed, --stop / --status win
    # because they exit before the server is started).  The dashboard has
    # no service manager and no PID file, so these scan the process table
    # for `hermes dashboard` cmdlines and SIGTERM them directly — the same
    # path `hermes update` uses to clean up stale dashboards.
    dashboard_parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop all running hermes dashboard processes and exit",
    )
    dashboard_parser.add_argument(
        "--status",
        action="store_true",
        help="List running hermes dashboard processes and exit",
    )
    # Backward-compat shim: older Hermes desktop app shells (<= 0.15.x) spawn the
    # backend as `hermes dashboard --no-open --tui --host ... --port ...`. The
    # `--tui` flag was removed from this subcommand in cae6b5486 (embedded chat is
    # always on now). When a user's CLI updates past that commit but their desktop
    # app binary has not, argparse used to hard-error with "unrecognized arguments:
    # --tui" and exit(2) — the backend died before becoming ready and the GUI just
    # showed "Hermes couldn't start" with no actionable cause. Accept and silently
    # ignore the flag so an old app + new CLI degrades gracefully instead of
    # bricking. Hidden from --help; safe to delete once the floor app version is
    # well past 0.16.0.
    dashboard_parser.add_argument(
        "--tui",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    dashboard_parser.set_defaults(func=cmd_dashboard)

    # `hermes dashboard register` — register a self-hosted dashboard OAuth
    # client with Nous Portal and write the client_id into ~/.hermes/.env.
    # Nested subparser so bare `hermes dashboard` keeps launching the server
    # (set_defaults(func=cmd_dashboard) above remains the default).
    dashboard_subparsers = dashboard_parser.add_subparsers(
        dest="dashboard_subcommand"
    )
    dashboard_register_parser = dashboard_subparsers.add_parser(
        "register",
        help="Register a self-hosted dashboard with Nous Portal (writes the OAuth client ID to .env)",
        description=(
            "Register this install as a self-hosted dashboard with your Nous "
            "Portal account. Creates an OAuth client, writes "
            "HERMES_DASHBOARD_OAUTH_CLIENT_ID into ~/.hermes/.env, and prints "
            "how to engage the login gate. Requires being logged in (hermes setup)."
        ),
    )
    dashboard_register_parser.add_argument(
        "--name",
        default=None,
        help="Human-readable label for the dashboard (default: an auto-generated name)",
    )
    dashboard_register_parser.add_argument(
        "--redirect-uri",
        dest="redirect_uri",
        default=None,
        help=(
            "Optional public HTTPS OAuth redirect URI for the dashboard, e.g. "
            "https://hermes.example.com/auth/callback. Omit for localhost-only use."
        ),
    )
    dashboard_register_parser.add_argument(
        "--portal-url",
        dest="portal_url",
        default=None,
        help=(
            "Override the Nous Portal base URL for registration (default: the "
            "portal you logged into). The access token must be valid at this "
            "portal. Also settable via HERMES_DASHBOARD_PORTAL_URL. Mainly for "
            "testing against a staging/preview portal."
        ),
    )
    dashboard_register_parser.set_defaults(func=cmd_dashboard_register)
