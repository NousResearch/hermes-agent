"""``hermes gateway`` and ``hermes proxy`` subcommand parsers.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Both parsers are built together because they shared one inline block (the
``gateway`` section also defined ``proxy``). Handlers injected to avoid
importing ``main``.
"""

from __future__ import annotations

import argparse
from typing import Callable

from hermes_cli.subcommands._shared import add_accept_hooks_flag


def _add_compat_platform_flag(parser: argparse.ArgumentParser) -> None:
    """Accept stale `gateway <verb> --platform X` docs without advertising it.

    Gateway service lifecycle commands operate on the gateway process, not a
    single messaging adapter.  Photon briefly printed a per-platform start
    command during setup; keep that command parseable so users following the
    old hint don't get blocked by argparse before the gateway can start.
    """
    parser.add_argument(
        "--platform",
        dest="platform",
        help=argparse.SUPPRESS,
    )


def build_gateway_parser(subparsers, *, cmd_gateway: Callable, cmd_proxy: Callable) -> None:
    """Attach the ``gateway`` and ``proxy`` subcommands to ``subparsers``."""
    # =========================================================================
    # gateway command
    # =========================================================================
    gateway_parser = subparsers.add_parser(
        "gateway",
        help="Messaging gateway management",
        description="Manage the messaging gateway (Telegram, Discord, WhatsApp, Weixin, and more)",
    )
    gateway_subparsers = gateway_parser.add_subparsers(dest="gateway_command")

    # gateway run (default)
    gateway_run = gateway_subparsers.add_parser(
        "run", help="Run gateway in foreground (recommended for WSL, Docker, Termux)"
    )
    gateway_run.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase stderr log verbosity (-v=INFO, -vv=DEBUG)",
    )
    gateway_run.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress all stderr log output"
    )
    gateway_run.add_argument(
        "--replace",
        action="store_true",
        help="Replace any existing gateway instance (useful for systemd)",
    )
    gateway_run.add_argument(
        "--no-supervise",
        action="store_true",
        help=(
            "Inside the s6-overlay Docker image, normally `gateway run` is "
            "automatically redirected to the supervised s6 service (so the "
            "gateway gets auto-restart on crash, plus a supervised dashboard "
            "if HERMES_DASHBOARD is set). Pass --no-supervise to opt out and "
            "get the historical pre-s6 foreground behavior: the gateway is "
            "the container's main process and the container exits with the "
            "gateway's exit code. No effect outside an s6 container."
        ),
    )
    add_accept_hooks_flag(gateway_run)
    add_accept_hooks_flag(gateway_parser)

    # gateway start
    gateway_start = gateway_subparsers.add_parser(
        "start", help="Start the installed systemd/launchd background service"
    )
    gateway_start.add_argument(
        "--system",
        action="store_true",
        help="Target the Linux system-level gateway service",
    )
    gateway_start.add_argument(
        "--all",
        action="store_true",
        help="Kill ALL stale gateway processes across all profiles before starting",
    )
    _add_compat_platform_flag(gateway_start)

    # gateway stop
    gateway_stop = gateway_subparsers.add_parser("stop", help="Stop gateway service")
    gateway_stop.add_argument(
        "--system",
        action="store_true",
        help="Target the Linux system-level gateway service",
    )
    gateway_stop.add_argument(
        "--all",
        action="store_true",
        help="Stop ALL gateway processes across all profiles",
    )

    # gateway restart
    gateway_restart = gateway_subparsers.add_parser(
        "restart", help="Restart gateway service"
    )
    gateway_restart.add_argument(
        "--system",
        action="store_true",
        help="Target the Linux system-level gateway service",
    )
    gateway_restart.add_argument(
        "--all",
        action="store_true",
        help="Kill ALL gateway processes across all profiles before restarting",
    )
    _add_compat_platform_flag(gateway_restart)

    # gateway status
    gateway_status = gateway_subparsers.add_parser("status", help="Show gateway status")
    gateway_status.add_argument("--deep", action="store_true", help="Deep status check")
    gateway_status.add_argument(
        "-l",
        "--full",
        action="store_true",
        help="Show full, untruncated service/log output where supported",
    )
    gateway_status.add_argument(
        "--system",
        action="store_true",
        help="Target the Linux system-level gateway service",
    )
    _add_compat_platform_flag(gateway_status)

    # gateway install
    gateway_install = gateway_subparsers.add_parser(
        "install", help="Install gateway as a systemd/launchd background service"
    )
    gateway_install.add_argument("--force", action="store_true", help="Force reinstall")
    gateway_install.add_argument(
        "--system",
        action="store_true",
        help="Install as a Linux system-level service (starts at boot)",
    )
    gateway_install.add_argument(
        "--run-as-user",
        dest="run_as_user",
        help="User account the Linux system service should run as",
    )
    gateway_install.add_argument(
        "--start-now",
        dest="start_now",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )
    gateway_install.add_argument(
        "--no-start-now",
        dest="start_now",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    gateway_install.add_argument(
        "--start-on-login",
        dest="start_on_login",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )
    gateway_install.add_argument(
        "--no-start-on-login",
        dest="start_on_login",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    gateway_install.add_argument(
        "--elevated-handoff",
        dest="elevated_handoff",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    # gateway uninstall
    gateway_uninstall = gateway_subparsers.add_parser(
        "uninstall", help="Uninstall gateway service"
    )
    gateway_uninstall.add_argument(
        "--system",
        action="store_true",
        help="Target the Linux system-level gateway service",
    )

    # gateway list
    gateway_subparsers.add_parser("list", help="List all profiles and their gateway status")

    # gateway setup
    gateway_subparsers.add_parser("setup", help="Configure messaging platforms")

    # gateway migrate-legacy
    gateway_migrate_legacy = gateway_subparsers.add_parser(
        "migrate-legacy",
        help="Remove legacy hermes.service units from pre-rename installs",
        description=(
            "Stop, disable, and remove legacy Hermes gateway unit files "
            "(e.g. hermes.service) left over from older installs. Profile "
            "units (hermes-gateway-<profile>.service) and unrelated "
            "third-party services are never touched."
        ),
    )
    gateway_migrate_legacy.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="List what would be removed without doing it",
    )
    gateway_migrate_legacy.add_argument(
        "-y",
        "--yes",
        dest="yes",
        action="store_true",
        help="Skip the confirmation prompt",
    )

    # =========================================================================
    # proxy command — local OpenAI-compatible proxy that attaches the user's
    # OAuth-authenticated provider credentials to outbound requests. Lets
    # external apps (OpenViking, Karakeep, Open WebUI, ...) ride a logged-in
    # subscription without copy-pasting static API keys.
    # =========================================================================
    proxy_parser = subparsers.add_parser(
        "proxy",
        help="Local OpenAI-compatible proxy to OAuth providers",
        description=(
            "Run a local HTTP server that forwards OpenAI-compatible requests "
            "to an OAuth-authenticated provider (e.g. Nous Portal). External "
            "apps can point at the proxy with any bearer token; the proxy "
            "attaches your real credentials."
        ),
    )
    proxy_subparsers = proxy_parser.add_subparsers(dest="proxy_command")

    proxy_start = proxy_subparsers.add_parser(
        "start", help="Run the proxy in the foreground"
    )
    proxy_start.add_argument(
        "--provider",
        default="nous",
        help="Upstream provider: nous or xai (default: nous). See `hermes proxy providers`.",
    )
    proxy_start.add_argument(
        "--host",
        default=None,
        help="Bind address (default: 127.0.0.1). Use 0.0.0.0 to expose on LAN.",
    )
    proxy_start.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (default: 8645)",
    )

    proxy_subparsers.add_parser(
        "status", help="Show which proxy upstreams are ready"
    )
    proxy_subparsers.add_parser(
        "providers", help="List available proxy upstream providers"
    )
    proxy_parser.set_defaults(func=cmd_proxy)
    gateway_parser.set_defaults(func=cmd_gateway)
