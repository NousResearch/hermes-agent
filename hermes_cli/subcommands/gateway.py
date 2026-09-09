"""``hermes gateway`` and ``hermes proxy`` subcommand parsers."""

from __future__ import annotations

import argparse
from typing import Callable

from hermes_cli.subcommands._shared import add_accept_hooks_flag


def _flag(parser, *names, help, **kw):
    parser.add_argument(*names, action="store_true", help=help, **kw)


def _add_compat_platform_flag(parser: argparse.ArgumentParser) -> None:
    """Accept stale `gateway <verb> --platform X` docs without advertising it.

    Lifecycle commands operate on the gateway process, not one adapter; Photon briefly
    printed a per-platform start command during setup, so keep it parseable.
    """
    parser.add_argument("--platform", dest="platform", help=argparse.SUPPRESS)


def _add_system_flag(parser: argparse.ArgumentParser) -> None:
    _flag(parser, "--system", help="Target the Linux system-level gateway service")


def build_gateway_parser(
    subparsers, *, cmd_gateway: Callable, cmd_proxy: Callable, cmd_gateway_enroll: Callable
) -> None:
    """Attach the ``gateway`` and ``proxy`` subcommands to ``subparsers``."""
    gateway_parser = subparsers.add_parser("gateway", help="Messaging gateway management",
        description="Manage the messaging gateway (Telegram, Discord, WhatsApp, Weixin, and more)")
    gateway_subparsers = gateway_parser.add_subparsers(dest="gateway_command")

    gateway_run = gateway_subparsers.add_parser(
        "run", help="Run gateway in foreground (recommended for WSL, Docker, Termux)")
    gateway_run.add_argument("-v", "--verbose", action="count", default=0,
        help="Increase stderr log verbosity (-v=INFO, -vv=DEBUG)")
    _flag(gateway_run, "-q", "--quiet", help="Suppress all stderr log output")
    _flag(
        gateway_run, "--replace", help="Replace any existing gateway instance (useful for systemd)")
    _flag(gateway_run, "--force",
        help="Start a foreground gateway even when a systemd/launchd/s6 service "
            "already supervises this profile. Without --force, the command "
            "refuses because a second dispatcher escapes the service and can "
            "corrupt shared gateway state.")
    _flag(gateway_run, "--no-supervise",
        help="Inside the s6-overlay Docker image, normally `gateway run` is "
            "automatically redirected to the supervised s6 service (so the "
            "gateway gets auto-restart on crash, plus a supervised dashboard "
            "if HERMES_DASHBOARD is set). Pass --no-supervise to opt out and "
            "get the historical pre-s6 foreground behavior: the gateway is "
            "the container's main process and the container exits with the "
            "gateway's exit code. No effect outside an s6 container.")
    _flag(gateway_run, "--external-supervisor",
        help="Declare that an external process manager owns this foreground "
            "gateway. In-chat restarts and updates exit back to that manager "
            "instead of spawning a detached replacement. Use this when a "
            "launchd/systemd wrapper strips its native environment markers.")
    add_accept_hooks_flag(gateway_run)
    add_accept_hooks_flag(gateway_parser)

    gateway_start = gateway_subparsers.add_parser(
        "start", help="Start the installed systemd/launchd background service")
    _add_system_flag(gateway_start)
    _flag(gateway_start, "--all",
        help="Kill ALL stale gateway processes across all profiles before starting")
    _add_compat_platform_flag(gateway_start)

    gateway_stop = gateway_subparsers.add_parser("stop", help="Stop gateway service")
    _add_system_flag(gateway_stop)
    _flag(gateway_stop, "--all", help="Stop ALL gateway processes across all profiles")

    gateway_restart = gateway_subparsers.add_parser("restart", help="Restart gateway service")
    _add_system_flag(gateway_restart)
    _flag(gateway_restart, "--all",
        help="Kill ALL gateway processes across all profiles before restarting")
    _add_compat_platform_flag(gateway_restart)

    gateway_status = gateway_subparsers.add_parser("status", help="Show gateway status")
    _flag(gateway_status, "--deep", help="Deep status check")
    _flag(gateway_status, "-l", "--full",
        help="Show full, untruncated service/log output where supported")
    _add_system_flag(gateway_status)
    _add_compat_platform_flag(gateway_status)

    gateway_install = gateway_subparsers.add_parser(
        "install", help="Install gateway as a systemd/launchd background service")
    _flag(gateway_install, "--force", help="Force reinstall")
    _flag(gateway_install, "--system",
        help="Install as a Linux system-level service (starts at boot)")
    gateway_install.add_argument("--run-as-user", dest="run_as_user",
        help="User account the Linux system service should run as")
    _flag(gateway_install, "--start-now", dest="start_now", default=None,
        help="Start the gateway service immediately after installing")
    gateway_install.add_argument("--no-start-now", dest="start_now", action="store_false",
        help="Do not start the gateway service after installing")
    _flag(gateway_install, "--start-on-login", dest="start_on_login", default=None,
        help="Enable the service to start automatically on login/boot")
    gateway_install.add_argument("--no-start-on-login", dest="start_on_login", action="store_false",
        help="Do not enable the service to start on login/boot")
    _flag(gateway_install, "--elevated-handoff", dest="elevated_handoff", help=argparse.SUPPRESS)

    gateway_uninstall = gateway_subparsers.add_parser("uninstall", help="Uninstall gateway service")
    _add_system_flag(gateway_uninstall)

    gateway_subparsers.add_parser("list", help="List all profiles and their gateway status")

    gateway_subparsers.add_parser("setup", help="Configure messaging platforms")

    gateway_migrate_legacy = gateway_subparsers.add_parser(
        "migrate-legacy", help="Remove legacy hermes.service units from pre-rename installs",
        description="Stop, disable, and remove legacy Hermes gateway unit files "
            "(e.g. hermes.service) left over from older installs. Profile "
            "units (hermes-gateway-<profile>.service) and unrelated "
            "third-party services are never touched.")
    _flag(gateway_migrate_legacy, "--dry-run", dest="dry_run",
        help="List what would be removed without doing it")
    _flag(gateway_migrate_legacy, "-y", "--yes", dest="yes", help="Skip the confirmation prompt")

    # enroll: redeem a single-use connector token for the per-gateway secret + per-tenant
    # delivery key, written to .env. See docs/relay-connector-contract.md. EXPERIMENTAL.
    gateway_enroll = gateway_subparsers.add_parser("enroll",
        help="Enroll this gateway with a relay connector (writes relay auth creds to .env)",
        description="Redeem a single-use enrollment token with a relay connector. "
            "Authenticates as your Nous Portal account (the connector derives the "
            "authoritative tenant from it), mints this gateway's per-gateway secret "
            "and per-tenant delivery key, and writes GATEWAY_RELAY_ID / "
            "GATEWAY_RELAY_SECRET / GATEWAY_RELAY_DELIVERY_KEY into ~/.hermes/.env. "
            "Requires being logged in (hermes setup). Not available in managed installs.")
    gateway_enroll.add_argument("--token", default=None,
        help="The single-use enrollment token from the connector (delivered with "
            "your gateway config). Also settable via GATEWAY_RELAY_ENROLL_TOKEN.")
    gateway_enroll.add_argument("--connector-url", dest="connector_url", default=None,
        help="The connector base/relay URL, e.g. wss://connector.example.com/relay "
            "or https://connector.example.com. Also settable via GATEWAY_RELAY_URL "
            "/ gateway.relay_url in config.yaml.")
    gateway_enroll.add_argument("--gateway-id", dest="gateway_id", default=None,
        help="A stable id for this gateway instance (kill-switch granularity). "
            "Defaults to gw-<hostname>.")
    gateway_enroll.add_argument("--wake-url", dest="wake_url", default=None,
        help="Phase 5 §5.2 wake URL: a reachable URL the connector pokes "
            "(payload-free GET) to wake this gateway when buffered work arrives "
            "while it's idle/suspended, so it reconnects and drains. Persisted as "
            "GATEWAY_RELAY_WAKE_URL in ~/.hermes/.env and forwarded at provision. "
            "Optional — without it the gateway still drains whenever it next "
            "reconnects on its own.")
    gateway_enroll.set_defaults(func=cmd_gateway_enroll)

    # proxy: local OpenAI-compatible proxy attaching the user's OAuth provider credentials,
    # so external apps (Open WebUI, Karakeep, ...) ride a logged-in subscription.
    proxy_parser = subparsers.add_parser(
        "proxy", help="Local OpenAI-compatible proxy to OAuth providers",
        description="Run a local HTTP server that forwards OpenAI-compatible requests "
            "to an OAuth-authenticated provider (e.g. Nous Portal). External "
            "apps can point at the proxy with any bearer token; the proxy "
            "attaches your real credentials.")
    proxy_subparsers = proxy_parser.add_subparsers(dest="proxy_command")

    proxy_start = proxy_subparsers.add_parser("start", help="Run the proxy in the foreground")
    proxy_start.add_argument("--provider", default="nous",
        help="Upstream provider: nous or xai (default: nous). See `hermes proxy providers`.")
    proxy_start.add_argument("--host", default=None,
        help="Bind address (default: 127.0.0.1). Use 0.0.0.0 to expose on LAN.")
    proxy_start.add_argument("--port", type=int, default=None, help="Bind port (default: 8645)")

    proxy_subparsers.add_parser("status", help="Show which proxy upstreams are ready")
    proxy_subparsers.add_parser("providers", help="List available proxy upstream providers")
    proxy_parser.set_defaults(func=cmd_proxy)
    gateway_parser.set_defaults(func=cmd_gateway)
