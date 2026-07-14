"""``hermes oauth-broker`` subcommand parser.

Handler injected to avoid importing ``main`` (same pattern as gateway.py).
"""

from __future__ import annotations

from typing import Callable

BROKER_DEFAULT_PORT = 17880


def build_oauth_broker_parser(subparsers, *, cmd_oauth_broker: Callable) -> None:
    parser = subparsers.add_parser(
        "oauth-broker",
        help="Loopback OAuth broker for Codex accounts (A/B/C)",
        description=(
            "Run and manage the local, loopback-only OAuth broker that owns "
            "one Keychain-backed Codex refresh chain per account alias."
        ),
    )
    sub = parser.add_subparsers(dest="oauth_broker_command")

    run_p = sub.add_parser("run", help="Run the broker in the foreground (loopback only)")
    run_p.add_argument("--host", default="127.0.0.1", help="Bind host (loopback only)")
    run_p.add_argument("--port", type=int, default=BROKER_DEFAULT_PORT)

    status_p = sub.add_parser("status", help="Query the local broker /health")
    status_p.add_argument("--port", type=int, default=BROKER_DEFAULT_PORT)

    doctor_p = sub.add_parser(
        "doctor", help="Check Keychain, launchd plist, and broker reachability"
    )
    doctor_p.add_argument("--port", type=int, default=BROKER_DEFAULT_PORT)

    install_p = sub.add_parser(
        "install",
        help="Render the launchd plist and client key (use --apply to load the service)",
    )
    install_p.add_argument("--port", type=int, default=BROKER_DEFAULT_PORT)
    install_p.add_argument(
        "--apply",
        action="store_true",
        help="Actually bootstrap via launchctl after an interactive confirmation",
    )

    uninstall_p = sub.add_parser(
        "uninstall",
        help="Report launchd removal commands (use --apply to boot the service out)",
    )
    uninstall_p.add_argument("--apply", action="store_true")

    auth_p = sub.add_parser(
        "auth", help="Manage per-account OAuth grants in the macOS Keychain"
    )
    auth_sub = auth_p.add_subparsers(dest="oauth_broker_auth_command")
    login_p = auth_sub.add_parser(
        "login", help="Device-code login for one account alias (grant goes to Keychain)"
    )
    login_p.add_argument("alias", choices=["A", "B", "C"])
    status_a = auth_sub.add_parser(
        "status", help="Show present/expiring/healthy booleans per alias"
    )
    status_a.add_argument("alias", nargs="?", choices=["A", "B", "C"])
    status_a.add_argument("--port", type=int, default=BROKER_DEFAULT_PORT)
    logout_p = auth_sub.add_parser(
        "logout", help="Delete one account grant from the Keychain"
    )
    logout_p.add_argument("alias", choices=["A", "B", "C"])
    logout_p.add_argument(
        "--yes", action="store_true", help="Required: confirm the deletion"
    )

    migrate_p = sub.add_parser(
        "migrate", help="Plan (default) or apply the profile broker migration"
    )
    migrate_p.add_argument(
        "--profiles-root",
        required=True,
        help="Root containing profiles/<name>/auth.json (never defaults to a live root)",
    )
    migrate_p.add_argument(
        "--groups", required=True, help="JSON file mapping profile name to A/B/C"
    )
    migrate_p.add_argument("--port", type=int, default=BROKER_DEFAULT_PORT)
    migrate_p.add_argument(
        "--snapshot", required=True, help="Path for the redacted rollback snapshot"
    )
    migrate_p.add_argument("--apply", action="store_true")

    rollback_p = sub.add_parser(
        "rollback", help="Roll a broker migration back from its snapshot"
    )
    rollback_p.add_argument("--profiles-root", required=True)
    rollback_p.add_argument("--snapshot", required=True)
    rollback_p.add_argument(
        "--yes", action="store_true", help="Required: confirm the rollback"
    )

    parser.set_defaults(func=cmd_oauth_broker)
