"""Parser for ``hermes reliability``."""

from __future__ import annotations


def build_reliability_parser(subparsers, *, cmd_reliability):
    parser = subparsers.add_parser(
        "reliability",
        help="Run bounded Hermes reliability checks",
        description=(
            "Inspect profile health, provider blockers, Kanban state, logs, "
            "and state growth. Bounded repairs preserve a backup and avoid "
            "credentials, deploys, merges, and destructive cleanup."
        ),
    )
    sub = parser.add_subparsers(dest="reliability_command")
    check = sub.add_parser("check", help="Run reliability checks")
    check.add_argument("--json", action="store_true", help="Print JSON report")
    check.add_argument(
        "--write-report",
        action="store_true",
        help="Write health/reliability.json under the profile",
    )
    check.add_argument(
        "--apply-bounded",
        action="store_true",
        help="Apply only backup-protected bounded local repairs",
    )
    check.add_argument(
        "--profile-home",
        help="Profile HERMES_HOME to inspect (defaults to current HERMES_HOME)",
    )
    check.add_argument(
        "--dashboard-url",
        help="Dashboard base URL to probe (default: http://127.0.0.1:9120)",
    )
    check.set_defaults(func=cmd_reliability)
    parser.set_defaults(func=cmd_reliability)
    return parser
