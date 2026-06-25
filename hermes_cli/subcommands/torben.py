"""``hermes torben`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_torben_parser(subparsers, *, cmd_torben: Callable) -> None:
    """Attach the ``torben`` subcommand to ``subparsers``."""
    torben_parser = subparsers.add_parser(
        "torben",
        help="Torben Signal COO operator helpers",
    )
    torben_subparsers = torben_parser.add_subparsers(dest="torben_action")

    ea_brief = torben_subparsers.add_parser(
        "ea-brief",
        help="Generate a conversational EA brief from bounded evidence JSON",
    )
    ea_brief.add_argument("--evidence", required=True, help="Path to evidence JSON")
    ea_brief.add_argument(
        "--ledger",
        default=None,
        help="Action ledger path (default: $HERMES_HOME/state/torben-action-ledger.json)",
    )
    ea_brief.add_argument("--json", action="store_true", help="Print JSON output")
    ea_brief.add_argument(
        "--now",
        default=None,
        help="Optional ISO timestamp for deterministic canaries and tests",
    )

    operating_brief = torben_subparsers.add_parser(
        "operating-brief",
        help="Generate one Signal-facing brief across EA, GTM, and Finance evidence",
    )
    operating_brief.add_argument("--evidence", required=True, help="Path to evidence JSON")
    operating_brief.add_argument(
        "--ledger",
        default=None,
        help="Action ledger path (default: $HERMES_HOME/state/torben-action-ledger.json)",
    )
    operating_brief.add_argument("--json", action="store_true", help="Print JSON output")
    operating_brief.add_argument(
        "--now",
        default=None,
        help="Optional ISO timestamp for deterministic canaries and tests",
    )

    resolve_reply = torben_subparsers.add_parser(
        "resolve-reply",
        help="Resolve a Signal reply against the Torben action ledger",
    )
    resolve_reply.add_argument("reply", nargs="+", help="Reply text to resolve")
    resolve_reply.add_argument(
        "--ledger",
        default=None,
        help="Action ledger path (default: $HERMES_HOME/state/torben-action-ledger.json)",
    )
    resolve_reply.add_argument("--json", action="store_true", help="Print JSON output")

    learn_contact = torben_subparsers.add_parser(
        "learn-contact",
        help="Apply one approved relationship-learning answer to the learned-contact store",
    )
    learn_contact.add_argument("--handle", required=True, help="Relationship-learning action handle")
    learn_contact.add_argument("answer", nargs="+", help="Eric's answer, e.g. 'Kim is an investor...'")
    learn_contact.add_argument(
        "--ledger",
        default=None,
        help="Action ledger path (default: $HERMES_HOME/state/torben-action-ledger.json)",
    )
    learn_contact.add_argument(
        "--relationship-context",
        default=None,
        help="Relationship context path (default: $HERMES_HOME/config/relationship_context.yaml)",
    )
    learn_contact.add_argument("--approved-by", default="signal-reply", help="Approval source for audit history")
    learn_contact.add_argument("--json", action="store_true", help="Print JSON output")

    scopes = torben_subparsers.add_parser(
        "scopes",
        help="Show Torben's hidden operator scopes and autonomy boundaries",
    )
    scopes.add_argument("--json", action="store_true", help="Print JSON output")

    google_accounts = torben_subparsers.add_parser(
        "google-accounts",
        help="List Torben Google Gmail/Calendar accounts",
    )
    google_accounts.add_argument(
        "--config",
        default=None,
        help="Google account config path (default: $HERMES_HOME/config/google_accounts.yaml)",
    )
    google_accounts.add_argument("--json", action="store_true", help="Print JSON output")

    google_auth_url = torben_subparsers.add_parser(
        "google-auth-url",
        help="Create a remote-safe Google OAuth approval URL for one account",
    )
    google_auth_url.add_argument("--account", required=True, help="Google account alias")
    google_auth_url.add_argument(
        "--config",
        default=None,
        help="Google account config path (default: $HERMES_HOME/config/google_accounts.yaml)",
    )
    google_auth_url.add_argument("--json", action="store_true", help="Print JSON output")

    google_auth_code = torben_subparsers.add_parser(
        "google-auth-code",
        help="Exchange a pasted Google redirect URL/code for a Torben token",
    )
    google_auth_code.add_argument("--account", required=True, help="Google account alias")
    google_auth_code.add_argument("code_or_url", help="Pasted redirect URL or OAuth code")
    google_auth_code.add_argument(
        "--config",
        default=None,
        help="Google account config path (default: $HERMES_HOME/config/google_accounts.yaml)",
    )
    google_auth_code.add_argument("--json", action="store_true", help="Print JSON output")

    google_auth_check = torben_subparsers.add_parser(
        "google-auth-check",
        help="Check or refresh Torben Google OAuth tokens",
    )
    google_auth_check.add_argument("--account", default=None, help="Optional Google account alias")
    google_auth_check.add_argument(
        "--config",
        default=None,
        help="Google account config path (default: $HERMES_HOME/config/google_accounts.yaml)",
    )
    google_auth_check.add_argument("--json", action="store_true", help="Print JSON output")

    google_ea_evidence = torben_subparsers.add_parser(
        "google-ea-evidence",
        help="Collect bounded Gmail/Calendar evidence for the Torben EA canary",
    )
    google_ea_evidence.add_argument(
        "--config",
        default=None,
        help="Google account config path (default: $HERMES_HOME/config/google_accounts.yaml)",
    )
    google_ea_evidence.add_argument(
        "--output",
        default=None,
        help="Output evidence path (default: $HERMES_HOME/state/torben-google-ea-evidence-latest.json)",
    )
    google_ea_evidence.add_argument(
        "--days",
        type=int,
        default=2,
        help="Calendar lookahead window in days",
    )
    google_ea_evidence.add_argument(
        "--max-calendar-events",
        type=int,
        default=8,
        help="Max primary-calendar events per account",
    )
    google_ea_evidence.add_argument(
        "--max-email-messages",
        type=int,
        default=5,
        help="Max Gmail message metadata rows per account",
    )
    google_ea_evidence.add_argument(
        "--max-calendar-block-candidates",
        type=int,
        default=3,
        help="Max calendar block candidates to return; use 0 for no cap",
    )
    google_ea_evidence.add_argument(
        "--include-secondary-calendars",
        action="store_true",
        help="Also collect writable non-hidden secondary calendars when available",
    )
    google_ea_evidence.add_argument(
        "--now",
        default=None,
        help="Optional ISO timestamp for deterministic tests",
    )
    google_ea_evidence.add_argument("--json", action="store_true", help="Print JSON output")

    calendar_audit = torben_subparsers.add_parser(
        "calendar-audit",
        help="Audit Google Calendar alignment across Torben accounts",
    )
    calendar_audit.add_argument(
        "--config",
        default=None,
        help="Google account config path (default: $HERMES_HOME/config/google_accounts.yaml)",
    )
    calendar_audit.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: $HERMES_HOME/state/torben-calendar-alignment-audit-latest.json)",
    )
    calendar_audit.add_argument(
        "--brief-output",
        default=None,
        help="Output text report path (default: $HERMES_HOME/state/torben-calendar-alignment-audit-latest.txt)",
    )
    calendar_audit.add_argument("--days", type=int, default=21, help="Calendar lookahead window in days")
    calendar_audit.add_argument(
        "--max-calendar-events",
        type=int,
        default=250,
        help="Max events per collected calendar",
    )
    calendar_audit.add_argument(
        "--max-block-candidates",
        type=int,
        default=250,
        help="Max drift candidates; use 0 for no cap",
    )
    calendar_audit.add_argument(
        "--max-items",
        type=int,
        default=15,
        help="Max drift rows shown in the text report",
    )
    calendar_audit.add_argument(
        "--primary-only",
        action="store_true",
        help="Skip writable secondary-calendar discovery and collect primary calendars only",
    )
    calendar_audit.add_argument(
        "--now",
        default=None,
        help="Optional ISO timestamp for deterministic tests",
    )
    calendar_audit.add_argument("--json", action="store_true", help="Print JSON output")

    morning_brief = torben_subparsers.add_parser(
        "morning-brief",
        help="Render the Morning Brief section from Google EA evidence",
    )
    morning_brief.add_argument("--evidence", required=True, help="Path to evidence JSON")
    morning_brief.add_argument(
        "--output",
        default=None,
        help="Optional output text path",
    )
    morning_brief.add_argument("--json", action="store_true", help="Print JSON output")

    gtm_radar = torben_subparsers.add_parser(
        "gtm-radar",
        help="Adapt the latest Magnus GTM radar artifact into Torben Signal text",
    )
    gtm_radar.add_argument(
        "--radar-path",
        default=None,
        help="Magnus GTM radar JSON path (default: /Users/ericfreeman/magnus/state/gtm-intelligence-radar/latest.json)",
    )
    gtm_radar.add_argument(
        "--state",
        default=None,
        help="Torben adapter dedupe state path (default: $HERMES_HOME/state/torben-gtm-radar-adapter-state.json)",
    )
    gtm_radar.add_argument(
        "--ledger",
        default=None,
        help="Torben action ledger path (default: $HERMES_HOME/state/torben-action-ledger.json)",
    )
    gtm_radar.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: $HERMES_HOME/state/torben-gtm-radar-latest.json when supplied)",
    )
    gtm_radar.add_argument(
        "--text-output",
        default=None,
        help="Output text path (default: $HERMES_HOME/state/torben-gtm-radar-latest.txt when supplied)",
    )
    gtm_radar.add_argument("--max-items", type=int, default=3, help="Max radar findings to surface")
    gtm_radar.add_argument("--preview", action="store_true", help="Do not mark findings delivered")
    gtm_radar.add_argument(
        "--now",
        default=None,
        help="Optional ISO timestamp for deterministic tests",
    )
    gtm_radar.add_argument("--json", action="store_true", help="Print JSON output")

    gtm_reply = torben_subparsers.add_parser(
        "gtm-reply",
        help="Route a GTM radar Signal reply into a draft-only content package",
    )
    gtm_reply.add_argument("reply", nargs="+", help="Reply text to route")
    gtm_reply.add_argument(
        "--ledger",
        default=None,
        help="Torben action ledger path (default: $HERMES_HOME/state/torben-action-ledger.json)",
    )
    gtm_reply.add_argument(
        "--output-dir",
        default=None,
        help="Content package output directory (default: $HERMES_HOME/state/gtm-content-packages)",
    )
    gtm_reply.add_argument(
        "--now",
        default=None,
        help="Optional ISO timestamp for deterministic tests",
    )
    gtm_reply.add_argument("--json", action="store_true", help="Print JSON output")

    inbox_audit = torben_subparsers.add_parser(
        "inbox-audit",
        help="Run a read-only Gmail audit for daily-brief source classification",
    )
    inbox_audit.add_argument(
        "--config",
        default=None,
        help="Google account config path (default: $HERMES_HOME/config/google_accounts.yaml)",
    )
    inbox_audit.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: $HERMES_HOME/state/torben-inbox-audit-latest.json)",
    )
    inbox_audit.add_argument(
        "--report-output",
        default=None,
        help="Output text report path (default: $HERMES_HOME/state/torben-inbox-audit-latest.txt)",
    )
    inbox_audit.add_argument(
        "--relationship-context",
        default=None,
        help="Relationship/source context YAML path (default: $HERMES_HOME/config/relationship_context.yaml)",
    )
    inbox_audit.add_argument("--days", type=int, default=60, help="Gmail lookback window in days")
    inbox_audit.add_argument(
        "--max-messages-per-account",
        type=int,
        default=5000,
        help="Safety cap per account; report warns if reached",
    )
    inbox_audit.add_argument(
        "--max-body-fetches-per-account",
        type=int,
        default=1000,
        help="Max candidate messages per account to fetch bodies/links for",
    )
    inbox_audit.add_argument(
        "--fetch-workers",
        type=int,
        default=8,
        help="Concurrent Gmail fetch workers per account",
    )
    inbox_audit.add_argument(
        "--max-sources",
        type=int,
        default=20,
        help="Max source rows shown in the text report",
    )
    inbox_audit.add_argument(
        "--max-messages",
        type=int,
        default=12,
        help="Max response/deadline/flag sample rows shown in the text report",
    )
    inbox_audit.add_argument(
        "--now",
        default=None,
        help="Optional ISO timestamp for deterministic tests",
    )
    inbox_audit.add_argument("--json", action="store_true", help="Print JSON output")

    secrets_check = torben_subparsers.add_parser(
        "secrets-check",
        help="Validate Torben's 1Password runtime env template",
    )
    secrets_check.add_argument(
        "--env-file",
        default=None,
        help="Runtime env template (default: $HERMES_HOME/runtime.env.op)",
    )
    secrets_check.add_argument(
        "--required-key",
        action="append",
        default=[],
        help="Require this key in the optional 1Password template; can be repeated",
    )
    secrets_check.add_argument("--json", action="store_true", help="Print JSON output")

    auth_check = torben_subparsers.add_parser(
        "auth-check",
        help="Validate Torben's OAuth/MCP-native runtime auth policy",
    )
    auth_check.add_argument(
        "--env-file",
        default=None,
        help="Optional 1Password env template to validate when present",
    )
    auth_check.add_argument("--json", action="store_true", help="Print JSON output")

    route = torben_subparsers.add_parser(
        "route",
        help="Show Torben model routing",
    )
    route.add_argument("--json", action="store_true", help="Print JSON output")

    torben_parser.set_defaults(func=cmd_torben)
