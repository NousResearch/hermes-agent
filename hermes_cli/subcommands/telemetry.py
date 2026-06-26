"""``hermes telemetry`` subcommand parser.

Telemetry control and inspection. ``preview`` shows the per-run summary events that
would be produced for aggregate metrics; there is no uploader, so it terminates as a
local view.

The handler is injected to avoid importing ``main`` (mirrors the insights subcommand).
"""

from __future__ import annotations

from typing import Callable


def build_telemetry_parser(subparsers, *, cmd_telemetry: Callable) -> None:
    """Attach the ``telemetry`` subcommand (with actions) to ``subparsers``."""
    p = subparsers.add_parser(
        "telemetry",
        help="Inspect local telemetry and export it",
        description=(
            "Local-first telemetry. Local telemetry records observability on this "
            "machine. Aggregate metrics are opt-in (set telemetry.consent_state via "
            "`hermes config set`); they have no uploader and are shown only via `preview`."
        ),
    )
    sub = p.add_subparsers(dest="telemetry_action")

    sub.add_parser("status", help="Show telemetry settings, consent state, and local data volume")

    prev = sub.add_parser(
        "preview",
        help="Show the aggregate events that would be produced (computed locally, not uploaded)",
    )
    prev.add_argument("--days", type=int, default=30, help="Window to roll up (default: 30)")
    prev.add_argument("--limit", type=int, default=10, help="Max events to print (default: 10)")
    prev.add_argument("--json", action="store_true", help="Print raw JSON events")

    exp = sub.add_parser(
        "export",
        help="Export local telemetry (and optional content) to a file, stream, or OTLP endpoint",
    )
    exp.add_argument("--out", help="Output file path (use - for stdout). Not needed with --otlp.")
    exp.add_argument("--format", dest="fmt", choices=["ndjson", "json"], default="ndjson",
                     help="Output format (default: ndjson)")
    exp.add_argument("--since", type=int, default=0,
                     help="Only telemetry from the last N days (0 = all)")
    exp.add_argument("--include-content", action="store_true",
                     help="Include session/message content (requires telemetry.trajectories.enabled). "
                          "Secrets always redacted; PII per telemetry.content_redaction.")
    exp.add_argument("--otlp", action="store_true",
                     help="Export to the configured OTLP endpoint (telemetry.export.otlp.*) "
                          "instead of a file. Requires the optional 'otlp' extra.")

    p.set_defaults(func=cmd_telemetry)
