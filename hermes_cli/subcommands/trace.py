"""``hermes trace`` subcommand parser.

Exports a session's reconstructed span tree to a portable trace file (OTLP/JSON
for Phoenix or any OpenTelemetry backend, or Chrome Trace format for Perfetto),
or prints a quick summary tree to the terminal. Handler injected to avoid
importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_trace_parser(subparsers, *, cmd_trace: Callable) -> None:
    """Attach the ``trace`` subcommand to ``subparsers``."""
    trace_parser = subparsers.add_parser(
        "trace",
        help="Export or inspect a session's execution trace",
        description=(
            "Reconstruct an OpenTelemetry-style span tree for any session "
            "(including its subagents) from the session store, and export it to "
            "a standard trace file or print a summary."
        ),
    )
    trace_sub = trace_parser.add_subparsers(dest="trace_action", required=True)

    export_p = trace_sub.add_parser(
        "export",
        help="Write a session trace to a file (OTLP/JSON or Chrome Trace)",
    )
    export_p.add_argument("session", help="Session id or unique prefix")
    export_p.add_argument(
        "--format",
        "-f",
        choices=["otlp", "chrome"],
        default="otlp",
        help="otlp = OTLP/JSON for Phoenix; chrome = Perfetto/chrome://tracing",
    )
    export_p.add_argument(
        "--output",
        "-o",
        help="Output path (default: stdout). Use '-' for stdout.",
    )
    export_p.add_argument(
        "--no-subagents",
        action="store_true",
        help="Exclude subagent (delegate) descendant sessions",
    )

    show_p = trace_sub.add_parser(
        "show",
        help="Print a summary span tree for a session to the terminal",
    )
    show_p.add_argument("session", help="Session id or unique prefix")
    show_p.add_argument(
        "--no-subagents",
        action="store_true",
        help="Exclude subagent (delegate) descendant sessions",
    )

    trace_parser.set_defaults(func=cmd_trace)
