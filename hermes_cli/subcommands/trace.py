"""``hermes trace`` subcommand parser.

Shows or exports the NeMo Relay events Hermes recorded for a session (see
``hermes_cli/observability/relay_traces.py``). Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_trace_parser(subparsers, *, cmd_trace: Callable) -> None:
    """Attach the ``trace`` subcommand to ``subparsers``."""
    trace_parser = subparsers.add_parser(
        "trace",
        help="Show or export a session's recorded execution trace",
        description=(
            "Print the span tree of the Relay events recorded for a session (subagents "
            "included), or export them as ATOF JSON lines."
        ),
    )
    trace_sub = trace_parser.add_subparsers(dest="trace_action", required=True)

    show_p = trace_sub.add_parser("show", help="Print the session's span tree")
    show_p.add_argument("session", help="Session id or unique prefix")

    export_p = trace_sub.add_parser("export", help="Write the session's Relay events as ATOF JSON lines")
    export_p.add_argument("session", help="Session id or unique prefix")
    export_p.add_argument("--output", "-o", help="Output path (default: stdout). Use '-' for stdout.")

    trace_parser.set_defaults(func=cmd_trace)
