"""``hermes monitoring`` subcommand parser.

Gateway monitoring control and inspection. ``status`` shows whether the
gateway health & diagnostics export is enabled, where it points, and the
redaction posture.

The handler is injected to avoid importing ``main`` (mirrors the insights
subcommand).
"""

from __future__ import annotations

from typing import Callable


def build_monitoring_parser(subparsers, *, cmd_monitoring: Callable) -> None:
    """Attach the ``monitoring`` subcommand (with actions) to ``subparsers``."""
    p = subparsers.add_parser(
        "monitoring",
        help="Inspect gateway monitoring (health & diagnostics export)",
        description=(
            "Gateway monitoring: service health metrics plus redacted "
            "diagnostics, exported over OTLP to an operator-configured "
            "endpoint. Content-free by construction — no prompts, messages, "
            "tool args/results, or usage analytics. Configure under "
            "monitoring.* in config.yaml."
        ),
    )
    sub = p.add_subparsers(dest="monitoring_action")

    sub.add_parser(
        "status",
        help="Show monitoring settings, export state, and redaction posture",
    )

    p.set_defaults(func=cmd_monitoring)
