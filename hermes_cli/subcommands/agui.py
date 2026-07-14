"""``hermes agui`` subcommand parser.

Standalone launcher for the AG-UI HTTP/SSE server (``agui_adapter``). Handler
injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_agui_parser(subparsers, *, cmd_agui: Callable) -> None:
    """Attach the ``agui`` subcommand to ``subparsers``."""
    agui_parser = subparsers.add_parser(
        "agui",
        help="Run Hermes as an AG-UI HTTP/SSE server",
        description=(
            "Start Hermes as an AG-UI server so AG-UI clients (e.g. CopilotKit) "
            "can drive it over HTTP/SSE. Env-driven, the same as the `hermes-agui` "
            "script (and `python -m agui_adapter`); the flags below are convenience "
            "overrides for the common HERMES_AGUI_* environment variables."
        ),
    )
    agui_parser.add_argument(
        "--host",
        dest="agui_host",
        help="Bind host (default 127.0.0.1; sets HERMES_AGUI_HOST). "
             "A non-loopback host requires --token.",
    )
    agui_parser.add_argument(
        "--port",
        dest="agui_port",
        type=int,
        help="Listen port (default 8000; sets HERMES_AGUI_PORT).",
    )
    agui_parser.add_argument(
        "--token",
        dest="agui_token",
        help="Session token (sets HERMES_AGUI_SESSION_TOKEN; required off-loopback).",
    )
    agui_parser.add_argument(
        "--check",
        action="store_true",
        help="Verify AG-UI dependencies and adapter imports, then exit.",
    )
    agui_parser.set_defaults(func=cmd_agui)
