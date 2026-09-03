"""``hermes credentials`` subcommand parser.

Read-only credential dependency map — see ``hermes_cli.credential_impact``
for the implementation. Handler is injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_credentials_parser(subparsers, *, cmd_credentials: Callable) -> None:
    """Attach the ``credentials`` subcommand to ``subparsers``."""
    credentials_parser = subparsers.add_parser(
        "credentials",
        help="Inspect which providers, tasks, MCP servers, and plugins "
        "depend on a credential env var",
    )
    credentials_subparsers = credentials_parser.add_subparsers(
        dest="credentials_action"
    )

    impact = credentials_subparsers.add_parser(
        "impact",
        help="Show every declared consumer of an env-var credential",
    )
    impact.add_argument(
        "var", help="Env var name to inspect (for example: OPENAI_API_KEY)"
    )
    impact.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON"
    )

    credentials_parser.set_defaults(func=cmd_credentials)
