"""Teams chat context plugin.

Registers operator-facing CLI surfaces. Retrieval is exposed through the
``plugins.teams_context.mcp_server`` stdio server configured under
``mcp_servers``.
"""

from __future__ import annotations

from plugins.teams_context.cli import register_cli, teams_context_command
from plugins.teams_context.relay_adapter import register_platform as register_relay_platform


def register(ctx) -> None:
    register_relay_platform(ctx)
    ctx.register_cli_command(
        name="teams-context",
        help="Capture and search selected Microsoft Teams chats",
        setup_fn=register_cli,
        handler_fn=teams_context_command,
        description=(
            "Backfill selected Teams chats, manage Graph subscriptions, "
            "validate configuration, and expose local chat context through MCP."
        ),
    )
