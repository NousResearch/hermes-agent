from __future__ import annotations

from .cli import hermes_gpt_command, register_cli


def register(ctx) -> None:
    ctx.register_cli_command(
        name="hermes-gpt",
        help="Run the Hermes GPT local MCP sidecar",
        setup_fn=register_cli,
        handler_fn=hermes_gpt_command,
        description=(
            "Expose selected local Hermes Agent capabilities through a "
            "local-dev MCP server with write, memory-write, terminal, and "
            "session-search features gated by environment variables."
        ),
    )
