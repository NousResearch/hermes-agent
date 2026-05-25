"""Cursor SDK integration plugin.

Registers a single delegation tool that lets Hermes hand a coding task to the
official Cursor Python SDK. Cursor remains an agent runtime, not a Hermes model
provider, so this lives in the plugin surface instead of the core conversation
loop.
"""

from __future__ import annotations

from .tools import (
    CURSOR_AGENT_SCHEMA,
    check_cursor_sdk_available,
    handle_cursor_agent,
)


def register(ctx) -> None:
    """Register Cursor SDK tools. Called once by the plugin loader."""
    ctx.register_tool(
        name="cursor_agent",
        toolset="cursor_sdk",
        schema=CURSOR_AGENT_SCHEMA,
        handler=handle_cursor_agent,
        check_fn=check_cursor_sdk_available,
        requires_env=["CURSOR_API_KEY"],
        emoji="C",
        max_result_size_chars=100_000,
    )
