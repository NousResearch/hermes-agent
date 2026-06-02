"""Hermes plugin: ClawTeam.

Two surfaces share the same CLI driver:
  - dashboard/plugin_api.py — FastAPI router for the UI tab
  - tools.py + register(ctx) below — agent-facing tools the LLM can call

Shared helper: _clawteam_cli.py (binary discovery, name validation,
subprocess invocation, JSON parsing, structured errors).
"""

from __future__ import annotations

from . import tools as _tools


def register(ctx) -> None:
    """Register all ClawTeam tools with the Hermes plugin context.

    Called once by the plugin loader when this plugin is enabled.
    """
    for name, schema, handler in _tools.TOOLS:
        ctx.register_tool(
            name=name,
            toolset="clawteam",
            schema=schema,
            handler=handler,
            check_fn=_tools.check_clawteam_available,
            emoji="🤝",
        )
