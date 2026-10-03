"""Observe-only AgentOps plugin entry point.

Phase 1 deliberately registers an operator CLI only.  It does not register
tools, lifecycle hooks, collectors, or any path that could affect a target.
"""

from __future__ import annotations

from plugins.agentops.cli import agentops_main_command, register_cli


def register(ctx) -> None:
    """Register the opt-in AgentOps CLI surface without side effects."""
    ctx.register_cli_command(
        name="agentops",
        help="Observe-only AgentOps control-plane diagnostics",
        setup_fn=register_cli,
        handler_fn=agentops_main_command,
        description="Run the local observe-only AgentOps daemon or diagnostics.",
    )
