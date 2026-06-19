"""Investment assistant plugin for stateful portfolio workflows."""

from __future__ import annotations

__all__ = [
    "IA_PORTFOLIO_WORKFLOW_SCHEMA",
    "handle_ia_portfolio_workflow",
    "transform_llm_output",
    "register",
]


def register(ctx) -> None:
    """Register investment assistant tools. Called by Hermes plugin loader."""
    from .tools import (
        IA_PORTFOLIO_WORKFLOW_SCHEMA,
        handle_ia_portfolio_workflow,
        transform_llm_output,
    )

    ctx.register_tool(
        name="ia_portfolio_workflow",
        toolset="investment_assistant",
        schema=IA_PORTFOLIO_WORKFLOW_SCHEMA,
        handler=handle_ia_portfolio_workflow,
    )
    ctx.register_hook("transform_llm_output", transform_llm_output)


def __getattr__(name: str):
    if name in {
        "IA_PORTFOLIO_WORKFLOW_SCHEMA",
        "handle_ia_portfolio_workflow",
        "transform_llm_output",
    }:
        from . import tools

        return getattr(tools, name)
    raise AttributeError(name)
