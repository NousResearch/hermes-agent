"""OpenAI Agents SDK bridge plugin for Hermes."""

from __future__ import annotations

from plugins.openai_agents.tools import (
    OPENAI_AGENTS_ARCHITECTURE_SCHEMA,
    OPENAI_AGENTS_EXECUTE_SCHEMA,
    OPENAI_AGENTS_REVIEW_SCHEMA,
    OPENAI_AGENTS_RUN_SCHEMA,
    OPENAI_AGENTS_VERIFY_SCHEMA,
    _check_openai_agents_available,
    _handle_openai_agents_architecture,
    _handle_openai_agents_execute,
    _handle_openai_agents_review,
    _handle_openai_agents_run,
    _handle_openai_agents_verify,
)

_TOOLS = (
    ("openai_agents_review", OPENAI_AGENTS_REVIEW_SCHEMA, _handle_openai_agents_review, "🔎"),
    ("openai_agents_execute", OPENAI_AGENTS_EXECUTE_SCHEMA, _handle_openai_agents_execute, "🧬"),
    ("openai_agents_verify", OPENAI_AGENTS_VERIFY_SCHEMA, _handle_openai_agents_verify, "✅"),
    ("openai_agents_architecture", OPENAI_AGENTS_ARCHITECTURE_SCHEMA, _handle_openai_agents_architecture, "🏗️"),
    # Backward-compatible alias for the initial bridge tool.
    ("openai_agents_run", OPENAI_AGENTS_RUN_SCHEMA, _handle_openai_agents_run, "🧬"),
)


def register(ctx) -> None:
    """Register governed OpenAI Agents SDK worker lanes."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="openai_agents",
            schema=schema,
            handler=handler,
            check_fn=_check_openai_agents_available,
            requires_env=["OPENAI_API_KEY"],
            emoji=emoji,
        )
