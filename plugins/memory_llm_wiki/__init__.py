"""Hermes memory -> LLM Wiki plugin."""

from __future__ import annotations

from . import core


def register(ctx) -> None:
    ctx.register_tool(
        name="memory_llm_wiki_status",
        toolset="memory_llm_wiki",
        schema=core.STATUS_SCHEMA,
        handler=core.handle_status,
        check_fn=core.check_available,
        emoji="🧠",
    )
    ctx.register_tool(
        name="memory_llm_wiki_export",
        toolset="memory_llm_wiki",
        schema=core.EXPORT_SCHEMA,
        handler=core.handle_export,
        check_fn=core.check_available,
        emoji="🧩",
    )
