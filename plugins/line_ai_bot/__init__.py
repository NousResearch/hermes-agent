from __future__ import annotations

from . import core

_CTX = None


def _ctx_llm():
    if _CTX is None:
        raise RuntimeError("line-ai-bot plugin is not registered yet")
    return _CTX.llm


def register(ctx) -> None:
    global _CTX
    _CTX = ctx
    core.bind_llm_factory(_ctx_llm)

    ctx.register_tool(
        name="line_ai_bot_status",
        toolset="line-ai-bot",
        schema=core.STATUS_SCHEMA,
        handler=core.handle_status,
        check_fn=core.check_available,
        description=core.STATUS_SCHEMA.get("description", ""),
        emoji="L",
    )
    ctx.register_tool(
        name="line_ai_bot_reply",
        toolset="line-ai-bot",
        schema=core.REPLY_SCHEMA,
        handler=core.handle_reply,
        check_fn=core.check_available,
        description=core.REPLY_SCHEMA.get("description", ""),
        emoji="L",
    )
    ctx.register_command(
        "line-ai-bot",
        handler=core.handle_slash,
        description="Generate LINE-ready bot replies through the active Hermes model.",
        args_hint="[status|reply <text>]",
    )
    ctx.register_conversation_plugin(
        "line-ai-bot",
        prompt_builder=core.conversation_prompt,
        matcher=core.matches_conversation_event,
    )
