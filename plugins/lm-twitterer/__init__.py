"""LM-twitterer Hermes plugin.

This is a standalone Hermes plugin inspired by
https://github.com/soichi11208/LM-twitterer.  It keeps model calls on the
Hermes side through ctx.llm, so the active Hermes provider can be OpenCode,
llama-cpp, OpenAI-compatible endpoints, xAI/Grok, or anything else Hermes
supports.
"""

from __future__ import annotations

from . import core
from .cli import lm_twitterer_command, register_cli


_CTX = None


def _ctx_llm():
    if _CTX is None:
        raise RuntimeError("lm-twitterer plugin is not registered yet")
    return _CTX.llm


def register(ctx) -> None:
    global _CTX
    _CTX = ctx
    core.bind_llm_factory(_ctx_llm)

    ctx.register_tool(
        name="lm_twitterer_post",
        toolset="lm_twitterer",
        schema=core.POST_SCHEMA,
        handler=core.handle_post,
        check_fn=core.check_available,
        emoji="X",
    )
    ctx.register_tool(
        name="lm_twitterer_reply_mentions",
        toolset="lm_twitterer",
        schema=core.REPLY_SCHEMA,
        handler=core.handle_reply_mentions,
        check_fn=core.check_available,
        emoji="@",
    )
    ctx.register_tool(
        name="lm_twitterer_status",
        toolset="lm_twitterer",
        schema=core.STATUS_SCHEMA,
        handler=core.handle_status,
        check_fn=core.check_available,
        emoji="i",
    )
    ctx.register_tool(
        name="lm_twitterer_auth_check",
        toolset="lm_twitterer",
        schema=core.AUTH_CHECK_SCHEMA,
        handler=core.handle_auth_check,
        check_fn=core.check_available,
        emoji="key",
    )
    ctx.register_tool(
        name="lm_twitterer_mentions",
        toolset="lm_twitterer",
        schema=core.MENTIONS_SCHEMA,
        handler=core.handle_mentions,
        check_fn=core.check_available,
        emoji="@",
    )

    ctx.register_command(
        "lm-twitterer",
        handler=core.handle_slash,
        description="Generate X posts and reply to mentions.",
        args_hint="[status|auth-check|mentions|post|replies|whitelist]",
    )
    ctx.register_cli_command(
        name="lm-twitterer",
        help="X posting and mention-reply automation",
        setup_fn=register_cli,
        handler_fn=lm_twitterer_command,
        description="Generate posts, reply to mentions, and install cron jobs.",
    )
