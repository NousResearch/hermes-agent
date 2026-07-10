"""NotebookLM automation plugin for Hermes implementation logs and X ideas."""

from __future__ import annotations

from . import core
from .cli import notebooklm_command, register_cli

_CTX = None


def _ctx_llm():
    if _CTX is None:
        raise RuntimeError("notebooklm plugin is not registered yet")
    return _CTX.llm


_TOOLS = (
    ("notebooklm_status", core.STATUS_SCHEMA, core.handle_status, "N"),
    ("notebooklm_collect", core.COLLECT_SCHEMA, core.handle_collect, "N"),
    ("notebooklm_brainstorm", core.BRAINSTORM_SCHEMA, core.handle_brainstorm, "N"),
    ("notebooklm_sync", core.SYNC_SCHEMA, core.handle_sync, "N"),
    ("notebooklm_run", core.RUN_SCHEMA, core.handle_run, "N"),
)


def register(ctx) -> None:
    """Register NotebookLM tools, slash command, and CLI command."""
    global _CTX
    _CTX = ctx
    core.bind_llm_factory(_ctx_llm)

    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="notebooklm",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            emoji=emoji,
        )

    ctx.register_command(
        "notebooklm",
        handler=core.handle_slash,
        description="Collect Hermes logs, brainstorm X posts, and sync NotebookLM sources.",
        args_hint="[status|collect|brainstorm|sync|setup-mcp|login|run]",
    )
    ctx.register_cli_command(
        name="notebooklm",
        help="NotebookLM source and X brainstorming automation",
        setup_fn=register_cli,
        handler_fn=notebooklm_command,
        description=(
            "Collect implementation logs and LM-twitterer activity into "
            "NotebookLM-ready sources, generate X post brainstorms, and "
            "sync via Enterprise API or notebooklm-mcp-cli (consumer)."
        ),
    )
