"""SurfSense plugin for self-hosted NotebookLM-style knowledge bases."""

from __future__ import annotations

from . import core
from .cli import register_cli, surfsense_command


_TOOLS = (
    ("surfsense_status", core.STATUS_SCHEMA, core.handle_status, "S"),
    ("surfsense_login", core.LOGIN_SCHEMA, core.handle_login, "S"),
    ("surfsense_searchspaces", core.SEARCHSPACES_SCHEMA, core.handle_searchspaces, "S"),
    ("surfsense_upload", core.UPLOAD_SCHEMA, core.handle_upload, "S"),
    ("surfsense_search", core.SEARCH_SCHEMA, core.handle_search, "S"),
    ("surfsense_ask", core.ASK_SCHEMA, core.handle_ask, "S"),
    ("surfsense_video_plan", core.VIDEO_PLAN_SCHEMA, core.handle_video_plan, "S"),
    ("surfsense_video_mux", core.VIDEO_MUX_SCHEMA, core.handle_video_mux, "S"),
)


def register(ctx) -> None:
    """Register SurfSense tools, slash command, and CLI command."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="surfsense",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            emoji=emoji,
        )

    ctx.register_command(
        "surfsense",
        handler=core.handle_slash,
        description="Use a self-hosted SurfSense knowledge base from Hermes.",
        args_hint="[status|login|spaces|search|ask|video-plan]",
    )
    ctx.register_cli_command(
        name="surfsense",
        help="Self-hosted SurfSense NotebookLM-style knowledge base",
        setup_fn=register_cli,
        handler_fn=surfsense_command,
        description=(
            "Check SurfSense readiness, save auth, list search spaces, upload "
            "documents, search sources, ask cited questions, and prepare "
            "NotebookLM-style video artifacts with optional LLM-wiki, "
            "codegraph, sleep, memory, AITuber OnAir, irodoriTTS, and MP4 audio layers."
        ),
    )
