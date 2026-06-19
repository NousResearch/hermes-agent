"""AITuber OnAir bridge for Hermes."""

from __future__ import annotations

from . import core
from .cli import aituber_onair_command, register_cli

_CTX = None


def _ctx_llm():
    if _CTX is None:
        raise RuntimeError("aituber-onair plugin is not registered yet")
    return _CTX.llm


_TOOLS = (
    ("aituber_onair_status", core.STATUS_SCHEMA, core.handle_status, "A"),
    (
        "aituber_onair_configure_hakua",
        core.CONFIGURE_HAKUA_SCHEMA,
        core.handle_configure_hakua,
        "A",
    ),
    ("aituber_onair_prepare", core.PREPARE_SCHEMA, core.handle_prepare, "A"),
    ("aituber_onair_start", core.START_SCHEMA, core.handle_start, "A"),
    ("aituber_onair_stop", core.STOP_SCHEMA, core.handle_stop, "A"),
    ("aituber_onair_tts_status", core.TTS_STATUS_SCHEMA, core.handle_tts_status, "A"),
    ("aituber_onair_start_tts", core.START_TTS_SCHEMA, core.handle_start_tts, "A"),
    ("aituber_onair_speak", core.SPEAK_SCHEMA, core.handle_speak, "A"),
    ("aituber_onair_say", core.SAY_SCHEMA, core.handle_say, "A"),
    (
        "aituber_onair_context_status",
        core.CONTEXT_STATUS_SCHEMA,
        core.handle_context_status,
        "A",
    ),
    (
        "aituber_onair_stream_start_tweet",
        core.STREAM_START_TWEET_SCHEMA,
        core.handle_stream_start_tweet,
        "A",
    ),
    ("aituber_onair_smoke", core.SMOKE_SCHEMA, core.handle_smoke, "A"),
    (
        "aituber_onair_youtube_ready",
        core.YOUTUBE_READY_SCHEMA,
        core.handle_youtube_ready,
        "A",
    ),
    (
        "aituber_onair_youtube_comments_status",
        core.YOUTUBE_COMMENTS_STATUS_SCHEMA,
        core.handle_youtube_comments_status,
        "A",
    ),
    (
        "aituber_onair_start_youtube_comments",
        core.START_YOUTUBE_COMMENTS_SCHEMA,
        core.handle_start_youtube_comments,
        "A",
    ),
    (
        "aituber_onair_stop_youtube_comments",
        core.STOP_YOUTUBE_COMMENTS_SCHEMA,
        core.handle_stop_youtube_comments,
        "A",
    ),
    (
        "aituber_onair_loops_status",
        core.LOOPS_STATUS_SCHEMA,
        core.handle_loops_status,
        "A",
    ),
    (
        "aituber_onair_start_autonomous_talk",
        core.START_AUTONOMOUS_TALK_SCHEMA,
        core.handle_start_autonomous_talk,
        "A",
    ),
    (
        "aituber_onair_start_comment_reactions",
        core.START_COMMENT_REACTIONS_SCHEMA,
        core.handle_start_comment_reactions,
        "A",
    ),
    (
        "aituber_onair_enqueue_comment",
        core.ENQUEUE_COMMENT_SCHEMA,
        core.handle_enqueue_comment,
        "A",
    ),
    (
        "aituber_onair_stop_loops",
        core.STOP_LOOPS_SCHEMA,
        core.handle_stop_loops,
        "A",
    ),
)


def register(ctx) -> None:
    """Register AITuber OnAir tools, slash command, and CLI command."""
    global _CTX
    _CTX = ctx
    core.bind_llm_factory(_ctx_llm)

    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="aituber-onair",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            description=schema.get("description", ""),
            emoji=emoji,
        )

    ctx.register_command(
        "aituber",
        handler=core.handle_slash,
        description="Run AITuber OnAir avatar streaming and Hakua Codex character chat.",
        args_hint="[status|configure|prepare|start|stop|say|smoke|youtube-ready|start-comments|start-autonomous|start-reactions|comment]",
    )
    ctx.register_cli_command(
        name="aituber-onair",
        help="AITuber OnAir avatar and Hakua Codex bridge",
        setup_fn=register_cli,
        handler_fn=aituber_onair_command,
        description=(
            "Configure Hakua, prepare AITuber OnAir's Codex SDK example, "
            "start FBX or VRoid/VRM apps, run Codex-auth character prompts, and route Hakua voice."
        ),
    )
