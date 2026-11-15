"""google_meet plugin — let the agent join a Meet call, transcribe it, follow up.

Headless Chromium (Playwright) joins the URL, enables live captions and scrapes them into a
transcript. Realtime mode adds agent speech (OpenAI Realtime + virtual audio device); remote
nodes run the bot on another machine. Explicit-by-design: only joins ``https://meet.google.com/``
URLs passed in — no calendar scanning, auto-dial or consent announcement.
"""

from __future__ import annotations

import logging
import platform

from plugins.google_meet import process_manager as pm
from plugins.google_meet.cli import meet_command as _meet_command, register_cli as _register_meet_cli
from plugins.google_meet.tools import (
    MEET_JOIN_SCHEMA, MEET_LEAVE_SCHEMA, MEET_SAY_SCHEMA, MEET_STATUS_SCHEMA, MEET_TRANSCRIPT_SCHEMA,
    check_meet_requirements, handle_meet_join, handle_meet_leave, handle_meet_say, handle_meet_status,
    handle_meet_transcript)

logger = logging.getLogger(__name__)


_TOOLS = (
    ("meet_join",       MEET_JOIN_SCHEMA,       handle_meet_join,       "📞"),
    ("meet_status",     MEET_STATUS_SCHEMA,     handle_meet_status,     "🟢"),
    ("meet_transcript", MEET_TRANSCRIPT_SCHEMA, handle_meet_transcript, "📝"),
    ("meet_leave",      MEET_LEAVE_SCHEMA,      handle_meet_leave,      "👋"),
    ("meet_say",        MEET_SAY_SCHEMA,        handle_meet_say,        "🗣️"))


def _on_session_end(**kwargs) -> None:
    """Leave a still-running call so we don't orphan a headless Chromium (never raises)."""
    try:
        status = pm.status()
        if status.get("ok") and status.get("alive"):
            pm.stop(reason="session ended")
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("google_meet on_session_end cleanup failed: %s", e)


def register(ctx) -> None:
    """Register tools, CLI, and lifecycle hooks (called once by the plugin loader)."""
    # Windows: no tested audio-routing path and flaky guest-join Chromium — refuse rather than half-work.
    system = platform.system().lower()
    if system not in {"linux", "darwin"}:
        logger.info("google_meet plugin: platform=%s not supported (linux/macos only)", system)
        return
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(name=name, toolset="google_meet", schema=schema, handler=handler,
                          check_fn=check_meet_requirements, emoji=emoji)
    ctx.register_cli_command(
        name="meet", help="Google Meet bot (join, transcribe, follow up)",
        setup_fn=_register_meet_cli, handler_fn=_meet_command,
        description=("Let the hermes agent join a Google Meet call and scrape live "
                     "captions into a transcript. See: hermes meet setup"))
    ctx.register_hook("on_session_end", _on_session_end)
