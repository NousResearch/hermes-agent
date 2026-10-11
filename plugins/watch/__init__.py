"""watch plugin — record a performance, then ask a model what it thinks.

"Watch me play this / perform this." The user records explicitly, stops
explicitly, and hands the take to a video-capable model. Three tracks travel
together: the screen, the audio, and a window/app timeline the desktop app can
observe while recording — the last one being the thing a native screen recorder
cannot give you, and the reason this is worth doing in-app.

Footprint: this is a CLI command plus a skill, NOT a set of model tools. The
user drives record on/off, so the agent does not need schema surface for it —
it runs ``hermes watch …`` like any other shell command, and escalation to a
model reuses the built-in ``video_analyze`` tool rather than adding a second
path to the same provider ladder.

Capture requires ffmpeg on PATH. Registration is unconditional across
linux/macos/windows because each host's capture backend reports its own
unsupported cases at start time (Wayland being the notable refusal) with a
message that says what to do instead — more useful than a plugin that silently
does not exist.
"""

from __future__ import annotations

import logging

from plugins.watch.cli import register_cli as _register_watch_cli
from plugins.watch.cli import watch_command as _watch_command

logger = logging.getLogger(__name__)


def _on_session_end(**_kwargs) -> None:
    """Never leave capture running after a session closes.

    Two things to shut down. An orphaned recorder does not just waste CPU: an
    mp4 whose moov atom was never written is unplayable, so abandoning it
    destroys the take. An orphaned live loop keeps grabbing frames and calling
    a model for a session nobody is reading.

    Swallows everything; session teardown must not fail here.
    """
    try:
        from plugins.watch import slash

        slash.shutdown()
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("watch: live shutdown failed: %s", exc)

    try:
        from plugins.watch import recorder as rec

        if rec.status().get("recording"):
            result = rec.stop()
            logger.info(
                "watch: finalized recording at session end (%s)",
                result.get("video_path", "unknown path"),
            )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("watch: on_session_end cleanup failed: %s", exc)


def register(ctx) -> None:
    """Register the CLI command, the slash command, and the session-end net."""
    ctx.register_cli_command(
        name="watch",
        help="Record the screen and ask a model about the take",
        setup_fn=_register_watch_cli,
        handler_fn=_watch_command,
        description=(
            "Record a performance (screen + audio + window timeline) and hand "
            "it to a video-capable model. See: hermes watch start"
        ),
    )

    # The GUI surface. Registering here puts /watch in the desktop composer
    # palette and the TUI at once — both discover plugin commands through the
    # same registry, and the desktop surfaces non-builtins as extensions — so
    # no core file needs to know this plugin exists.
    from plugins.watch.slash import handle as _handle_slash

    ctx.register_command(
        "watch",
        handler=_handle_slash,
        description="Watch the screen and comment on what's happening.",
        args_hint="live|stop|status|record|takes|replay",
    )

    ctx.register_hook("on_session_end", _on_session_end)
