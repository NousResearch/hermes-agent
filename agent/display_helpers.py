"""AIAgent display/emit helpers — extracted from run_agent.py.

Handles safe printing, verbose mode suppression, status/warning emission,
and spinner gate logic. Each function takes ``agent`` (AIAgent instance)
as its first argument.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


def safe_print(agent, *args, **kwargs) -> None:
    """Print that silently handles broken pipes / closed stdout.

    In headless environments (systemd, Docker, nohup) stdout may become
    unavailable mid-session.  A raw ``print()`` raises ``OSError`` which
    can crash cron jobs and lose completed work.
    """
    try:
        fn = agent._print_fn or print
        fn(*args, **kwargs)
    except (OSError, ValueError):
        pass


def vprint(agent, *args, force: bool = False, **kwargs) -> None:
    """Verbose print — suppressed when actively streaming tokens.

    Pass ``force=True`` for error/warning messages that should always be
    shown even during streaming playback (TTS or display).
    """
    if getattr(agent, "suppress_status_output", False):
        return
    if not force and getattr(agent, "_mute_post_response", False):
        return
    if not force and _has_stream_consumers(agent) and not agent._executing_tools:
        return
    safe_print(agent, *args, **kwargs)


def _has_stream_consumers(agent) -> bool:
    """Return True when any stream consumer is registered."""
    return bool(
        agent.stream_delta_callback
        or agent.reasoning_callback
        or agent.tool_gen_callback
        or agent.interim_assistant_callback
    )


def should_start_quiet_spinner(agent) -> bool:
    """Return True when quiet-mode spinner output has a safe sink."""
    if agent._print_fn is not None:
        return True
    stream = getattr(sys, "stdout", None)
    if stream is None:
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def should_emit_quiet_tool_messages(agent) -> bool:
    """Return True when quiet-mode tool summaries should print directly."""
    return (
        agent.quiet_mode
        and not agent.tool_progress_callback
        and getattr(agent, "platform", "") == "cli"
    )


def emit_status(agent, message: str) -> None:
    """Emit a lifecycle status message to both CLI and gateway channels.

    CLI users see the message via ``vprint(force=True)`` so it is always
    visible regardless of verbose/quiet mode.  Gateway consumers receive
    it through ``status_callback(\"lifecycle\", ...)``.
    """
    try:
        vprint(agent, f"{agent.log_prefix}{message}", force=True)
    except Exception:
        pass
    if agent.status_callback:
        try:
            agent.status_callback("lifecycle", message)
        except Exception:
            logger.debug("status_callback error in emit_status", exc_info=True)


def emit_warning(agent, message: str) -> None:
    """Emit a user-visible warning through the same status plumbing.

    Unlike debug logs, these warnings are meant for degraded side paths
    such as auxiliary compression or memory flushes where the main turn can
    continue but the user needs to know something important failed.
    """
    try:
        vprint(agent, f"{agent.log_prefix}{message}", force=True)
    except Exception:
        pass
    if agent.status_callback:
        try:
            agent.status_callback("warn", message)
        except Exception:
            logger.debug("status_callback error in emit_warning", exc_info=True)


def emit_auxiliary_failure(agent, task: str, exc: BaseException) -> None:
    """Surface a compact warning for failed auxiliary work."""
    try:
        detail = _summarize_api_error(exc)
    except Exception:
        detail = str(exc)
    detail = (detail or exc.__class__.__name__).strip()
    if len(detail) > 220:
        detail = detail[:217].rstrip() + "..."
    emit_warning(agent, f"\u26a0 Auxiliary {task} failed: {detail}")


def _summarize_api_error(error: Exception) -> str:
    """Extract a concise, user-readable summary from an API error."""
    msg = str(error) or ""
    # Strip common noisy prefixes
    for prefix in [
        "Error code: ", "OpenAI error: ",
        "Anthropic API error: ", "API error: ",
    ]:
        if msg.startswith(prefix):
            msg = msg[len(prefix):]
    # Strip long JSON bodies
    if len(msg) > 500:
        msg = msg[:497] + "..."
    return msg
