"""Chronoception — a sense of elapsed time between turns.

A stateless language model has no clock. Given no temporal cue in context it
asserts "0 seconds elapsed" with full confidence and will act on stale state as
if no time has passed. This plugin injects the real elapsed time each turn (a
compact clock) and a one-shot notice when the agent resumes after a long idle
gap, so the model can reason about staleness.

Self-contained: the elapsed-time signal is derived from wall-clock only, so the
plugin has no external dependencies. It contributes ephemeral context via the
``pre_llm_call`` hook (appended to the current user message at API-call time,
never persisted), fenced and attributed. Fail-closed: without
``chronoception.enabled: true`` in config it is inert, and any internal error
returns ``None`` so a turn is never broken.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _pre_llm_call(**kwargs: Any) -> Optional[dict]:
    """Turn-prologue hook: contribute an ephemeral timing block, or nothing."""
    try:
        from plugins.chronoception.settings import get_settings

        settings = get_settings()
        if not settings["enabled"]:
            return None

        from plugins.chronoception.sense import build

        # A falsy session id must not merge distinct conversations into one
        # elapsed-time clock; fall back to a per-thread key.
        session_id = str(kwargs.get("session_id") or "")
        if not session_id:
            session_id = f"no-session-{threading.get_ident()}"

        text = build(session_id, settings)
        return {"context": text} if text else None
    except Exception:
        logger.debug("chronoception failed; staying silent", exc_info=True)
        return None


def register(ctx) -> None:
    """Called once by the plugin loader."""
    ctx.register_hook("pre_llm_call", _pre_llm_call)
