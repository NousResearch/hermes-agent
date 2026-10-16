"""Temporary-chat policy for an AIAgent.

A temporary chat neither reads nor writes long-term state: no built-in or external memory, no
background memory/skill review, no recall of other conversations, no auto-title. Its session row
is stamped temporary (hidden, unlisted, unsearchable) so the owning surface can delete it when the
user leaves the chat. Storage side: ``hermes_state_temporary``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Recall and memory tools read or write state that outlives the chat.
TEMPORARY_DISABLED_TOOLSETS = ("memory", "session_search")


def temporary_agent_options(
    agent: Any, temporary: bool, skip_memory: bool, skip_background_review: bool,
    disabled_toolsets: Optional[list[str]],
) -> tuple[bool, bool, Optional[list[str]]]:
    """(skip_memory, skip_background_review, disabled_toolsets) with the temporary policy applied;
    records the flag on *agent* as ``_temporary``."""
    agent._temporary = bool(temporary)
    if not temporary:
        return skip_memory, skip_background_review, disabled_toolsets
    merged = list(disabled_toolsets or [])
    merged += [name for name in TEMPORARY_DISABLED_TOOLSETS if name not in merged]
    return True, True, merged


def turn_log_preview(agent: Any, user_message: Any, summarize: Optional[Callable[[Any], str]] = None) -> str:
    """The message preview for the ``conversation turn:`` log line: a temporary chat's words never
    reach agent.log."""
    if getattr(agent, "_temporary", False):
        return "<temporary chat>"
    if summarize is None:
        from agent.codex_responses_adapter import _summarize_user_message_for_log as summarize
    text = summarize(user_message)
    return (text[:80] + ("..." if len(text) > 80 else "")).replace("\n", " ")


def mark_agent_session_temporary(agent: Any) -> None:
    """Stamp the agent's row temporary + hidden. The marker is never carried in the create-time
    model_config: the create upsert fills a NULL model_config on an EXISTING row, which would silently
    turn a saved conversation temporary. (Compression children inherit it in SessionDB.)"""
    session_db = getattr(agent, "_session_db", None)
    target = getattr(agent, "session_id", None)
    if not getattr(agent, "_temporary", False) or session_db is None or not target:
        return
    try:
        # A row with history that is not already temporary is a saved conversation (a resume or
        # session switch reached this agent): stamping it would delete it when the chat closes.
        if not session_db.is_session_temporary(target) and (session_db.get_session(target) or {}).get("message_count"):
            logger.warning("refusing to mark saved session %s temporary", target)
            return
        session_db.mark_session_temporary(target)
    except Exception:
        logger.warning("failed to mark temporary session %s", target, exc_info=True)
