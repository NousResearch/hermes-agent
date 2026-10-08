"""User-visible hook notices: a hook result's ``notice`` (Hermes) or ``systemMessage`` (Claude-Code
dialect) is shown to the user and never enters model context.

Port of MiniMax-AI/minimax-code#376. Until now a plugin or shell hook could only speak to the model
(``context``, a block message, a transformed result); a formatter that wants to say "reformatted
3 files" had to either stay silent or spend the model's context on it. The text rides the existing
driver-agnostic ``AgentNotice`` channel (CLI print, TUI/Desktop ``notification.show``, messaging
gateway one-shot line), so it is excluded from canonical history, compaction and export by
construction: it is never appended to ``messages``.

Hooks fire from places that hold no agent (``model_tools.handle_function_call``), only the payload's
``session_id``; live agents register here at construction and the notice is routed to the one whose
current ``session_id`` matches (compaction can rotate the id mid-turn, so the match is evaluated at
delivery, not at registration).
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NOTICE_KEYS = ("notice", "systemMessage")
MAX_NOTICE_CHARS = 2000

_AGENTS: "weakref.WeakSet[Any]" = weakref.WeakSet()
_LOCK = threading.Lock()


def register_agent(agent: Any) -> None:
    """Make *agent* a candidate sink for notices whose payload names its ``session_id``."""
    with _LOCK:
        _AGENTS.add(agent)


def extract_hook_notice(result: Any) -> Optional[str]:
    """The sanitized user-visible notice carried by one hook result, or ``None``."""
    if not isinstance(result, dict):
        return None
    for key in NOTICE_KEYS:
        text = result.get(key)
        if isinstance(text, str) and text.strip():
            from tools.ansi_strip import sanitize_display_text

            return sanitize_display_text(text.strip())[:MAX_NOTICE_CHARS]
    return None


def _agent_for_session(session_id: str) -> Optional[Any]:
    with _LOCK:
        candidates = [a for a in list(_AGENTS) if getattr(a, "session_id", None) == session_id]
    return candidates[0] if candidates else None


def deliver_hook_notices(hook_name: str, kwargs: Dict[str, Any], results: List[Any]) -> None:
    """Show every notice in *results* to the user of the session the hook fired for. Never raises."""
    notices = [n for n in (extract_hook_notice(r) for r in results) if n]
    if not notices:
        return
    session_id = kwargs.get("session_id")
    agent = _agent_for_session(session_id) if isinstance(session_id, str) and session_id else None
    if agent is None:
        logger.info("hook %s returned a notice but no live agent owns session %r; dropped: %s",
                    hook_name, session_id, notices[0][:120])
        return
    for text in notices:
        try:
            agent._emit_hook_notice(hook_name, text)
        except Exception:
            logger.debug("hook notice delivery failed (hook=%s)", hook_name, exc_info=True)
