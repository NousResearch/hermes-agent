"""Turn-loop bridge to the native adviser.

The conversation loop and the stop gates only need two things: a session handle
(lazily created on the first turn, so an agent whose config has ``adviser.enabled:
false`` pays nothing) and one terminal-boundary capture taken before the stop
gates decide whether the turn ends. Keeping both here leaves the loop phases at
their audited complexity: the call site is one function call, not a branch.

Session kind decides which ``WATCHDOG.yml`` scopes may build: the top-level
session hosts ``main``/``all`` entries, a delegated subagent hosts
``subagent``/``all`` entries, and cron runs or detached forks host none.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def ensure_adviser(agent: Any) -> Optional[Any]:
    """Return (creating once) the agent's adviser session, or None when construction failed."""
    existing = getattr(agent, "adviser", None)
    if existing is not None:
        return existing
    with _LOCK:
        existing = getattr(agent, "adviser", None)
        if existing is not None:
            return existing
        try:
            from agent.adviser_runtime import AdviserSession, adviser_session_kind

            kind = adviser_session_kind(agent)
            if kind is None:  # cron runs and detached forks never host a reviewer
                return None
            session = AdviserSession(agent, kind=kind)
        except Exception:  # health: allow BLE001 -- optional feature: a broken adviser must never block a turn
            logger.debug("adviser session init failed", exc_info=True)
            return None
        agent.adviser = session
        return session


def step_review(agent: Any, messages: list[Any]) -> None:
    """Capture a completed tool-round boundary; observational, never blocks the turn."""
    adviser = getattr(agent, "adviser", None)
    if adviser is None:
        return
    try:
        adviser.on_step(messages, terminal=False)
    except Exception:  # health: allow BLE001 -- reviewer capture is observational; the turn must continue
        logger.debug("adviser step capture failed", exc_info=True)


def terminal_review(agent: Any, messages: list[Any], final_response: Any) -> None:
    """Capture the terminal boundary with the candidate answer, before the stop gates.

    The caputed row is a copy: the answer is not appended until a gate accepts it,
    but the reviewer must be able to judge the answer that is about to be shown.
    """
    if not isinstance(final_response, str) or not final_response.strip():
        return
    adviser = ensure_adviser(agent)
    if adviser is None:
        return
    try:
        adviser.on_step([*messages, {"role": "assistant", "content": final_response}], terminal=True)
    except Exception:  # health: allow BLE001 -- reviewer capture is observational; the turn must continue
        logger.debug("adviser terminal capture failed", exc_info=True)


__all__ = ["ensure_adviser", "step_review", "terminal_review"]
