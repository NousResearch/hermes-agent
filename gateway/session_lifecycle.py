"""Gateway session lifecycle management — extracted from gateway/run.py.

Handles agent cache expiry, session expiry watcher, handoff processing,
and scheduled resume of pending sessions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def schedule_resume_pending_sessions(
    session_db: Any,
    session_keys: Set[str],
    platform: str,
) -> int:
    """Resume any sessions that were interrupted by a restart.

    Returns count of resumed sessions.
    """
    resumed = 0
    try:
        if not session_db:
            return 0
        active = session_db.get_active_sessions(source=platform)
        for sess in active:
            sid = sess.get("session_id", "")
            if sid and sid not in session_keys:
                session_keys.add(sid)
                resumed += 1
    except Exception as e:
        logger.debug("Failed to resume pending sessions: %s", e)
    return resumed


async def session_expiry_watcher(
    agent_cache: Dict[str, Any],
    expiry_ts: Dict[str, float],
    stop_event: asyncio.Event,
    interval: int = 300,
    idle_ttl: float = 3600.0,
) -> None:
    """Background task that evicts idle agents from the cache.

    Runs every ``interval`` seconds.  Evicts agents whose last activity
    is older than ``idle_ttl`` seconds.
    """
    while not stop_event.is_set():
        try:
            now = time.monotonic()
            expired = [
                key for key, ts in expiry_ts.items()
                if now - ts > idle_ttl
            ]
            for key in expired:
                agent = agent_cache.pop(key, None)
                expiry_ts.pop(key, None)
                if agent:
                    try:
                        agent.close()
                    except Exception:
                        pass
            if expired:
                logger.info("Evicted %d idle agents from cache", len(expired))
        except Exception as e:
            logger.debug("Session expiry check failed: %s", e)
        await asyncio.sleep(interval)


async def process_handoff(
    handoff_row: Dict[str, Any],
    session_manager: Any,
    agent_factory: Any,
) -> None:
    """Process a single handoff row: reconstruct agent state and continue."""
    try:
        session_id = handoff_row.get("session_id", "")
        if not session_id:
            return
        target_session = handoff_row.get("target_session", session_id)
        handoff_data = handoff_row.get("data", {})
        message = handoff_data.get("message", "")
        history = handoff_data.get("history", [])

        # Reconstruct agent
        agent = agent_factory(session_id=target_session)
        if history:
            agent.run_conversation(message, conversation_history=history)
        else:
            agent.chat(message)
    except Exception as e:
        logger.error("Handoff processing failed: %s", e)
