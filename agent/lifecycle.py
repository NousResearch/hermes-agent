"""AIAgent lifecycle management — extracted from run_agent.py.

Handles interrupt/steer/close/release and related lifecycle operations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def is_interrupted(agent) -> bool:
    """Check if an interrupt has been requested."""
    return agent._interrupt_requested


def interrupt(agent, message: str = None) -> None:
    """Request graceful interruption of the current tool-calling loop."""
    agent._interrupt_requested = True
    agent._interrupt_message = message
    from tools.interrupt import set_interrupt
    set_interrupt(True)


def clear_interrupt(agent) -> None:
    """Clear a pending interrupt request."""
    agent._interrupt_requested = False
    agent._interrupt_message = None
    from tools.interrupt import set_interrupt
    set_interrupt(False)


def steer(agent, text: str) -> bool:
    """Inject a steer message for the next LLM response."""
    try:
        if agent._pending_steer is not None:
            agent._pending_steer += "\n" + text
        else:
            agent._pending_steer = text
        return True
    except Exception:
        return False


def drain_pending_steer(agent) -> Optional[str]:
    """Drain any pending steer message and return it."""
    text = agent._pending_steer
    agent._pending_steer = None
    return text


def release_clients(agent) -> None:
    """Release all HTTP clients (OpenAI, Anthropic, etc.) for cleanup."""
    client = getattr(agent, "_openai_client", None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass
        agent._openai_client = None

    anthropic = getattr(agent, "_anthropic_client", None)
    if anthropic is not None:
        try:
            anthropic.close()
        except Exception:
            pass
        agent._anthropic_client = None

    for oc in list(getattr(agent, "_openai_clients", set())):
        try:
            oc.close()
        except Exception:
            pass
    agent._openai_clients.clear()


def close(agent) -> None:
    """Full agent shutdown: persist session, release clients, clean up."""
    # Persist session if available
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if session_db and session_id:
        try:
            messages = getattr(agent, "_session_messages", None)
            if messages:
                from agent.session_state import save_session_log
                save_session_log(agent, messages)
        except Exception as e:
            logger.debug("Session save on close failed: %s", e)

    # Shutdown memory provider
    try:
        if hasattr(agent, "shutdown_memory_provider"):
            agent.shutdown_memory_provider()
    except Exception:
        pass

    # Release clients
    release_clients(agent)

    # Close session DB
    if session_db:
        try:
            session_db.close()
        except Exception:
            pass


def get_rate_limit_state(agent) -> dict:
    """Return current rate-limit tracking state."""
    return {
        "rate_limited_until": getattr(agent, "_rate_limited_until", 0),
        "retry_after": getattr(agent, "_retry_after", 0),
    }


def get_activity_summary(agent) -> dict:
    """Return a summary of agent activity for the current session."""
    return {
        "api_calls": getattr(agent, "session_api_calls", 0),
        "total_tokens": getattr(agent, "session_total_tokens", 0),
        "estimated_cost": getattr(agent, "session_estimated_cost_usd", 0.0),
        "interrupted": getattr(agent, "_interrupt_requested", False),
    }


def touch_activity(agent, desc: str) -> None:
    """Record a brief activity description for status display."""
    agent._last_activity = desc
    agent._last_activity_at = __import__('time').time()


def current_main_runtime(agent) -> Dict[str, str]:
    """Return the live main runtime for session-scoped auxiliary routing."""
    return {
        "model": getattr(agent, "model", "") or "",
        "provider": getattr(agent, "provider", "") or "",
        "base_url": getattr(agent, "base_url", "") or "",
        "api_key": getattr(agent, "api_key", "") or "",
        "api_mode": getattr(agent, "api_mode", "") or "",
    }
