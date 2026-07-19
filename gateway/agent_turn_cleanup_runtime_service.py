"""Turn-scoped cleanup for gateway foreground handler finally blocks.

Promoted from ``GatewayRunner._handle_message_with_agent`` finally: drop
unconsumed turn-sidecar notes and restore session env tokens.
"""

from __future__ import annotations

from typing import Any


def cleanup_gateway_agent_turn(
    *,
    runner: Any,
    session_key: str | None,
    session_env_tokens: Any,
    logger: Any,
) -> None:
    """Run handler finally cleanup (idempotent)."""
    if session_key:
        leftover = runner._consume_pending_turn_sidecar_notes(session_key)
        if leftover:
            logger.debug(
                "Cleared %d unconsumed turn sidecar note(s) for session %s",
                len(leftover),
                session_key,
            )
    runner._clear_session_env(session_env_tokens)
