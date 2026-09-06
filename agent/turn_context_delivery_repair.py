"""Fold pending async-delegation completions into the next turn's context for stateless API clients.

Gateway platforms that manage persistent sessions (TUI, Desktop, CLI) ingest delegation
completions via events and live-merge them into ongoing conversation state. Stateless API
endpoints (OpenAI /v1/chat/completions without X-Hermes-Session-Id header) derive the
session from a request fingerprint but build model context from the client-provided message
array each turn. A background delegation persists its completion as a DB row; the next turn
for the same session must fold that completion into the message list so the model sees it.

This repair belt runs once per turn immediately after conversation_history is copied. It:
1. Checks whether pending delivery rows exist for agent.session_id.
2. Reads active display_kind='async_delegation_complete' rows from the DB.
3. Inserts each completion as a synthetic user message in temporal order before the live
   user turn (so the model sees "task completed [summary]" before it reads the user's prompt).

Design constraints:
- Zero cost when no delivery rows exist (EXIT-BEFORE-DB call).
- Idempotent: rows are not removed by the repair (they remain until compaction); callers
  with persistent conversation state (TUI, CLI) feed the same rows via conversation_history
  and would see duplicates if they called this.
- DB session binding happens at line 875 in build_turn_context (_ensure_session_row);
  repair runs at line 832 BEFORE that, when agent._session_db is already bound by the platform.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _fold_pending_delegation_completions(
    agent: Any, messages: List[Dict[str, Any]], session_id: Optional[str]
) -> None:
    """Inject pending async-delegation completions into stateless-client message context.

    Args:
        agent: Agent instance (must have ._session_db and .quiet_mode)
        messages: Mutable model context list; completions are inserted BEFORE the final user msg.
        session_id: Session key; None/empty skips repair (no session = no persistent completions).
    """
    if not session_id or not agent._session_db:
        return
    if not messages or messages[-1].get("role") != "user":
        # Unusual: repair expects the tail to be the fresh user message. Warn + skip.
        logger.debug(
            "pending delegation repair skipped: final message is not role=user (session=%s)",
            session_id,
        )
        return

    # Query DB for delivery rows. The contract: active=1 AND display_kind='async_delegation_complete'.
    # active=1: row is part of working history (not archived by compaction).
    # display_kind field carries the purpose; we filter to the exact completion marker.
    try:
        rows = agent._session_db.get_messages(
            session_id=session_id, include_inactive=False, include_compacted=False
        )
    except Exception:
        logger.debug("pending delegation repair DB read failed", exc_info=True)
        return

    completions = [
        r
        for r in rows
        if r.get("display_kind") == "async_delegation_complete" and r.get("role") == "user"
    ]
    if not completions:
        return

    # Insert each completion into messages BEFORE the final user turn. Order matters: insert older first.
    completions.sort(key=lambda r: r.get("id", 0))
    insert_idx = len(messages) - 1  # before the tail user message

    for row in completions:
        # Reconstruct a conversation-dict from the DB row. Content is already user-formatted.
        msg = {"role": "user", "content": row["content"]}
        messages.insert(insert_idx, msg)
        insert_idx += 1

    if not agent.quiet_mode:
        logger.info(
            "folded %d pending async-delegation completion(s) into turn context (session=%s)",
            len(completions),
            session_id,
        )
