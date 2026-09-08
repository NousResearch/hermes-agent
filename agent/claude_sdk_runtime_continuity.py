"""Session continuity for the claude-agent-sdk runtime: the persisted SDK resume
id and the digest that primes a fresh SDK session from the Hermes transcript.
Extracted from ``claude_sdk_runtime.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.claude_sdk_runtime_state import _SdkTurnState

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.claude_sdk_runtime")


def _persisted_sdk_session_id(agent) -> Optional[str]:
    """The SDK session id stored on the Hermes session row (or None)."""
    if getattr(agent, "_persist_disabled", False):
        return None
    if not (getattr(agent, "_session_db", None) and getattr(agent, "session_id", None)):
        return None
    try:
        row = agent._session_db.get_session(agent.session_id) or {}
        return row.get("claude_sdk_session_id") or None
    except Exception:
        logger.debug("resume-id read failed", exc_info=True)
        return None


def _store_sdk_session_id(agent, value: Optional[str]) -> None:
    """Persist (or clear, with None) the SDK session id on the session row."""
    if getattr(agent, "_persist_disabled", False):
        # A review/curator fork shares the parent's session_id — it must
        # never write its own resume id onto the parent's row.
        return
    if not (getattr(agent, "_session_db", None) and getattr(agent, "session_id", None)):
        return
    try:
        agent._session_db.update_claude_sdk_session_id(agent.session_id, value)
    except Exception:
        logger.debug("resume-id write failed", exc_info=True)


_CONTINUITY_DIGEST_MAX_CHARS = 4000


def _render_continuity_digest(prior_messages: List[Dict[str, Any]]) -> str:
    """Bounded text preamble for a FRESH SDK session that has prior Hermes
    history (resume impossible: no stored id, or the stored one went stale).
    Reuses _digest_history's compaction, then flattens to capped text."""
    # Projected background results are the agent's OWN answers, already
    # delivered outbound; re-presenting them here is the double-presentation
    # pathology the background lane exists to kill. Filter before the
    # compaction pass — _digest_history may rebuild dicts and drop the mark.
    prior_messages = [
        m for m in (prior_messages or [])
        if not (
            isinstance(m, dict)
            and m.get("display_kind") == "sdk_background_result"
        )
    ]
    try:
        from agent.background_review import _digest_history

        msgs = _digest_history(list(prior_messages or []), tail=8)
    except Exception:  # pragma: no cover - compaction is best-effort
        msgs = list(prior_messages or [])[-8:]
    lines: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not content:
            continue
        text = str(content).replace("\n", " ").strip()
        if text:
            lines.append(f"{role.upper()}: {text[:400]}")
    if not lines:
        return ""
    body = "\n".join(lines)
    if len(body) > _CONTINUITY_DIGEST_MAX_CHARS:
        body = body[-_CONTINUITY_DIGEST_MAX_CHARS:]
    return (
        "[Continuity digest — the runtime restarted and the live model "
        "context was lost; recent turns from the stored transcript, oldest "
        "first:]\n" + body + "\n[End digest. The user's new message follows.]\n\n"
    )


def _persist_turn(agent, state: _SdkTurnState) -> None:
    """Flush the projected rows FIRST, then persist the SDK resume id."""
    turn = state.turn
    messages = state.messages
    if turn.projected_messages:
        messages.extend(turn.projected_messages)
        # Early-return path bypasses conversation_loop's per-step persistence;
        # flush the new projected rows ourselves (idempotent via the intrinsic
        # _DB_PERSISTED_MARKER — the user turn was flushed at turn start).
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug(
                    "claude-sdk projected-message flush failed", exc_info=True
                )

    if not getattr(turn, "should_retire", False) and state.failover_reason is None:
        # Persist the SDK session id for restart/eviction/interrupt resume.
        # AFTER the flush on purpose: the flush's _ensure_db_session retry is
        # what (re)creates the session row when turn-start persistence hit a
        # transient lock — storing first would silently discard the id.
        thread_id = getattr(turn, "thread_id", None)
        if thread_id:
            _store_sdk_session_id(agent, thread_id)
