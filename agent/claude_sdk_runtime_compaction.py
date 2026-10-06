"""CLI compaction status edges for the claude-agent-sdk runtime.

The start edge (PreCompact hook), the completion edge (compact_boundary) and
the end-of-turn fallback that closes a status whose boundary never streamed.
What used to be closures nested in ``run_claude_agent_sdk_turn`` take the
agent explicitly; the session receives them bound with ``functools.partial``.
Extracted from ``claude_sdk_runtime.py``.
"""

from __future__ import annotations

import logging

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.claude_sdk_runtime")


def _on_compaction(agent, trigger: str) -> None:
    """CLI is compacting — surface it with the SHARED status wording.

    Reuses conversation_compression's constants rather than inventing a
    second vocabulary: the gateway's noise filter is built from those
    same templates (#69550), so a re-inlined string would be silently
    dropped on chat surfaces.

    A manual /compact is the user's own action and already has its own
    feedback, so only the automatic case is announced -- that is the one
    that stalls a turn with no explanation.
    """
    if str(trigger).strip().lower() == "manual":
        return
    try:
        from agent.conversation_compression import (
            COMPACTION_STATUS,
            COMPACTION_STATUS_KEY,
        )

        agent._sdk_compaction_pending = True
        emit = getattr(agent, "_emit_status_kind", None)
        if callable(emit):
            emit(COMPACTION_STATUS_KEY, COMPACTION_STATUS, origin="claude_sdk_compaction")
            logger.info("CLI compaction started (trigger=%s); status emitted", trigger)
        else:
            logger.info(
                "CLI compaction started (trigger=%s); no _emit_status_kind, "
                "status not emitted",
                trigger,
            )
    except Exception:
        logger.debug("failed to emit CLI compaction status", exc_info=True)


def _on_compact_boundary(agent, trigger: str) -> None:
    """Compaction finished — close the status the PreCompact hook opened.

    This is the real terminal edge, mid-turn, ~1 minute before the turn
    ends. The end-of-turn emit (``_emit_dangling_compaction_done``) is now
    only a fallback for a CLI that stops streaming compact_boundary;
    whichever fires first clears the pending flag, so the notice is emitted
    exactly once.

    Guarded on the pending flag rather than emitted unconditionally: a
    manual /compact is never announced on the start side, and announcing
    only its completion would be a notice for an event the user was
    never told had begun.
    """
    if not getattr(agent, "_sdk_compaction_pending", False):
        return
    agent._sdk_compaction_pending = False
    try:
        from agent.conversation_compression import _emit_compaction_done

        logger.info("CLI compaction finished (trigger=%s)", trigger)
        _emit_compaction_done(agent)
    except Exception:
        logger.debug("failed to emit CLI compaction completion", exc_info=True)


def _emit_dangling_compaction_done(agent) -> None:
    """End-of-turn FALLBACK for a compaction whose compact_boundary never streamed.

    ``_on_compact_boundary`` is the real terminal edge and normally clears
    the flag mid-turn; reaching here with it still set means the CLI started
    a compaction and never streamed its compact_boundary (older/newer CLI,
    or a turn that died mid-compaction). A completed turn still proves the
    compaction ended, so this closes the dangling status rather than leaving
    "🗜️ Compacting..." as the user's last word from the turn.

    Do NOT promote this back to the primary path: end-of-turn is exactly
    where end-of-turn progress cleanup deletes the message, so the notice is
    emitted and destroyed in the same instant (see _handle_compact_boundary).
    """
    if getattr(agent, "_sdk_compaction_pending", False):
        agent._sdk_compaction_pending = False
        try:
            from agent.conversation_compression import _emit_compaction_done

            logger.info(
                "CLI compaction: no compact_boundary seen; emitting completion "
                "at turn end (fallback)"
            )
            _emit_compaction_done(agent)
        except Exception:
            logger.debug("failed to emit CLI compaction completion", exc_info=True)
