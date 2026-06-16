"""multi-agent hook — per-agent routing and pseudo-conversational state.

Implements the #9514 single-daemon multi-agent architecture using the
IBM CICS TCB model: one Gateway process, many AgentContexts.

Events
------
agent:start
    Resolves the active agent via AgentRoutingTable, loads its
    AgentContext from disk, and writes a session hint file that
    a cooperating SOUL.md or skill can reference.
agent:end
    Snapshots the agent's pseudo-conversational state to disk.

Limitations (current hook API)
------------------------------
This hook CAN route and snapshot state, but CANNOT inject the
AgentContext into the LLM prompt directly — the hook API is
observer-only (read context dict, fire side effects).  For full
context injection, a future ``agent:pre_dispatch`` hook (or a
one-line gateway patch) would be needed.  See HOOK.md for the
design rationale and upgrade path.
"""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("hermes.hooks.multi-agent")


# ── safe access ──────────────────────────────────────────────────────


def _safe_get(d: dict, key: str, default: str = "") -> str:
    """Read a context field safely — survives key renames or missing fields."""
    try:
        return str(d.get(key, default))
    except Exception:
        return default


# ── lazy imports (hook loads inside gateway, sys.path has hermes root) ──

_AGENT_POOL = None
_ROUTING_TABLE = None
_CURRENT_AGENT_ID: str | None = None  # set on agent:start, used by agent:end


def _init_lazy():
    global _AGENT_POOL, _ROUTING_TABLE
    if _AGENT_POOL is not None:
        return
    from src.orchestration.agent_routing import AgentRoutingTable
    from src.orchestration.agent_pool import AgentContextPool

    _ROUTING_TABLE = AgentRoutingTable()
    _AGENT_POOL = AgentContextPool()


# ── agent routing (agent:start) ──────────────────────────────────────


async def _on_agent_start(context: dict) -> None:
    """Resolve which agent should handle this message, load its context."""
    global _CURRENT_AGENT_ID

    platform = _safe_get(context, "platform", "unknown")
    chat_id = _safe_get(context, "chat_id", "")
    thread_id = _safe_get(context, "thread_id", "")  # topic_id on Telegram

    if not chat_id:
        logger.debug("agent:start — no chat_id, skipping routing")
        return

    _init_lazy()
    agent_id = _ROUTING_TABLE.resolve(platform, chat_id, thread_id or None)
    _CURRENT_AGENT_ID = agent_id

    ctx = _AGENT_POOL.get_or_create(agent_id)

    # Write session hint — SOUL.md or skills can reference this file
    hint = {
        "agent_id": agent_id,
        "platform": platform,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "session_id": _safe_get(context, "session_id", ""),
        "memory": ctx.memory,
        "skills": ctx.skills,
        "state": ctx.state,
    }
    hint_path = Path.home() / ".hermes" / "agents" / agent_id / "current_session.json"
    hint_path.parent.mkdir(parents=True, exist_ok=True)
    hint_path.write_text(json.dumps(hint, indent=2, default=str))

    logger.info(
        "multi-agent: routed %s/%s/%s → %s (turn %s)",
        platform, chat_id, thread_id or "-", agent_id,
        ctx.state.get("turn_count", 0),
    )


# ── state snapshot (agent:end) ───────────────────────────────────────


async def _on_agent_end(context: dict) -> None:
    """Save the agent's state after a turn completes."""
    global _CURRENT_AGENT_ID

    if _CURRENT_AGENT_ID is None:
        return

    _init_lazy()
    ctx = _AGENT_POOL.get_or_create(_CURRENT_AGENT_ID)

    # Update turn count and snapshot
    ctx.state["turn_count"] = ctx.state.get("turn_count", 0) + 1
    _AGENT_POOL.snapshot(_CURRENT_AGENT_ID)

    logger.debug(
        "multi-agent: snapshotted %s (turns: %s)",
        _CURRENT_AGENT_ID, ctx.state["turn_count"],
    )


# ── hook entry point ─────────────────────────────────────────────────


async def handle(event_type: str, context: dict) -> None:
    """Gateway hooks entry point.  All exceptions MUST be caught."""
    try:
        if event_type == "agent:start":
            await _on_agent_start(context)
        elif event_type == "agent:end":
            await _on_agent_end(context)
    except Exception:
        logger.debug("multi-agent hook error (non-fatal)", exc_info=True)
