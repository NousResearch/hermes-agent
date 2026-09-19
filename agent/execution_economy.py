"""Opt-in, non-interrupting execution-economy checkpoints (issue #110129).

Gated by ``agent.execution_economy_checkpoints`` (config ``agent.execution_economy_checkpoints``,
default OFF): a turn that keeps issuing single-call tool rounds, or that repeats a
normalized-equivalent call, gets ONE bounded advisory line appended to its newest *unpersisted*
tool result — in band, on the result the model is about to read, with no extra message row.

Advisory only, by construction. Nothing here denies a tool, changes permission / iteration /
billing state, ends a turn, or rewrites history: the append targets the row the incremental flush
is about to persist and bails out when that row is already stamped ``_DB_PERSISTED_MARKER``.
Cancellation results are never annotated, the tracker is per-turn (``agent.turn_context``
installs a fresh state), notices are capped per turn and spaced by a cooldown, and logging
carries reason / count / tool names only — never arguments or result content.

The detection half (``observe_tool_round``) runs once per tool round before dispatch; the
injection half (``inject_execution_economy_checkpoint``) runs on the incremental-persist path in
``agent.tool_executor._flush_session_db_after_tool_progress``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

logger = logging.getLogger("agent.conversation_loop")

# Consecutive tool rounds that each contained exactly ONE call before the batch hint fires.
# Deliberately high: serial read -> patch -> test cycling is normal work, and this only flags a
# long unbroken stretch with no attempt to batch independent calls.
SINGLE_TOOL_ROUND_THRESHOLD = 8
# Consecutive rounds whose normalized call set is identical (same tools, same canonical args).
# One repeat can be a legitimate double-check; three in a row is a loop.
REPEAT_ROUND_THRESHOLD = 3
# Per-turn notice cap, and the minimum number of rounds between two notices (cooldown).
MAX_NOTICES_PER_TURN = 2
COOLDOWN_ROUNDS = 6
# Hard bound on the appended text, so a checkpoint can never grow a tool result beyond this.
MAX_NOTICE_CHARS = 400
# Tool names are quoted in the notice; bound the list so it stays one short line.
MAX_TOOL_NAMES_CHARS = 80

SINGLE_TOOL_ROUNDS = "single_tool_rounds"
REPEATED_CALLS = "repeated_calls"

_SINGLE_TOOL_NOTICE = (
    "[hermes note: {count} tool rounds in a row each issued a single call. If upcoming calls are "
    "independent, issue them together in one round to save round-trips. Advisory only — nothing was "
    "blocked or changed; keep working as you see fit.]"
)
_REPEATED_CALL_NOTICE = (
    "[hermes note: the last {count} rounds repeated the same normalized call set ({tools}). An "
    "equivalent re-run rarely adds information; vary the arguments or use the results you already "
    "have. Advisory only — nothing was blocked or changed; keep working as you see fit.]"
)


@dataclass
class ExecutionEconomyState:
    """Per-turn tracker; a fresh instance is installed by ``agent.turn_context`` each turn."""

    rounds: int = 0
    single_tool_streak: int = 0
    repeat_streak: int = 0
    last_signature: Tuple[str, ...] = ()
    notices: int = 0
    last_notice_round: int = 0
    # (reason, count, tool names) latched for THIS round's tool result.
    pending: Optional[Tuple[str, int, str]] = None


def reset_state(agent: Any) -> None:
    """Install a fresh per-turn tracker (``None`` while the feature is off)."""
    agent._execution_economy = (
        ExecutionEconomyState() if getattr(agent, "_execution_economy_checkpoints", False) else None
    )


def enabled_state(agent: Any) -> Optional[ExecutionEconomyState]:
    """The active tracker, or ``None`` when the feature is off / the state is missing."""
    if not getattr(agent, "_execution_economy_checkpoints", False):
        return None
    state = getattr(agent, "_execution_economy", None)
    return state if isinstance(state, ExecutionEconomyState) else None


def observe_tool_round(agent: Any, tool_calls: Sequence[Any]) -> None:
    """Count one tool round and latch a pending notice when a threshold is crossed.

    Called once per round before dispatch. Advisory only: the return value is deliberately unused,
    so no caller can turn this into a gate.
    """
    state = enabled_state(agent)
    if state is None or not tool_calls:
        return
    # A notice latched by an earlier round that never reached an unpersisted tool result
    # (interrupt, cancelled batch) is dropped, never carried into a later round.
    state.pending = None
    state.rounds += 1
    signature = _round_signature(tool_calls)
    state.repeat_streak = state.repeat_streak + 1 if signature == state.last_signature else 1
    state.last_signature = signature
    state.single_tool_streak = state.single_tool_streak + 1 if len(tool_calls) == 1 else 0

    if state.notices >= MAX_NOTICES_PER_TURN:
        return
    if state.last_notice_round and state.rounds - state.last_notice_round < COOLDOWN_ROUNDS:
        return
    names = ", ".join(sorted({_tool_name(tc) for tc in tool_calls}))[:MAX_TOOL_NAMES_CHARS]
    if state.repeat_streak >= REPEAT_ROUND_THRESHOLD:
        state.pending = (REPEATED_CALLS, state.repeat_streak, names)
    elif state.single_tool_streak >= SINGLE_TOOL_ROUND_THRESHOLD:
        state.pending = (SINGLE_TOOL_ROUNDS, state.single_tool_streak, names)


def inject_execution_economy_checkpoint(agent: Any, messages: Any) -> bool:
    """Append the pending bounded notice to the newest unpersisted tool result.

    Runs on the incremental-persist path just before the flush, so the notice becomes durable with
    the result the model reads and no already-written row is ever rewritten. Returns True when a
    notice was appended.
    """
    state = enabled_state(agent)
    if state is None or state.pending is None:
        return False
    if getattr(agent, "_interrupt_requested", False):
        return False  # cancelled turns are not annotated (their results still persist)

    from agent.context_compressor import _DB_PERSISTED_MARKER

    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        if message.get(_DB_PERSISTED_MARKER):
            return False  # durable row: annotating it would rewrite history
        reason, count, tools = state.pending
        notice = _render_notice(reason, count, tools)
        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = content + f"\n\n{notice}"
        elif isinstance(content, list) or content is None:
            # Multimodal result: append a text block so image blocks stay untouched.
            message["content"] = [*(content or []), {"type": "text", "text": notice}]
        else:
            return False
        state.pending = None
        state.notices += 1
        state.last_notice_round = state.rounds
        logger.info(
            "execution economy checkpoint appended: reason=%s count=%d tool=%s", reason, count, tools
        )
        return True
    return False


def _render_notice(reason: str, count: int, tools: str) -> str:
    template = _REPEATED_CALL_NOTICE if reason == REPEATED_CALLS else _SINGLE_TOOL_NOTICE
    return template.format(count=count, tools=tools or "unknown")[:MAX_NOTICE_CHARS]


def _round_signature(tool_calls: Sequence[Any]) -> Tuple[str, ...]:
    """Normalized identity of a round: per call, the tool name plus a hash of its canonical args.

    Reuses the guardrails' ``ToolCallSignature`` canonicalization (sorted compact JSON -> sha256),
    so reformatted arguments and reordered keys compare equal and raw arguments never enter this
    state. Unparsable arguments fall back to the bounded raw text.
    """
    signature = []
    for tool_call in tool_calls:
        raw = getattr(getattr(tool_call, "function", None), "arguments", None)
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            args = None
        digest = ""
        if isinstance(args, dict):
            try:
                from agent.tool_guardrails import ToolCallSignature

                digest = ToolCallSignature.from_call(_tool_name(tool_call), args).args_hash
            except Exception:
                digest = ""
        signature.append(f"{_tool_name(tool_call)}|{digest or str(raw)[:120]}")
    return tuple(signature)


def _tool_name(tool_call: Any) -> str:
    return getattr(getattr(tool_call, "function", None), "name", "") or "tool"
