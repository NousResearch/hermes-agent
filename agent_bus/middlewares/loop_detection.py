"""LoopDetectionMiddleware — detects repeated tool calls and hard-stops the
agent into a text response.

Why
---
DeerFlow's `LoopDetectionMiddleware` (§1.B #17) detects when the agent calls
the same tool with the same arguments N times in a row — a runaway loop. It
strips the tool_calls on the current AIMessage so the agent is forced to
produce a textual answer.

Heuristic
---------
Look at the last K AIMessages (default K=5). If the most recent AIMessage has
a tool_call whose `(name, arguments)` signature appears in ≥ `threshold`
(default 3) of those K messages → loop detected.

On detection, this middleware:
1. Clears `tool_calls` on the most recent AIMessage.
2. Appends a short reminder in its `content` so the agent sees why it was
   stopped.
3. Records the decision in ctx.decisions.

Runs at `after_model` — right after LLM returns, before tool dispatch.

Env var: HERMES_MW_LOOP_DETECT (off / core)
Config via ctx.metadata:
    loop_window    — window size (default 5)
    loop_threshold — repetitions to trip (default 3)
"""

from __future__ import annotations

import json
from typing import Any

from agent_bus.middleware import BaseMiddleware, MiddlewareContext, register


def _tool_call_signature(tc: Any) -> str:
    if not isinstance(tc, dict):
        return ""
    name = tc.get("name") or tc.get("function", {}).get("name") or ""
    args = tc.get("args") or tc.get("arguments") or tc.get("function", {}).get("arguments") or {}
    if isinstance(args, dict):
        try:
            args_repr = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except TypeError:
            args_repr = repr(sorted(args.items()))
    else:
        args_repr = str(args)
    return f"{name}|{args_repr}"


class LoopDetectionMiddleware(BaseMiddleware):
    """Detect repeated tool_calls and force a text response."""

    name = "loop-detection"

    def after_model(self, ctx: MiddlewareContext) -> MiddlewareContext:
        msgs = ctx.messages
        if not msgs:
            return ctx

        window = int(ctx.metadata.get("loop_window", 5))
        threshold = int(ctx.metadata.get("loop_threshold", 3))

        # Find AI messages (most recent first)
        ai_messages = [m for m in msgs if m.get("role") == "assistant"]
        if not ai_messages:
            return ctx
        recent = ai_messages[-window:]
        latest = ai_messages[-1]

        latest_tcs = latest.get("tool_calls") or []
        if not latest_tcs:
            return ctx

        # For each tool_call on the latest AIMessage, count matches across the
        # recent window. If any one exceeds threshold, trip the loop.
        tripped_signature: str | None = None
        tripped_count = 0
        for tc in latest_tcs:
            sig = _tool_call_signature(tc)
            if not sig:
                continue
            cnt = 0
            for m in recent:
                for m_tc in (m.get("tool_calls") or []):
                    if _tool_call_signature(m_tc) == sig:
                        cnt += 1
                        break  # one per AI message
            if cnt >= threshold:
                tripped_signature = sig
                tripped_count = cnt
                break

        if tripped_signature is None:
            return ctx

        # Hard-stop: clear tool_calls on the latest AIMessage and nudge with text
        latest["tool_calls"] = []
        existing = latest.get("content") or ""
        reminder = (
            f"\n\n[system: detected repeated tool call "
            f"`{tripped_signature[:60]}...` {tripped_count}× in last {window} turns. "
            f"Stopping tool loop — respond in plain text with what you know so far.]"
        )
        latest["content"] = (existing + reminder).strip() if existing else reminder.strip()
        latest["_loop_stopped"] = True

        ctx.record(
            self.name, "after_model", "loop-stopped",
            f"sig={tripped_signature[:80]} count={tripped_count} window={window}",
        )
        return ctx
