"""DanglingToolCallMiddleware — injects placeholder ToolMessages for orphan
tool_calls before the next model invocation.

Why
---
When a provider or user interrupts a tool-call loop, an AIMessage with
tool_calls may not have matching ToolMessage responses. OpenAI-compatible
strict reasoning models then reject the malformed history. DeerFlow's
`DanglingToolCallMiddleware` (§1.B #4) solves this by stripping raw tool-call
metadata on forced-stop AIMessages and injecting placeholder tool results for
dangling calls.

This middleware runs at `before_model` — right before the LLM sees the
history. It finds AIMessages with tool_calls that lack corresponding
ToolMessages and injects minimal placeholder error responses so the history
stays well-formed.

Message shape (dict):
    {"role": "assistant", "content": ..., "tool_calls": [{"id": "call_x", ...}]}
    {"role": "tool", "content": ..., "tool_call_id": "call_x"}

Env var: HERMES_MW_DANGLING_TOOL (off / core)
"""

from __future__ import annotations

from agent_bus.middleware import BaseMiddleware, MiddlewareContext, register

PLACEHOLDER_TEMPLATE = (
    "[system: tool call `{tool_call_id}` has no recorded result "
    "(likely interrupted or lost). Treated as error; continue without this result.]"
)


class DanglingToolCallMiddleware(BaseMiddleware):
    """Inject placeholder ToolMessages for orphan AIMessage.tool_calls."""

    name = "dangling-tool-call"

    def before_model(self, ctx: MiddlewareContext) -> MiddlewareContext:
        msgs = ctx.messages
        if not msgs:
            return ctx

        # Collect all tool_call_ids that HAVE a response
        responded_ids: set[str] = set()
        for m in msgs:
            if m.get("role") == "tool":
                tid = m.get("tool_call_id")
                if tid:
                    responded_ids.add(tid)

        # Find orphan tool_calls in AIMessages
        fixed_count = 0
        new_msgs: list[dict] = []
        for m in msgs:
            new_msgs.append(m)
            if m.get("role") != "assistant":
                continue
            tcs = m.get("tool_calls") or []
            if not tcs:
                continue
            for tc in tcs:
                tid = tc.get("id") if isinstance(tc, dict) else None
                if not tid or tid in responded_ids:
                    continue
                # Inject placeholder immediately after this AIMessage
                placeholder = {
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": PLACEHOLDER_TEMPLATE.format(tool_call_id=tid),
                    "_placeholder": True,
                }
                new_msgs.append(placeholder)
                responded_ids.add(tid)
                fixed_count += 1

        if fixed_count:
            ctx.messages = new_msgs
            ctx.record(
                self.name, "before_model", "injected-placeholders",
                f"fixed={fixed_count}",
            )
        return ctx
