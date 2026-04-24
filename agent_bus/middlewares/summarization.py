"""SummarizationMiddleware — rolls up old messages when history gets long.

Inspired by DeerFlow's SummarizationMiddleware (§1.B #9 + §1.F.2).

Strategy
--------
- Estimate token count (rough: 1 token ≈ 4 chars)
- If total estimate > `trigger_tokens` (default 80_000), compress:
  - Keep last N messages verbatim (default 10)
  - Compress older messages into a single system message summary
  - The summary is a structured placeholder (for MVP) or a real LLM call
    (opt-in via env var)

Runs at `before_model` so the LLM sees the trimmed history. Env var gated
so it only triggers on long sessions and can be turned off per-thread.

Env vars
--------
HERMES_MW_SUMMARIZATION      off | core (default core)
HERMES_SUMM_TRIGGER_TOKENS   default 80000
HERMES_SUMM_KEEP_LAST        default 10
HERMES_SUMM_LLM              on | off (default off; MVP uses cheap placeholder)
"""

from __future__ import annotations

import logging
import os

from agent_bus.middleware import BaseMiddleware, MiddlewareContext

logger = logging.getLogger(__name__)

DEFAULT_TRIGGER_TOKENS = 80_000
DEFAULT_KEEP_LAST = 10


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough estimate: 1 token ≈ 4 chars of content (conservative)."""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    total += len(part.get("text", "") or "")
        # Approximate overhead per message
        total += 20
    return total // 4


def _placeholder_summarize(messages: list[dict]) -> str:
    """MVP: compact structured summary without LLM.

    Enumerates role counts and first-line titles. Good enough to preserve
    ordering hints without the cost of an LLM call.
    """
    if not messages:
        return "(empty)"
    by_role: dict[str, int] = {}
    first_lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        by_role[role] = by_role.get(role, 0) + 1
        c = m.get("content")
        if isinstance(c, str):
            first = c.split("\n", 1)[0].strip()[:80]
            if first:
                first_lines.append(f"{role}: {first}")
    role_summary = ", ".join(f"{k}×{v}" for k, v in by_role.items())
    sample = "\n  - " + "\n  - ".join(first_lines[:8]) if first_lines else ""
    return (
        f"[prior context summary — {len(messages)} messages: {role_summary}]{sample}"
    )


class SummarizationMiddleware(BaseMiddleware):
    """Compress early history when token estimate crosses threshold."""

    name = "summarization"

    def before_model(self, ctx: MiddlewareContext) -> MiddlewareContext:
        msgs = ctx.messages
        if not msgs:
            return ctx
        trigger = int(os.environ.get("HERMES_SUMM_TRIGGER_TOKENS", str(DEFAULT_TRIGGER_TOKENS)))
        keep_last = int(os.environ.get("HERMES_SUMM_KEEP_LAST", str(DEFAULT_KEEP_LAST)))

        estimate = _estimate_tokens(msgs)
        if estimate <= trigger:
            return ctx
        if len(msgs) <= keep_last:
            return ctx

        head = msgs[:-keep_last]
        tail = msgs[-keep_last:]

        # Preserve any system message at the head verbatim (don't compress)
        leading_system = []
        compressible = head
        if head and head[0].get("role") == "system":
            leading_system = [head[0]]
            compressible = head[1:]
        if not compressible:
            return ctx

        use_llm = os.environ.get("HERMES_SUMM_LLM", "off").lower() == "on"
        if use_llm:
            summary_text = self._llm_summarize(compressible)
        else:
            summary_text = _placeholder_summarize(compressible)

        summary_msg = {
            "role": "system",
            "content": summary_text,
            "_summary": True,
            "_compressed_count": len(compressible),
        }
        ctx.messages = leading_system + [summary_msg] + tail

        before_tok = estimate
        after_tok = _estimate_tokens(ctx.messages)
        ctx.record(
            self.name, "before_model", "compressed",
            f"msgs {len(msgs)}→{len(ctx.messages)} tokens≈{before_tok}→{after_tok}",
        )
        return ctx

    def _llm_summarize(self, messages: list[dict]) -> str:
        """Placeholder for real LLM call. Opt-in via env var when ready."""
        logger.debug("LLM summarize placeholder — %d msgs", len(messages))
        return _placeholder_summarize(messages)
