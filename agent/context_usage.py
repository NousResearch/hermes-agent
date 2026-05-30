"""Shared helpers for exposing agent context fill and session token usage."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_context_usage_payload(agent: Any) -> dict[str, Any]:
    """Build a normalized context-usage payload for HTTP/TUI consumers."""

    def _get(key: str, fallback: str | None = None) -> int:
        value = getattr(agent, key, 0) or 0
        if not value and fallback:
            value = getattr(agent, fallback, 0) or 0
        return int(value)

    session = {
        "input_tokens": _get("session_input_tokens", "session_prompt_tokens"),
        "output_tokens": _get("session_output_tokens", "session_completion_tokens"),
        "cache_read_tokens": _get("session_cache_read_tokens"),
        "cache_write_tokens": _get("session_cache_write_tokens"),
        "reasoning_tokens": _get("session_reasoning_tokens"),
        "prompt_tokens": _get("session_prompt_tokens"),
        "completion_tokens": _get("session_completion_tokens"),
        "total_tokens": _get("session_total_tokens"),
    }

    payload: dict[str, Any] = {
        "model": getattr(agent, "model", "") or "",
        "compressions": 0,
        "session": session,
    }

    categories = _build_categories(agent)
    if categories:
        payload["categories"] = categories

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        ctx_used = getattr(compressor, "last_prompt_tokens", 0) or session["total_tokens"] or 0
        ctx_max = getattr(compressor, "context_length", 0) or 0
        payload["context_used"] = int(ctx_used)
        payload["context_max"] = int(ctx_max)
        if ctx_max:
            payload["context_percent"] = max(
                0,
                min(100, round(ctx_used / ctx_max * 100)),
            )
        else:
            payload["context_percent"] = 0
        payload["compressions"] = int(getattr(compressor, "compression_count", 0) or 0)
    else:
        payload["context_used"] = session["total_tokens"]
        payload["context_max"] = 0
        payload["context_percent"] = 0

    try:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

        cost = estimate_usage_cost(
            payload["model"],
            CanonicalUsage(
                input_tokens=session["input_tokens"],
                output_tokens=session["output_tokens"],
                cache_read_tokens=session["cache_read_tokens"],
                cache_write_tokens=session["cache_write_tokens"],
            ),
            provider=getattr(agent, "provider", None),
            base_url=getattr(agent, "base_url", None),
        )
        payload["cost_status"] = cost.status
        if cost.amount_usd is not None:
            payload["cost_usd"] = float(cost.amount_usd)
    except Exception:
        pass

    return payload


def emit_context_usage(agent: Any) -> None:
    """Invoke ``context_usage_callback`` when configured."""
    callback = getattr(agent, "context_usage_callback", None)
    if not callable(callback):
        return
    try:
        callback(build_context_usage_payload(agent))
    except Exception:
        logger.debug("context_usage_callback failed", exc_info=True)


def _build_categories(agent: Any) -> list[dict[str, Any]]:
    """Best-effort per-section token estimate for the active prompt fill.

    Buckets reflect Hermes' actual prompt assembly: stable identity +
    guidance, project rules/context files, volatile memory, tool
    schemas, and the live conversation history. Numbers are coarse
    (~4 chars/token) — exact counting would require provider-specific
    tokenizers and isn't worth the complexity for a UI indicator.
    """
    try:
        from agent.model_metadata import (
            estimate_messages_tokens_rough,
            estimate_tokens_rough,
        )
    except Exception:
        return []

    categories: list[dict[str, Any]] = []

    def _add(key: str, label: str, tokens: int) -> None:
        if tokens > 0:
            categories.append({"key": key, "label": label, "tokens": int(tokens)})

    try:
        from agent.system_prompt import build_system_prompt_parts

        parts = build_system_prompt_parts(agent)
    except Exception:
        parts = None

    if isinstance(parts, dict):
        _add("system", "System prompt", estimate_tokens_rough(parts.get("stable") or ""))
        _add("rules", "Rules", estimate_tokens_rough(parts.get("context") or ""))
        _add("memory", "Memory", estimate_tokens_rough(parts.get("volatile") or ""))

    tools = getattr(agent, "tools", None) or []
    if tools:
        try:
            tools_json = json.dumps(tools, default=str)
            _add("tools", "Tool definitions", estimate_tokens_rough(tools_json))
        except Exception:
            pass

    history = getattr(agent, "conversation_history", None)
    if isinstance(history, list) and history:
        try:
            _add(
                "conversation",
                "Conversation",
                estimate_messages_tokens_rough(history),
            )
        except Exception:
            pass

    return categories
