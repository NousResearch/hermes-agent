"""Context budget diagnostics for Hermes model requests.

The estimators here are intentionally rough. They do not try to reproduce a
provider tokenizer; they split the outgoing request into human-useful buckets so
operators can see what is filling the window: stable system prompt, recalled
memory, conversation/tool history, current user text, and tool schemas.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from agent.model_metadata import estimate_messages_tokens_rough

_MEMORY_BLOCK_RE = re.compile(
    r"<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>",
    re.IGNORECASE,
)


def _tokens_for_text(value: Any) -> int:
    if value is None:
        return 0
    return (len(str(value)) + 3) // 4


def _message_tokens(message: Dict[str, Any]) -> int:
    return estimate_messages_tokens_rough([message])


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def _strip_memory_blocks(text: str) -> tuple[str, int, int]:
    blocks = _MEMORY_BLOCK_RE.findall(text or "")
    memory_chars = sum(len(block) for block in blocks)
    memory_tokens = sum(_tokens_for_text(block) for block in blocks)
    stripped = _MEMORY_BLOCK_RE.sub("", text or "")
    return stripped, memory_chars, memory_tokens


def build_context_budget_report(
    *,
    api_messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None,
    model: str = "",
    provider: str = "",
    context_length: int | None = None,
    session_id: str = "",
    api_call_count: int = 0,
) -> Dict[str, Any]:
    """Return a rough per-bucket token budget for an outgoing request."""
    api_messages = list(api_messages or [])
    tools = list(tools or [])

    system_tokens = 0
    system_chars = 0
    message_start = 0
    if api_messages and api_messages[0].get("role") == "system":
        system_chars = len(_content_text(api_messages[0].get("content")))
        system_tokens = _message_tokens(api_messages[0])
        message_start = 1

    current_user_idx = -1
    for idx in range(len(api_messages) - 1, message_start - 1, -1):
        if api_messages[idx].get("role") == "user":
            current_user_idx = idx
            break

    current_user_tokens = 0
    memory_context_tokens = 0
    memory_context_chars = 0
    prior_messages_tokens = 0
    role_tokens: dict[str, int] = {}
    largest_messages: list[dict[str, Any]] = []

    for idx, msg in enumerate(api_messages[message_start:], start=message_start):
        role = str(msg.get("role") or "unknown")
        tokens = _message_tokens(msg)
        role_tokens[role] = role_tokens.get(role, 0) + tokens
        largest_messages.append({"index": idx, "role": role, "tokens": tokens})

        if idx == current_user_idx:
            content = _content_text(msg.get("content"))
            stripped, mem_chars, mem_tokens = _strip_memory_blocks(content)
            memory_context_chars += mem_chars
            memory_context_tokens += mem_tokens
            if mem_tokens:
                current_user_shadow = dict(msg)
                current_user_shadow["content"] = stripped
                current_user_tokens = _message_tokens(current_user_shadow)
            else:
                current_user_tokens = tokens
        else:
            prior_messages_tokens += tokens

    largest_messages.sort(key=lambda item: int(item.get("tokens") or 0), reverse=True)
    largest_messages = largest_messages[:5]

    tool_schema_tokens = _tokens_for_text(tools) if tools else 0
    message_tokens = estimate_messages_tokens_rough(api_messages)
    total_tokens = message_tokens + tool_schema_tokens
    pct = None
    if context_length:
        try:
            pct = round((total_tokens / int(context_length)) * 100, 1)
        except Exception:
            pct = None

    buckets = {
        "system_prompt": system_tokens,
        "memory_context": memory_context_tokens,
        "current_user": current_user_tokens,
        "prior_messages": prior_messages_tokens,
        "tool_schemas": tool_schema_tokens,
    }
    # Account for sanitizer/prefill/role wrapper deltas so bucket sum matches the
    # rough total instead of hiding estimator drift.
    bucket_sum = sum(buckets.values())
    if total_tokens > bucket_sum:
        buckets["other_request_overhead"] = total_tokens - bucket_sum

    return {
        "model": model,
        "provider": provider,
        "session_id": session_id,
        "api_call": api_call_count,
        "message_count": len(api_messages),
        "tool_count": len(tools),
        "context_length": context_length,
        "total_tokens": total_tokens,
        "message_tokens": message_tokens,
        "percent_of_context": pct,
        "buckets": buckets,
        "role_tokens": role_tokens,
        "largest_messages": largest_messages,
        "memory_context_chars": memory_context_chars,
        "system_prompt_chars": system_chars,
    }


def format_context_budget_report(report: Dict[str, Any], *, markdown: bool = False) -> List[str]:
    """Format a report as compact human-readable lines."""
    if not report:
        return []
    total = int(report.get("total_tokens") or 0)
    context_length = report.get("context_length") or 0
    pct = report.get("percent_of_context")
    pct_part = f" · {pct:.1f}%" if isinstance(pct, (int, float)) else ""
    if context_length:
        header = f"Context budget: ~{total:,}/{int(context_length):,} tokens{pct_part}"
    else:
        header = f"Context budget: ~{total:,} tokens"
    lines = [header]
    buckets = report.get("buckets") or {}
    if buckets:
        ordered = sorted(buckets.items(), key=lambda kv: int(kv[1] or 0), reverse=True)
        lines.append("Buckets: " + ", ".join(f"{name} ~{int(value):,}" for name, value in ordered if value))
    lines.append(
        f"Request shape: {int(report.get('message_count') or 0)} messages, "
        f"{int(report.get('tool_count') or 0)} tools"
    )
    largest = report.get("largest_messages") or []
    if largest:
        top = ", ".join(
            f"#{item.get('index')} {item.get('role')} ~{int(item.get('tokens') or 0):,}"
            for item in largest[:3]
        )
        lines.append("Largest messages: " + top)
    return lines


def compact_context_budget_log(report: Dict[str, Any]) -> str:
    if not report:
        return "context_budget unavailable"
    buckets = report.get("buckets") or {}
    bucket_text = " ".join(
        f"{name}={int(value or 0)}"
        for name, value in sorted(buckets.items(), key=lambda kv: int(kv[1] or 0), reverse=True)
        if value
    )
    pct = report.get("percent_of_context")
    pct_text = f" pct={pct:.1f}" if isinstance(pct, (int, float)) else ""
    return (
        f"context_budget session={report.get('session_id') or '-'} "
        f"call={report.get('api_call') or 0} total={int(report.get('total_tokens') or 0)}"
        f" context_length={report.get('context_length') or 0}{pct_text} "
        f"messages={report.get('message_count') or 0} tools={report.get('tool_count') or 0} "
        f"{bucket_text}"
    ).strip()
