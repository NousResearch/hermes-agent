"""Live session context-window breakdown for UI surfaces.

Estimates how the next provider request is composed: system prompt tiers,
tool schemas, and conversation history. Uses the same rough char/4 heuristic
as ``agent.model_metadata.estimate_request_tokens_rough`` so numbers align
with compression thresholds — not exact tokenizer counts.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SKILLS_BLOCK_RE = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)

_SUBAGENT_TOOL_NAMES = frozenset({"delegate_task"})

_CATEGORY_COLORS = {
    "system_prompt": "var(--context-usage-system)",
    "tool_definitions": "var(--context-usage-tools)",
    "rules": "var(--context-usage-rules)",
    "skills": "var(--context-usage-skills)",
    "mcp": "var(--context-usage-mcp)",
    "subagent_definitions": "var(--context-usage-subagents)",
    "memory": "var(--context-usage-memory)",
    "conversation": "var(--context-usage-conversation)",
}

_CONTEXT_FULL_SOURCE_LABEL = (
    "Reconstructed base context (cached prefix + history; excludes per-turn "
    "ephemeral injections)"
)
_CONTEXT_FULL_SLICE_CHAR_CAP = 200_000


def _chars_to_tokens(text: str) -> int:
    if not text:
        return 0
    return (len(text) + 3) // 4


def _json_tokens(value: Any) -> int:
    if not value:
        return 0
    return _chars_to_tokens(json.dumps(value, ensure_ascii=False))


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _truncate_slice_text(text: str) -> Tuple[str, bool]:
    if len(text) <= _CONTEXT_FULL_SLICE_CHAR_CAP:
        return text, False
    note = (
        "\n\n[Context Inspector note: slice truncated after "
        f"{_CONTEXT_FULL_SLICE_CHAR_CAP} characters; original length "
        f"{len(text)} characters.]"
    )
    return text[:_CONTEXT_FULL_SLICE_CHAR_CAP] + note, True


def _tool_name(tool: dict) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return str(tool.get("name") or "")


def _split_tools(tools: Sequence[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    builtin: List[dict] = []
    mcp: List[dict] = []
    subagent: List[dict] = []
    for tool in tools:
        name = _tool_name(tool)
        if name.startswith("mcp_"):
            mcp.append(tool)
        elif name in _SUBAGENT_TOOL_NAMES:
            subagent.append(tool)
        else:
            builtin.append(tool)
    return builtin, mcp, subagent


def _memory_blocks(agent: Any) -> Tuple[str, str]:
    memory_block = ""
    user_block = ""
    store = getattr(agent, "_memory_store", None)
    if store is None:
        return memory_block, user_block
    try:
        if getattr(agent, "_memory_enabled", True):
            memory_block = store.format_for_system_prompt("memory") or ""
        if getattr(agent, "_user_profile_enabled", True):
            user_block = store.format_for_system_prompt("user") or ""
    except Exception:
        pass
    return memory_block, user_block


def _strip_blocks(text: str, *blocks: str) -> str:
    out = text
    for block in blocks:
        if block:
            out = out.replace(block, "")
    return out.strip()


def _message_content_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _context_full_slice(
    slice_id: str,
    label: str,
    bucket: str,
    content_text: str,
    source_accuracy: str,
    tokens: Optional[int] = None,
) -> Dict[str, Any]:
    shown_text, truncated = _truncate_slice_text(content_text or "")
    return {
        "id": slice_id,
        "label": label,
        "bucket": bucket,
        "content_text": shown_text,
        "source_accuracy": source_accuracy,
        "tokens": _chars_to_tokens(content_text or "") if tokens is None else tokens,
        "truncated": truncated,
    }


def _compose_cold_system_prompt(parts: Dict[str, str]) -> str:
    return "\n\n".join(
        part
        for part in (
            parts.get("stable", "") or "",
            parts.get("context", "") or "",
            parts.get("volatile", "") or "",
        )
        if part
    ).strip()


def compute_session_context_full(
    agent: Any,
    messages: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Return raw reconstructed-base context slices plus ordered messages.

    The ordered system message is cache-safe: prefer the agent's already-cached
    system prompt bytes and never rebuild them mid-conversation. Layered slices
    are best-effort current decomposition for inspection only.
    """
    history = list(messages or [])

    cached_system = getattr(agent, "_cached_system_prompt", None)
    cached_text = cached_system if isinstance(cached_system, str) else ""
    has_cached_system = bool(cached_text)
    if has_cached_system:
        system_text: str = cached_text
        parts: Dict[str, str] = {"stable": cached_text, "context": "", "volatile": ""}
        system_accuracy = "cached_exact"
    else:
        from agent.system_prompt import build_system_prompt_parts

        raw_parts = build_system_prompt_parts(agent)
        parts = {
            "stable": str(raw_parts.get("stable", "") or ""),
            "context": str(raw_parts.get("context", "") or ""),
            "volatile": str(raw_parts.get("volatile", "") or ""),
        }
        system_text = _compose_cold_system_prompt(parts)
        system_accuracy = "reconstructed_current"

    ephemeral = getattr(agent, "ephemeral_system_prompt", None)
    if ephemeral:
        system_text = (system_text + "\n\n" + str(ephemeral)).strip()

    stable = parts.get("stable", "") or ""
    context = parts.get("context", "") or ""
    volatile = parts.get("volatile", "") or ""

    skills_match = _SKILLS_BLOCK_RE.search(stable)
    skills_index = skills_match.group(0) if skills_match else ""
    memory_block, user_block = _memory_blocks(agent)
    memory_text = "\n\n".join(part for part in (memory_block, user_block) if part).strip()

    if has_cached_system:
        system_prompt_text = system_text
    else:
        system_core = _strip_blocks(stable, skills_index)
        system_tail = _strip_blocks(volatile, memory_block, user_block)
        system_prompt_text = "\n\n".join(part for part in (system_core, system_tail) if part).strip()

    tools = list(getattr(agent, "tools", None) or [])
    builtin_tools, mcp_tools, subagent_tools = _split_tools(tools)
    copied_history = [deepcopy(msg) for msg in history]
    conversation_text = _json_text(copied_history) if copied_history else ""

    slices = [
        _context_full_slice(
            "system_prompt",
            "System prompt",
            "system",
            system_prompt_text,
            system_accuracy,
        ),
        _context_full_slice("rules", "Rules", "system", context, "reconstructed_current"),
        _context_full_slice("skills", "Skills", "system", skills_index, "reconstructed_current"),
        _context_full_slice("memory", "Memory", "system", memory_text, "reconstructed_current"),
        _context_full_slice(
            "tool_definitions",
            "Tool definitions",
            "tools",
            _json_text(builtin_tools) if builtin_tools else "",
            "reconstructed_current",
            _json_tokens(builtin_tools),
        ),
        _context_full_slice(
            "mcp",
            "MCP",
            "tools",
            _json_text(mcp_tools) if mcp_tools else "",
            "reconstructed_current",
            _json_tokens(mcp_tools),
        ),
        _context_full_slice(
            "subagent_definitions",
            "Subagent definitions",
            "tools",
            _json_text(subagent_tools) if subagent_tools else "",
            "reconstructed_current",
            _json_tokens(subagent_tools),
        ),
        _context_full_slice(
            "conversation",
            "Conversation",
            "conversation",
            conversation_text,
            "reconstructed_current",
            _json_tokens(copied_history),
        ),
    ]

    ordered_messages: List[Dict[str, Any]] = []
    if system_text:
        system_raw = {"role": "system", "content": system_text}
        ordered_messages.append(
            {
                "index": 0,
                "role": "system",
                "content_text": system_text,
                "raw": system_raw,
                "tokens": _json_tokens(system_raw),
            }
        )
    for raw in copied_history:
        content_text = _message_content_text(raw)
        ordered_messages.append(
            {
                "index": len(ordered_messages),
                "role": str(raw.get("role") or ""),
                "content_text": content_text,
                "raw": raw,
                "tokens": _json_tokens(raw),
            }
        )

    estimated_total = sum(item["tokens"] for item in slices)
    comp = getattr(agent, "context_compressor", None)
    context_max = int(getattr(comp, "context_length", 0) or 0) if comp else 0
    measured_used = int(getattr(comp, "last_prompt_tokens", 0) or 0) if comp else 0
    if measured_used < 0:
        measured_used = 0
    context_used = measured_used if measured_used > 0 else estimated_total

    return {
        "available": True,
        "state": "ready",
        "source": "reconstructed_base",
        "source_label": _CONTEXT_FULL_SOURCE_LABEL,
        "raw_unredacted": True,
        "model": getattr(agent, "model", "") or "",
        "context_max": context_max,
        "context_used": context_used,
        "slices": slices,
        "messages": ordered_messages,
        "exact_capture_available": False,
    }


def compute_session_context_breakdown(
    agent: Any,
    messages: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Return a Cursor-style context usage breakdown for one live agent."""
    from agent.model_metadata import estimate_messages_tokens_rough
    from agent.system_prompt import build_system_prompt_parts

    parts = build_system_prompt_parts(agent)
    stable = parts.get("stable", "") or ""
    context = parts.get("context", "") or ""
    volatile = parts.get("volatile", "") or ""

    skills_match = _SKILLS_BLOCK_RE.search(stable)
    skills_index = skills_match.group(0) if skills_match else ""

    memory_block, user_block = _memory_blocks(agent)
    memory_text = "\n\n".join(part for part in (memory_block, user_block) if part).strip()

    system_core = _strip_blocks(stable, skills_index)
    system_tail = _strip_blocks(volatile, memory_block, user_block)
    system_prompt_text = "\n\n".join(part for part in (system_core, system_tail) if part).strip()

    tools = list(getattr(agent, "tools", None) or [])
    builtin_tools, mcp_tools, subagent_tools = _split_tools(tools)

    conversation_tokens = estimate_messages_tokens_rough(messages or [])

    categories = [
        ("system_prompt", "System prompt", _chars_to_tokens(system_prompt_text)),
        ("tool_definitions", "Tool definitions", _json_tokens(builtin_tools)),
        ("rules", "Rules", _chars_to_tokens(context)),
        ("skills", "Skills", _chars_to_tokens(skills_index)),
        ("mcp", "MCP", _json_tokens(mcp_tools)),
        ("subagent_definitions", "Subagent definitions", _json_tokens(subagent_tools)),
        ("memory", "Memory", _chars_to_tokens(memory_text)),
        ("conversation", "Conversation", conversation_tokens),
    ]

    estimated_total = sum(tokens for _, _, tokens in categories)

    comp = getattr(agent, "context_compressor", None)
    context_max = int(getattr(comp, "context_length", 0) or 0) if comp else 0
    measured_used = int(getattr(comp, "last_prompt_tokens", 0) or 0) if comp else 0
    context_used = measured_used if measured_used > 0 else estimated_total
    context_percent = (
        max(0, min(100, round(context_used / context_max * 100)))
        if context_max
        else 0
    )

    return {
        "categories": [
            {
                "color": _CATEGORY_COLORS.get(category_id, "var(--ui-text-tertiary)"),
                "id": category_id,
                "label": label,
                "tokens": tokens,
            }
            for category_id, label, tokens in categories
            if tokens > 0
        ],
        "context_max": context_max,
        "context_percent": context_percent,
        "context_used": context_used,
        "estimated_total": estimated_total,
        "model": getattr(agent, "model", "") or "",
    }
