"""Non-destructive prompt-view compaction for stale heavy payloads."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from agent.model_metadata import estimate_messages_tokens_rough


DEFAULT_PROTECT_LAST_N = 20
DEFAULT_MIN_CHARS = 12_000
DEFAULT_PREVIEW_CHARS = 600


@dataclass
class ContextAssemblyStats:
    """Stats for payloads compacted out of an API-only prompt view."""

    messages_compacted: int = 0
    tool_results_compacted: int = 0
    tool_call_args_compacted: int = 0
    media_parts_compacted: int = 0
    estimated_tokens_before: int = 0
    estimated_tokens_after: int = 0

    @property
    def estimated_tokens_evicted(self) -> int:
        return max(0, self.estimated_tokens_before - self.estimated_tokens_after)


def compact_stale_payloads_for_prompt(
    messages: List[Dict[str, Any]],
    *,
    enabled: bool = True,
    protect_last_n: int = DEFAULT_PROTECT_LAST_N,
    min_chars: int = DEFAULT_MIN_CHARS,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    preserve_tools: Optional[Iterable[str]] = None,
) -> tuple[List[Dict[str, Any]], ContextAssemblyStats]:
    """Return an API-only message view with old heavy payloads replaced.

    The returned list is detached from ``messages`` only when a payload is
    compacted.  Callers may therefore pass canonical conversation history; this
    helper never mutates it.
    """

    stats = ContextAssemblyStats()
    if not enabled or not messages:
        return messages, stats

    protect_last_n = max(0, int(protect_last_n))
    min_chars = max(0, int(min_chars))
    preview_chars = max(0, int(preview_chars))
    protected = _protected_message_indexes(messages, protect_last_n)
    preserve_tool_names = {str(t) for t in (preserve_tools or []) if str(t)}

    compacted: Optional[List[Dict[str, Any]]] = None
    changed_indexes: Set[int] = set()

    def editable(index: int) -> Dict[str, Any]:
        nonlocal compacted
        if compacted is None:
            compacted = [copy.deepcopy(m) for m in messages]
        changed_indexes.add(index)
        return compacted[index]

    for idx, msg in enumerate(messages):
        if idx in protected or not isinstance(msg, dict):
            continue

        role = msg.get("role")
        tool_name = str(msg.get("tool_name") or "")
        if role == "tool" and tool_name not in preserve_tool_names:
            content = msg.get("content")
            if _payload_chars(content) >= min_chars:
                new_msg = editable(idx)
                new_msg["content"] = _tool_result_placeholder(
                    tool_name=tool_name or "unknown",
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    content=content,
                    preview_chars=preview_chars,
                )
                stats.tool_results_compacted += 1

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            new_calls = None
            compacted_args = 0
            for call_idx, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name") or "unknown")
                if name in preserve_tool_names:
                    continue
                arguments = function.get("arguments")
                if not isinstance(arguments, str) or len(arguments) < min_chars:
                    continue
                if new_calls is None:
                    new_calls = copy.deepcopy(tool_calls)
                new_calls[call_idx]["function"]["arguments"] = _compact_arguments_json(
                    arguments,
                    tool_name=name,
                    preview_chars=preview_chars,
                )
                compacted_args += 1
            if compacted_args:
                editable(idx)["tool_calls"] = new_calls
                stats.tool_call_args_compacted += compacted_args

        content = msg.get("content")
        replaced = _compact_media_content(content)
        if replaced is not content:
            editable(idx)["content"] = replaced
            stats.media_parts_compacted += 1

    if compacted is None:
        return messages, stats

    stats.messages_compacted = len(changed_indexes)
    stats.estimated_tokens_before = estimate_messages_tokens_rough(messages)
    stats.estimated_tokens_after = estimate_messages_tokens_rough(compacted)
    return compacted, stats


def context_assembly_config_from_mapping(
    compression_config: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Normalize ``compression.context_assembly`` settings."""

    raw = {}
    if isinstance(compression_config, Mapping):
        value = compression_config.get("context_assembly", {})
        if isinstance(value, Mapping):
            raw = dict(value)

    return {
        "enabled": _as_bool(raw.get("enabled", True), default=True),
        "protect_last_n": _as_int(
            raw.get("protect_last_n", DEFAULT_PROTECT_LAST_N),
            default=DEFAULT_PROTECT_LAST_N,
            minimum=0,
        ),
        "min_chars": _as_int(
            raw.get("min_chars", DEFAULT_MIN_CHARS),
            default=DEFAULT_MIN_CHARS,
            minimum=1,
        ),
        "preview_chars": _as_int(
            raw.get("preview_chars", DEFAULT_PREVIEW_CHARS),
            default=DEFAULT_PREVIEW_CHARS,
            minimum=0,
        ),
        "preserve_tools": _as_str_list(raw.get("preserve_tools", [])),
    }


def _protected_message_indexes(messages: List[Dict[str, Any]], protect_last_n: int) -> Set[int]:
    if protect_last_n <= 0:
        return set()
    non_system = [
        idx for idx, msg in enumerate(messages)
        if isinstance(msg, dict) and msg.get("role") != "system"
    ]
    return set(non_system[-protect_last_n:])


def _payload_chars(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(str(value))


def _preview(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = max(0, limit // 2)
    tail = max(0, limit - head)
    return f"{text[:head]}\n...\n{text[-tail:]}"


def _tool_result_placeholder(
    *,
    tool_name: str,
    tool_call_id: str,
    content: Any,
    preview_chars: int,
) -> str:
    size = _payload_chars(content)
    preview = _preview(content, preview_chars)
    lines = [
        "[stale tool result compacted before model call]",
        f"tool: {tool_name}",
        f"tool_call_id: {tool_call_id or 'unknown'}",
        f"original_chars: {size}",
    ]
    if preview:
        lines.extend(["preview:", preview])
    return "\n".join(lines)


def _compact_arguments_json(arguments: str, *, tool_name: str, preview_chars: int) -> str:
    preview = _preview(arguments, preview_chars)
    placeholder = {
        "_hermes_compacted_stale_tool_arguments": True,
        "tool": tool_name,
        "original_chars": len(arguments),
        "preview": preview,
    }
    return json.dumps(placeholder, separators=(",", ":"), sort_keys=True)


def _compact_media_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content

    changed = False
    compacted = []
    for part in content:
        if not isinstance(part, dict):
            compacted.append(part)
            continue
        ptype = part.get("type")
        if ptype in {"image", "image_url", "input_image"}:
            changed = True
            compacted.append({
                "type": "text",
                "text": "[stale media payload compacted before model call]",
            })
        else:
            compacted.append(part)
    return compacted if changed else content


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, *, default: int, minimum: int) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _as_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]
