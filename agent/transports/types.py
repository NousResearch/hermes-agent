"""Shared types for normalized provider responses.

These dataclasses define the canonical shape that all provider adapters
normalize responses to.  The shared surface is intentionally minimal —
only fields that every downstream consumer reads are top-level.
Protocol-specific state goes in ``provider_data`` dicts (response-level
and per-tool-call) so that protocol-aware code paths can access it
without polluting the shared type.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """A normalized tool call from any provider.

    ``id`` is the protocol's canonical identifier — what gets used in
    ``tool_call_id`` / ``tool_use_id`` when constructing tool result
    messages.  May be ``None`` when the provider omits it; the agent
    fills it via ``_deterministic_call_id()`` before storing in history.

    ``provider_data`` carries per-tool-call protocol metadata that only
    protocol-aware code reads:

    * Codex: ``{"call_id": "call_XXX", "response_item_id": "fc_XXX"}``
    * Gemini: ``{"extra_content": {"google": {"thought_signature": "..."}}}``
    * Others: ``None``
    """

    id: Optional[str]
    name: str
    arguments: str  # JSON string
    provider_data: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class Usage:
    """Token usage from an API response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class NormalizedResponse:
    """Normalized API response from any provider.

    Shared fields are truly cross-provider — every caller can rely on
    them without branching on api_mode.  Protocol-specific state goes in
    ``provider_data`` so that only protocol-aware code paths read it.

    Response-level ``provider_data`` examples:

    * Anthropic: ``{"reasoning_details": [...]}``
    * Codex: ``{"codex_reasoning_items": [...]}``
    * Others: ``None``
    """

    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]
    finish_reason: str  # "stop", "tool_calls", "length", "content_filter"
    reasoning: Optional[str] = None
    usage: Optional[Usage] = None
    provider_data: Optional[Dict[str, Any]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_tool_call(
    id: Optional[str],
    name: str,
    arguments: Any,
    **provider_fields: Any,
) -> ToolCall:
    """Build a ``ToolCall``, auto-serialising *arguments* if it's a dict.

    Any extra keyword arguments are collected into ``provider_data``.
    """
    args_str = json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
    pd = dict(provider_fields) if provider_fields else None
    return ToolCall(id=id, name=name, arguments=args_str, provider_data=pd)


def map_finish_reason(reason: Optional[str], mapping: Dict[str, str]) -> str:
    """Translate a provider-specific stop reason to the normalised set.

    Falls back to ``"stop"`` for unknown or ``None`` reasons.
    """
    if reason is None:
        return "stop"
    return mapping.get(reason, "stop")
