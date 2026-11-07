"""Content extraction from an auxiliary LLM response.

Split out of ``auxiliary_client`` so that parsing the reply stays separate from the
provider ladder. Callers keep importing these names from ``agent.auxiliary_client``,
which re-exports them.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _message_field(msg, name):
    return msg.get(name) if isinstance(msg, dict) else getattr(msg, name, None)


def _coerce_llm_message(response):
    """Pull a message (dict, object, or str) out of a response-or-message value: dict-shaped
    responses/bare messages (compression, proxies) and ChatCompletion objects; MagicMock
    ``reasoning_*`` attrs are deliberately not strings."""
    if response is None or isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "choices" not in response:
            return response
        choices = response.get("choices") or []
    else:
        choices = getattr(response, "choices", None)
        if not choices:
            return response
    return _message_field(choices[0], "message") if choices else None


def extract_content_or_reasoning(response, *, max_reasoning_chars: int | None = None) -> str:
    """Extract content from an LLM response, falling back to reasoning fields.
    Order: ``content`` (inline think blocks stripped) → ``reasoning``/``reasoning_content`` →
    ``reasoning_details`` (OpenRouter array). Accepts a response or bare message;
    ``max_reasoning_chars`` bounds a reasoning fallback so unbounded chain-of-thought can't
    become the compaction summary. Returns ``""`` if nothing found."""
    msg = _coerce_llm_message(response)
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg.strip()
    raw = _message_field(msg, "content")
    if not isinstance(raw, str):
        # A block list (Anthropic-style content: text/thinking parts) must be flattened to its
        # visible text, not `str()`-ed: repr'ing it hands the caller a Python literal full of
        # escaped quotes and thinking tokens, which the main model then has to interpret.
        from agent.agent_runtime_helpers import _flatten_content_text

        raw = _flatten_content_text(raw) if raw else ""
    content = raw.strip()
    if content:
        # Same precompiled closed-pair patterns as strip_think_blocks.
        from agent.agent_runtime_helpers import _REASONING_BLOCK_PATTERNS

        cleaned = content
        for pattern in _REASONING_BLOCK_PATTERNS:
            cleaned = pattern.sub("", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            return cleaned
    # Content is empty or reasoning-only — try structured reasoning fields
    reasoning_parts: list[str] = []
    for field in ("reasoning", "reasoning_content"):
        val = _message_field(msg, field)
        if val and isinstance(val, str) and val.strip() and val not in reasoning_parts:
            reasoning_parts.append(val.strip())
    details = _message_field(msg, "reasoning_details")
    if details and isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                summary = detail.get("summary") or detail.get("content") or detail.get("text")
                if summary and summary not in reasoning_parts:
                    reasoning_parts.append(summary.strip() if isinstance(summary, str) else str(summary))
    if not reasoning_parts:
        return ""
    text = "\n\n".join(reasoning_parts)
    if max_reasoning_chars is not None and len(text) > max_reasoning_chars:
        logger.warning(
            "fell back to reasoning fields (%d chars); truncating to %d",
            len(text),
            max_reasoning_chars,
        )
        return text[:max_reasoning_chars]
    return text


__all__ = ["_coerce_llm_message", "_message_field", "extract_content_or_reasoning"]
