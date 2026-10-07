"""Codex Responses → chat.completions normalization for auxiliary callers.

Aux consumers speak Chat Completions, so a Responses result is mapped onto
``(text_parts, tool_calls, usage, finish_reason)`` through the main loop's shared
normalizer, keeping its phase and completion gates.
"""

from types import SimpleNamespace
from typing import Any, List, Optional, Tuple


def _parse_codex_final_response(
    final: Any, *, issuer_kind: Optional[str] = None, issuer_model: Optional[str] = None,
) -> Tuple[List[str], List[Any], Any, str]:
    """Normalize Responses output without losing phase or completion state for aux callers."""
    from agent.codex_responses_adapter import (
        _lower_or_none, _normalize_codex_response, _safe_response_output_text,
    )

    # The shared normalizer reads SDK-style items. Keep support for compatible hosts
    # returning dict items, and the aux adapter's legacy empty completed response.
    output = [
        SimpleNamespace(**item) if isinstance(item, dict) else item
        for item in (getattr(final, "output", None) or [])
    ]
    normalized_final = SimpleNamespace(
        output=output or [SimpleNamespace(type="message", content=[])],
        output_text=_safe_response_output_text(final),
        status=getattr(final, "status", None),
        incomplete_details=getattr(final, "incomplete_details", None),
        error=getattr(final, "error", None),
    )
    # Aux has no continuation to re-elicit a leaked tool call, so tool-call-shaped text stays content
    # and goes through the normalizer's own phase/completion gates (no clear, no WARNING).
    message, finish_reason = _normalize_codex_response(
        normalized_final, issuer_kind=issuer_kind, issuer_model=issuer_model, recover_leaked_tool_call=False,
    )
    # Aux consumers speak Chat Completions: "length" activates their existing
    # partial-summary rejection/fallback, whereas Codex's "incomplete" does not.
    # Any provider-incomplete response (token cap or content_filter) is a partial no aux consumer
    # may commit; completed tool calls stay "tool_calls" so dispatchers (e.g. MCP sampling) still run them.
    if finish_reason != "tool_calls" and (
        finish_reason == "incomplete" or _lower_or_none(normalized_final.status) == "incomplete"
    ):
        finish_reason = "length"
    text_parts = [message.content] if message.content else []
    tool_calls_raw = message.tool_calls
    usage = None
    resp_usage = getattr(final, "usage", None)
    if resp_usage:
        def _u(key: str) -> int:
            return getattr(resp_usage, key, 0) or (resp_usage.get(key, 0) if isinstance(resp_usage, dict) else 0)
        usage = SimpleNamespace(
            prompt_tokens=_u("input_tokens"), completion_tokens=_u("output_tokens"),
            total_tokens=_u("total_tokens"))
    return text_parts, tool_calls_raw, usage, finish_reason
