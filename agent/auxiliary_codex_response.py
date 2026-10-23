"""Codex Responses → chat.completions normalization for auxiliary callers.

Aux consumers speak Chat Completions, so a Responses result is mapped onto
``(text_parts, tool_calls, usage, finish_reason)`` through the main loop's shared
normalizer, keeping its phase and completion gates.
"""

from types import SimpleNamespace
from typing import Any, List, Optional, Tuple


def _parse_codex_final_response(
    final: Any, *, issuer_kind: Optional[str] = None, issuer_model: Optional[str] = None,
) -> tuple[list[str], list[Any], Any, str]:
    """Normalize Responses output without losing phase or completion state for aux callers."""
    from agent.codex_responses_adapter import _lower_or_none, _normalize_codex_response

    # The shared normalizer reads SDK-style items. Keep support for compatible hosts
    # returning dict items, and the aux adapter's legacy empty completed response.
    output = [
        SimpleNamespace(**item) if isinstance(item, dict) else item
        for item in (getattr(final, "output", None) or [])
    ]
    normalized_final = SimpleNamespace(
        output=output or [SimpleNamespace(type="message", content=[])],
        output_text=getattr(final, "output_text", None),
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
        # Aux usage accounting calls normalize_usage without api_mode (Chat shape) while aux
        # hooks pass the route's codex_responses shape, so mirror the Responses usage under
        # BOTH field-name families instead of dropping everything past the three totals —
        # cached/reasoning tokens and the raw usage a provider get_usage_cost reads (#135657).
        from agent.usage_pricing import _usage_field
        cached = _usage_field(resp_usage, "input_tokens_details", "cached_tokens")
        cache_write = _usage_field(resp_usage, "input_tokens_details", "cache_write_tokens")
        cache_creation = _usage_field(resp_usage, "input_tokens_details", "cache_creation_tokens")
        reasoning = _usage_field(resp_usage, "output_tokens_details", "reasoning_tokens")

        def _dump() -> dict:
            if isinstance(resp_usage, dict):
                return dict(resp_usage)
            dumper = getattr(resp_usage, "model_dump", None)
            if callable(dumper):
                return dict(dumper())
            return dict(getattr(resp_usage, "__dict__", None) or {})

        usage = SimpleNamespace(
            prompt_tokens=_u("input_tokens"), completion_tokens=_u("output_tokens"),
            total_tokens=_u("total_tokens"),
            input_tokens=_u("input_tokens"), output_tokens=_u("output_tokens"),
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=cached, cache_write_tokens=cache_write,
                cache_creation_input_tokens=cache_creation),
            input_tokens_details=SimpleNamespace(
                cached_tokens=cached, cache_write_tokens=cache_write,
                cache_creation_tokens=cache_creation),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
            model_dump=_dump,
        )
    return text_parts, tool_calls_raw, usage, finish_reason

