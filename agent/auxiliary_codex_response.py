"""Codex Responses → chat.completions normalization for auxiliary callers.

Aux consumers speak Chat Completions, so a Responses result is mapped onto
Chat Completions through the main loop's shared normaliser, with the same
phase and completion gates.
"""

from types import SimpleNamespace
from typing import Any, Optional


def _parse_codex_final_response(
    final: Any, *, issuer_kind: Optional[str] = None, issuer_model: Optional[str] = None,
) -> SimpleNamespace:
    """Normalize Responses output without losing phase or completion state for aux callers."""
    from agent.codex_responses_adapter import _field, _lower_or_none, _normalize_codex_response

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
    refusals = [
        _field(part, "refusal")
        for item in output if _field(item, "type") == "message"
        and _lower_or_none(_field(item, "phase")) not in {"commentary", "analysis"}
        for part in (_field(item, "content") or [])
        if _field(part, "type") == "refusal" and isinstance(_field(part, "refusal"), str)
    ]
    message.refusal = "\n".join(refusals) if refusals else None
    message.role = "assistant"
    message.content = message.content.strip() or None
    message.tool_calls = message.tool_calls or None
    usage = None
    resp_usage = getattr(final, "usage", None)
    if resp_usage:
        usage = SimpleNamespace(
            prompt_tokens=_field(resp_usage, "input_tokens") or 0,
            completion_tokens=_field(resp_usage, "output_tokens") or 0,
            total_tokens=_field(resp_usage, "total_tokens") or 0,
        )
        if (details := _field(resp_usage, "input_tokens_details")) is not None:
            usage.prompt_tokens_details = SimpleNamespace(cached_tokens=_field(details, "cached_tokens") or 0)
            cache_write = _field(details, "cache_write_tokens")
            if cache_write is None:
                cache_write = _field(details, "cache_creation_tokens")
            if cache_write is not None:
                usage.prompt_tokens_details.cache_write_tokens = cache_write
        if (details := _field(resp_usage, "output_tokens_details")) is not None:
            usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=_field(details, "reasoning_tokens") or 0)
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        id=getattr(final, "id", None), model=getattr(final, "model", None),
        choices=[choice], usage=usage,
    )

