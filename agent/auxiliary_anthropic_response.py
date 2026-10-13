"""Project native Anthropic responses for auxiliary Chat Completions callers."""

from types import SimpleNamespace
from typing import Any


def _normalize_anthropic_auxiliary_response(
    response: Any, *, is_oauth: bool
) -> SimpleNamespace:
    from agent.native_usage import NativeUsageCounts
    from agent.transports import get_transport

    normalized = get_transport("anthropic_messages").normalize_response(response, strip_tool_prefix=is_oauth)
    usage = None
    if (native_usage := getattr(response, "usage", None)) is not None:
        counts = NativeUsageCounts.read(native_usage, kind="anthropic")
        if counts.input is not None and counts.output is not None:
            prompt = counts.input + counts.cached + counts.cache_write
            usage = SimpleNamespace(
                prompt_tokens=prompt,
                completion_tokens=counts.output,
                total_tokens=prompt + counts.output,
                cache_read_input_tokens=counts.cached,
                cache_creation_input_tokens=counts.cache_write,
            )
    choice = SimpleNamespace(
        index=0, finish_reason=normalized.finish_reason,
        message=SimpleNamespace(
            content=normalized.content, tool_calls=normalized.tool_calls, reasoning=normalized.reasoning,
            reasoning_details=getattr(normalized, "reasoning_details", None),
            anthropic_content_blocks=getattr(normalized, "anthropic_content_blocks", None),
        ),
    )
    return SimpleNamespace(
        id=getattr(response, "id", None), model=getattr(response, "model", None),
        choices=[choice], usage=usage,
    )
