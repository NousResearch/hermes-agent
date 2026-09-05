"""Inference progress on the Chat Completions and Anthropic Messages wires."""


def _has_fields(value, *fields) -> bool:
    return any(getattr(value, field, None) for field in fields)


def chunk_has_progress(chunk) -> bool:
    """Ignore empty/role-only deltas and pings, but retain reasoning and tools.

    Protocol boundaries are progress too: a tool can start without arguments,
    and a finish chunk can carry no text. Raw transport traffic alone is not
    evidence that a provider is still generating a response.
    """
    choices = getattr(chunk, "choices", None)
    if choices is not None:
        for choice in choices:
            if getattr(choice, "finish_reason", None):
                return True
            delta = getattr(choice, "delta", None)
            if _has_fields(delta, "content", "reasoning_content", "reasoning", "reasoning_details", "refusal", "audio"):
                return True
            for tool in getattr(delta, "tool_calls", None) or ():
                if getattr(tool, "id", None) or _has_fields(getattr(tool, "function", None), "name", "arguments"):
                    return True
            if _has_fields(getattr(delta, "function_call", None), "name", "arguments"):
                return True
        return False

    # Anthropic's raw stream and its synthesized SDK delta events share this
    # watchdog. Ping events and empty deltas must not extend its deadline.
    if getattr(chunk, "type", None) in {
        "message_start", "message_stop", "content_block_start", "content_block_stop",
    }:
        return True
    return _has_fields(chunk, "text", "thinking", "partial_json", "signature") or _has_fields(
        getattr(chunk, "delta", None), "text", "thinking", "partial_json", "signature", "stop_reason",
    )
