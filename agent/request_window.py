"""Request-fit checks that account for the model's output budget."""

from __future__ import annotations

from typing import Any, Optional


# Reserve a few thousand tokens for the answer, while keeping small model windows usable.
RESERVED_OUTPUT_TOKENS = 4_096


def request_exceeds_model_window(agent: Any, request_tokens: Any) -> Optional[bool]:
    """Whether a request leaves too little context-window space for a useful answer.

    ``None`` when either side is unknown (no compressor / unresolvable window / no estimate), so
    callers keep their conservative default instead of treating "unknown" as "fits".
    """
    window = getattr(getattr(agent, "context_compressor", None), "context_length", None)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        return None
    if isinstance(request_tokens, bool) or not isinstance(request_tokens, int):
        return None
    output_runway = min(RESERVED_OUTPUT_TOKENS, window // 10)
    return request_tokens > window - output_runway
