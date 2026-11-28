"""Transport liveness for streaming calls: raw bytes on the wire, SSE comments included.

The OpenAI SDK's SSE decoder drops comment lines (``: keepalive``) before they become
events, so a proxy that keeps a thinking model's connection alive with them looks
silent to every chunk-counting watchdog (#151280). Watching the response's raw byte
iterator sees them.
"""

from __future__ import annotations

from typing import Any, Callable

# Patience while bytes keep arriving but no output does: this many stale windows since the
# last real chunk (180s base -> 12 min; 240s -> 16 min). Bounded, so a gateway that only
# ever pings is still cut; a fully silent socket keeps the base window.
KEEPALIVE_PATIENCE_FACTOR = 4.0


def watch_raw_bytes(raw_stream: Any, on_bytes: Callable[[], None]) -> None:
    """Call ``on_bytes`` for every raw chunk the stream's httpx response reads.

    The SDK consumes the body via ``response.iter_bytes()`` -> ``self.iter_raw()``, so an
    instance-level wrapper of ``iter_raw`` sees every byte the socket delivers and passes
    the chunks through unchanged. Streams without an httpx response (test fakes, Relay
    replays) are left alone.
    """
    response: Any = getattr(raw_stream, "response", None)
    original: Any = getattr(response, "iter_raw", None)
    if not callable(original) or getattr(response, "_hermes_wire_watched", False):
        return

    def iter_raw_with_activity(chunk_size=None):
        for chunk in original(chunk_size=chunk_size):
            on_bytes()
            yield chunk

    response.iter_raw = iter_raw_with_activity
    response._hermes_wire_watched = True


# Delta fields that carry model output (the predicate from #34541, plus refusal/details).
_PROGRESS_FIELDS = ("content", "reasoning_content", "reasoning", "reasoning_details",
                    "tool_calls", "function_call", "refusal")


def chat_chunk_has_progress(chunk: Any) -> bool:
    """A chat-completions chunk carrying output, a finish reason or usage. Anything else
    (empty ``choices``, a role-only delta) is a keepalive: transport liveness only."""
    if getattr(chunk, "usage", None):
        return True
    for choice in getattr(chunk, "choices", None) or ():
        delta = getattr(choice, "delta", None)
        if getattr(choice, "finish_reason", None) or any(getattr(delta, f, None) for f in _PROGRESS_FIELDS):
            return True
        extra = getattr(delta, "model_extra", None)  # provider-specific output fields
        if isinstance(extra, dict) and any(extra.values()):
            return True
    return False
