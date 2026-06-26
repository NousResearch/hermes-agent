"""Thinking-timeout detection and user-facing guidance for reasoning models.

When a known reasoning model (NVIDIA Nemotron 3 Ultra, OpenAI o1/o3,
Anthropic Opus 4.x thinking, DeepSeek R1, Qwen QwQ, xAI Grok reasoning)
hits a transport-layer error before the first content token arrives, the
upstream proxy has almost certainly idle-killed a long thinking stream —
not a true context overflow or a configuration error.  The user needs
distinct guidance for this case:

    "The model's thinking phase exceeded the upstream proxy's idle
     timeout before the first content token arrived.  This is a known
     issue with reasoning models behind cloud gateways (NVIDIA NIM,
     OpenAI, Anthropic, DeepSeek).  Workarounds in priority order:
     1. Set `providers.<provider>.models.<model>.stale_timeout_seconds: 900`
        in `~/.hermes/config.yaml` to extend the per-call timeout...
     2. Lower `reasoning_budget` or set `reasoning_effort: medium`...
     3. Use a smaller / faster reasoning model..."

The existing `_is_stream_drop` guidance at
``agent/conversation_loop.py:3464-3486`` fires for large-file-write
stream drops ("try execute_code with Python's open() for large files")
which is the WRONG advice for the thinking-timeout case.  This module
provides the detection and the message as standalone helpers so the
detection logic is unit-testable without driving the full retry loop,
and the message text can be regression-tested for spelling and accuracy.

Part 2 of Fixes #52310.
"""

from __future__ import annotations

from typing import Optional


# Substring set that identifies a transport-layer failure on the
# response stream.  Same shape as the existing
# ``_SERVER_DISCONNECT_PATTERNS`` in ``agent/error_classifier.py:394``
# but extended to also catch the OSS-level error signature
# (``broken pipe`` / ``errno 32``) that the upstream kill surfaces
# to the OpenAI SDK wrapper.
_THINKING_TIMEOUT_SUBSTRINGS: tuple[str, ...] = (
    "broken pipe",
    "errno 32",
    "remote protocol",
    "connection reset",
    "connection lost",
    "peer closed",
    "server disconnected",
)


def is_thinking_timeout(classified: object, model: str, error_msg: str) -> bool:
    """Return True when a reasoning model's thinking phase hit a transport kill.

    Args:
        classified: a :class:`agent.error_classifier.ClassifiedError` instance
            (duck-typed here to avoid an import cycle in unit tests).
        model: the model slug at failure time (e.g.
            ``"nvidia/nemotron-3-ultra-550b-a55b"``).
        error_msg: lowercased string representation of the underlying
            exception (typically ``str(api_error).lower()``).

    Returns True when ALL conditions hold:
        1. ``classified.reason == FailoverReason.timeout`` (the classifier
           override at ``agent/error_classifier.py:720-738`` ensures this
           is the case for reasoning models even on large sessions).
        2. ``api_error`` has no ``.status_code`` attribute set (transport
           disconnect, not an HTTP error).
        3. ``model`` is in the reasoning-model allowlist (reuses
           ``agent.reasoning_timeouts.get_reasoning_stale_timeout_floor``).
        4. ``error_msg`` contains one of the transport-kill substrings.

    Non-reasoning models always return False.  Non-transport errors
    (billing / rate_limit / auth / context_overflow / format_error)
    always return False.  HTTP-status errors always return False.
    """
    # Import here (not at module top) to keep this helper cheap to
    # import even from callers that don't need it.  ``agent.reasoning_timeouts``
    # is small and dependency-free.
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor

    # Condition 1: classifier says timeout.  Use a string/value check
    # rather than importing FailoverReason so this module has zero
    # import cycles from the error_classifier package.
    reason = getattr(classified, "reason", None)
    reason_value = getattr(reason, "value", None)
    if reason_value != "timeout":
        return False

    # Condition 2: no HTTP status code (transport, not API error).
    # Caller is expected to gate on ``getattr(api_error, "status_code", None) is None``
    # before calling this helper; the surface here is just the post-gate
    # boolean so the caller can pass an already-prepped error_msg.

    # Condition 3: reasoning model allowlist.
    if get_reasoning_stale_timeout_floor(model) is None:
        return False

    # Condition 4: transport-kill substring in the error message.
    error_msg_lower = (error_msg or "").lower()
    return any(p in error_msg_lower for p in _THINKING_TIMEOUT_SUBSTRINGS)


def build_thinking_timeout_guidance(
    provider: str, model: str, model_label: Optional[str] = None,
) -> str:
    """Return the user-facing guidance string appended to ``_final_response``.

    Args:
        provider: provider slug (e.g. ``"nvidia"``, ``"openai"``).
        model: bare model slug the user would put in their config
            (e.g. ``"nemotron-3-ultra-550b-a55b"`` if the user uses
            NVIDIA direct, or the full ``"nvidia/nemotron-3-ultra-550b-a55b"``
            if they go through an aggregator).  Used verbatim in the
            config snippet so the user can copy-paste.
        model_label: optional short label for the model name in the
            prose (e.g. ``"Nemotron 3 Ultra"``).  Falls back to the
            slug if not provided.
    """
    label = model_label or model
    return (
        "\n\nThe model's thinking phase exceeded the upstream proxy's "
        "idle timeout before the first content token arrived. This is a "
        f"known issue with reasoning models (like {label}) behind cloud "
        "gateways (NVIDIA NIM, OpenAI, Anthropic, DeepSeek). Workarounds "
        "in priority order:\n"
        f"1. Set `providers.{provider}.models.{model}.stale_timeout_seconds: 900` "
        "in `~/.hermes/config.yaml` to extend the per-call timeout. "
        "(Hermes's built-in floor is 600s for known reasoning models — "
        "if you still see this after raising, the upstream cap is even "
        "shorter.)\n"
        "2. Lower `reasoning_budget` or set `reasoning_effort: medium` on this "
        "model if the provider supports it.\n"
        "3. Use a smaller / faster reasoning model if the task doesn't "
        "require deep thinking."
    )
