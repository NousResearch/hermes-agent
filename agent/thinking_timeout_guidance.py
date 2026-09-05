"""Thinking-timeout detection and user-facing guidance for reasoning models.

A known reasoning model hitting a transport-layer error before the first content token was
almost certainly idle-killed mid-think by the upstream proxy — not a context overflow — so
the generic stream-drop guidance in conversation_loop is wrong for that case.
"""

from __future__ import annotations

from typing import Optional


# Transport-layer failure signatures: the classifier's server-disconnect set plus the OS-level
# ``broken pipe`` / ``errno 32`` the upstream kill surfaces through the OpenAI SDK wrapper.
_THINKING_TIMEOUT_SUBSTRINGS: tuple[str, ...] = (
    "broken pipe", "errno 32", "remote protocol", "connection reset", "connection lost",
    "peer closed", "server disconnected",
)


def is_thinking_timeout(classified: object, model: str, error_msg: str) -> bool:
    """True when a reasoning model's thinking phase hit a transport kill.

    All must hold: ``classified.reason`` is the ``timeout`` FailoverReason (duck-typed via
    ``.value`` to avoid importing error_classifier), ``model`` is in the reasoning allowlist,
    and ``error_msg`` carries a transport-kill substring. The caller gates on the error having
    no HTTP ``status_code``.
    """
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor

    if getattr(getattr(classified, "reason", None), "value", None) != "timeout":
        return False
    if get_reasoning_stale_timeout_floor(model) is None:
        return False
    return any(p in (error_msg or "").lower() for p in _THINKING_TIMEOUT_SUBSTRINGS)


def build_thinking_timeout_guidance(provider: str, model: str, model_label: Optional[str] = None) -> str:
    """User-facing guidance appended to the final response. ``model`` is used verbatim in
    the config snippet so it is copy-pasteable; ``model_label`` is the optional prose name."""
    label = model_label or model
    return (
        "\n\nThe model's thinking phase exceeded the upstream proxy's idle timeout before the first content token "
        "arrived. This is a "
        f"known issue with reasoning models (like {label}) behind cloud "
        "gateways (NVIDIA NIM, OpenAI, Anthropic, DeepSeek). Workarounds in priority order:\n"
        f"1. Set `providers.{provider}.models.{model}.stale_timeout_seconds: 900` "
        "in `~/.hermes/config.yaml` to extend the per-call timeout. (Hermes's built-in floor is 600s for known "
        "reasoning models — if you still see this after raising, the upstream cap is even shorter.)\n2. Lower "
        "`reasoning_budget` or set `reasoning_effort: medium` on this model if the provider supports it.\n3. Use a "
        "smaller / faster reasoning model if the task doesn't require deep thinking."
    )
