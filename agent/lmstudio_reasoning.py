"""LM Studio reasoning-effort resolution (chat-completions transport + run_agent's
iteration-limit summary path). LM Studio publishes per-model
``capabilities.reasoning.allowed_options`` (``["off","on"]`` for toggle models,
``["off","minimal","low"]`` for graduated ones); the user's ``reasoning_config`` is
mapped onto LM Studio's vocabulary, then clamped to the allowed set so the server
doesn't 400."""

from __future__ import annotations

from typing import List, Optional

_LM_VALID_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}

# Toggle vocabulary → request vocabulary; also applied to published allowed_options.
_LM_EFFORT_ALIASES = {"off": "none", "on": "medium"}

# Hermes' ladder grew past LM Studio's vocabulary ("max", "ultra"); without this
# ceiling clamp they'd fall to the "medium" default (more yields less than "xhigh").
# Separate from _LM_EFFORT_ALIASES, which must not rewrite allowed_options.
_LM_EFFORT_CLAMP = {"max": "xhigh", "ultra": "xhigh"}


def resolve_lmstudio_effort(reasoning_config: Optional[dict], allowed_options: Optional[List[str]]) -> Optional[str]:
    """Return the ``reasoning_effort`` to send to LM Studio, or ``None`` = omit the
    field (the user picked a level the model can't honor, so LM Studio falls back
    to the model's declared default rather than a silently substituted effort).
    Falsy ``allowed_options`` (probe failed) skips clamping."""
    effort = "medium"
    if reasoning_config and isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is False:
            effort = "none"
        else:
            raw = (reasoning_config.get("effort") or "").strip().lower()
            raw = _LM_EFFORT_ALIASES.get(raw, raw)
            raw = _LM_EFFORT_CLAMP.get(raw, raw)
            if raw in _LM_VALID_EFFORTS:
                effort = raw
    if allowed_options and effort not in {_LM_EFFORT_ALIASES.get(opt, opt) for opt in allowed_options}:
        return None
    return effort
