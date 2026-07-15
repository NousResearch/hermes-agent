"""LM Studio reasoning-effort resolution shared by the chat-completions
transport and run_agent's iteration-limit summary path.

LM Studio publishes per-model ``capabilities.reasoning.allowed_options`` (e.g.
``["off","on"]`` for toggle-style models, ``["off","minimal","low"]`` for
graduated models). We map the user's ``reasoning_config`` onto LM Studio's
OpenAI-compatible vocabulary, then clamp against the model's allowed set so
the server doesn't 400 on an unsupported effort.
"""

from __future__ import annotations

from typing import List, Optional

from hermes_constants import project_reasoning_effort

# LM Studio accepts these top-level reasoning_effort values via its
# OpenAI-compatible chat.completions endpoint.
_LM_VALID_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}

# Toggle-style models publish allowed_options as ["off","on"] in /api/v1/models.
# Map them onto the OpenAI-compatible request vocabulary.
_LM_EFFORT_ALIASES = {"off": "none", "on": "medium"}


def resolve_lmstudio_effort(
    reasoning_config: Optional[dict],
    allowed_options: Optional[List[str]],
) -> Optional[str]:
    """Return the ``reasoning_effort`` string to send to LM Studio, or ``None``.

    When options are advertised, ``max`` projects to the strongest lower
    supported tier. Other unsupported efforts retain the existing omission
    behavior. When ``allowed_options`` is falsy (probe failed), preserve the
    legacy fallback instead of sending an unverified ``max`` value.
    """
    effort = "medium"
    if reasoning_config and isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is False:
            effort = "none"
        else:
            raw = (reasoning_config.get("effort") or "").strip().lower()
            raw = _LM_EFFORT_ALIASES.get(raw, raw)
            if raw == "max" or raw in _LM_VALID_EFFORTS:
                effort = raw
    if allowed_options:
        allowed = set()
        for option in allowed_options:
            normalized_option = str(option).strip().lower()
            allowed.add(_LM_EFFORT_ALIASES.get(normalized_option, normalized_option))
        if effort == "none":
            return effort if effort in allowed else None
        if effort in allowed:
            return effort
        if effort == "max":
            return project_reasoning_effort(effort, allowed)
        return None
    return "medium" if effort == "max" else effort
