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

from hermes_constants import VALID_REASONING_EFFORTS, project_reasoning_effort

# Toggle-style models publish allowed_options as ["off","on"] in /api/v1/models.
# Map them onto the OpenAI-compatible request vocabulary.
_LM_EFFORT_ALIASES = {"off": "none", "on": "medium"}


def resolve_lmstudio_effort(
    reasoning_config: Optional[dict],
    allowed_options: Optional[List[str]],
) -> Optional[str]:
    """Return the ``reasoning_effort`` string to send to LM Studio, or ``None``.

    ``None`` means "omit the field": no advertised non-empty effort exists at
    or below the request. When ``allowed_options`` is falsy (probe failed), skip
    projection and send the resolved effort anyway.
    """
    effort = "medium"
    if reasoning_config and isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is False:
            effort = "none"
        else:
            raw = (reasoning_config.get("effort") or "").strip().lower()
            raw = _LM_EFFORT_ALIASES.get(raw, raw)
            if raw == "ultra":
                raw = "max"
            if raw in VALID_REASONING_EFFORTS:
                effort = raw
    if allowed_options:
        allowed = {
            _LM_EFFORT_ALIASES.get(normalized, normalized)
            for option in allowed_options
            if (normalized := str(option or "").strip().lower())
        }
        if effort == "none":
            return effort if effort in allowed else None
        return project_reasoning_effort(effort, allowed)
    return effort
