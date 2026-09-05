"""Shared NVIDIA NIM reasoning-family and effort policy."""

import re

_GLM5_MODEL_RE = re.compile(r"(^|/)glm[-_]?5", re.IGNORECASE)
_NEMOTRON_3_ULTRA_MODEL_RE = re.compile(
    r"(^|/)nemotron[-_]?3[-_]?ultra(?:[-_/]|$)", re.IGNORECASE
)
_STANDARD_THINKING_FAMILY_MARKERS = (
    "deepseek",
    "kimi",
    "moonshot",
    "qwen3",
    "qwen-3",
    "qwen_3",
)
_HERMES_TO_NIM_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
    "ultra": "high",
}


def is_glm5_nim_model(model: str | None) -> bool:
    return bool(model and _GLM5_MODEL_RE.search(model))


def is_nemotron_3_ultra_nim_model(model: str | None) -> bool:
    return bool(model and _NEMOTRON_3_ULTRA_MODEL_RE.search(model))


def is_nim_thinking_model(model: str | None) -> bool:
    normalized = (model or "").lower()
    return (
        is_glm5_nim_model(model)
        or is_nemotron_3_ultra_nim_model(model)
        or any(marker in normalized for marker in _STANDARD_THINKING_FAMILY_MARKERS)
    )


def normalize_nim_reasoning_effort(effort: object) -> tuple[str, str]:
    """Return ``(raw_effort, low|medium|high)`` for a Hermes effort."""
    raw = str(effort or "medium").strip().lower()
    return raw, _HERMES_TO_NIM_EFFORT.get(raw, "high")
