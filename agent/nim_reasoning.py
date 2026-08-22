"""NVIDIA NIM thinking-model classification.

NIM exposes ordinary instruct models and models whose tool-call messages must
round-trip ``reasoning_content`` through the same OpenAI-compatible endpoint.
Keep that distinction model-driven so enabling reasoning on an instruct model
does not add a field that strict backends may reject.
"""

from __future__ import annotations


_EXPLICIT_THINKING_MARKERS = (
    "-thinking",
    "-reasoning",
    "/deepseek-r1",
    "/deepseek-v3",
    "/deepseek-v4",
)


def is_nim_thinking_model(model: str | None) -> bool:
    """Return whether *model* belongs to a NIM thinking-output family."""
    normalized = (model or "").strip().lower()
    if not normalized:
        return False
    if any(marker in normalized for marker in _EXPLICIT_THINKING_MARKERS):
        return True
    # Qwen3's thinking-capable variants use either an explicit ``thinking``
    # suffix or the base family name.  ``*-instruct`` variants are strict
    # non-thinking chat models and must not receive reasoning_content.
    return "/qwen3" in normalized and "-instruct" not in normalized
