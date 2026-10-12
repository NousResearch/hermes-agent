"""User-facing reply snippets a gateway turn assembles from i18n or raw prompt text.

Self-contained formatters with no runner state, split out of ``run_turn.py`` so the facade keeps
its FILE_LINES ratchet (moved code keeps its cap).
"""

from __future__ import annotations

from agent.i18n import t


def unexpected_silence_reply() -> str:
    """Reply when the model returned only a silence marker for a message that needed an answer."""
    return t("gateway.errors.unexpected_silence")


def bg_prompt_preview(prompt: str, limit: int = 60) -> str:
    """Short single-line quote of a /bg prompt for its failure notice (the task id means nothing to the user)."""
    text = " ".join(str(prompt or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
