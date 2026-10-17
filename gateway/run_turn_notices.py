"""Reply-notice copy helpers for ``gateway/run_turn.py``.

Moved out of the turn facade verbatim to hold its file-line budget: ``run_turn.py`` is over
target and may only go down. Both helpers build user-facing notice text — a reply for a model
that answered with only a silence marker, and a one-line quote of a ``/bg`` prompt for its
failure notice.
"""

from __future__ import annotations

from agent.i18n import t


def _unexpected_silence_reply() -> str:
    """Reply when the model returned only a silence marker for a message that needed an answer."""
    return t("gateway.errors.unexpected_silence")


def _bg_prompt_preview(prompt: str, limit: int = 60) -> str:
    """Short single-line quote of a /bg prompt for its failure notice (the task id means nothing to the user)."""
    text = " ".join(str(prompt or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
