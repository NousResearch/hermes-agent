"""Fail-closed in-context payload bound for turns where hygiene has not landed.

Pure transcript math split out of the ``gateway/run_turn`` facade (god-file ratchet);
no gateway dependencies."""

from __future__ import annotations

from typing import Any

# Setup/prefix rows rather than conversation: the agent rebuilds its own system prompt, and a
# transcript meta row is logging-only — neither reaches the model, but both are the head a
# fail-closed payload keeps.
_HYGIENE_SETUP_ROLES = ("system", "session_meta")


def bound_model_input_without_hygiene(history: list[Any], limit: int) -> list[Any]:
    """Fail-closed in-context bound for a turn where hygiene has not landed (#111988).

    Keeps the leading ``system``/``session_meta`` setup rows plus the newest tail, total <= ``limit``.
    Deterministic (the same transcript always yields the same cut) and payload-only: the stored
    transcript is never touched, so the agent's durable-prefix slice (``history_offset``) is
    unaffected. Returns ``history`` unchanged — same object — when nothing needs dropping, so the
    landed-compression and below-the-limit paths stay byte-identical.
    """
    if len(history) <= limit:
        return history
    head_end = 0
    while (head_end < len(history) and isinstance(history[head_end], dict)
           and history[head_end].get("role") in _HYGIENE_SETUP_ROLES):
        head_end += 1
    # Always leave room for the newest row: a setup-only payload would answer nothing.
    head_end = min(head_end, limit - 1)
    tail_start = len(history) - (limit - head_end)
    # Never start the kept tail on a tool result: its parent assistant(tool_calls) row is dropped
    # with it, and an orphaned tool result is an invalid sequence for every provider.
    while (tail_start < len(history) and isinstance(history[tail_start], dict)
           and history[tail_start].get("role") == "tool"):
        tail_start += 1
    return history[:head_end] + history[tail_start:]
