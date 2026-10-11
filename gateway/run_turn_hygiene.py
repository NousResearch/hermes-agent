"""Fail-closed hygiene helpers for GatewayTurnMixin.

Split out of ``gateway/run_turn.py`` (file-line ratchet): the in-context payload bound for
turns where session hygiene has not landed (#111988) and the no-commit reason naming for
hygiene compressions (#71097). Mixed into ``GatewayRunner`` through ``GatewayTurnMixin``.
"""

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


def hygiene_no_commit_reason(agent) -> str:
    """Name WHY a hygiene compression left the session id unchanged with no in-place commit.
    The terminal ``else`` used to blame "no session_db on the hygiene agent" for every route into it,
    but that is one of several causes (#71097): an attempt that ABORTED before any commit boundary
    (lock skip, transient cooldown, summary timeout, codex thread interrupted) leaves
    ``_last_compression_attempt_in_place`` at ``None``; a DB-less agent is only the case when
    ``_session_db`` really is missing. Read the per-attempt signals the compressor sets, in that order."""
    if not bool(getattr(agent, "_last_compression_attempt_recorded", False)):
        return "compression did not run"
    lock_skip = getattr(agent, "_compression_skipped_due_to_lock", None)
    if lock_skip is True or isinstance(lock_skip, str):
        return "attempt skipped: compression lease held by another process"
    blocked = getattr(agent, "_compression_blocked_transient", None)
    if blocked:
        return f"attempt blocked: {blocked}"
    if getattr(agent, "_last_compression_attempt_in_place", None) is None:
        detail = "summary timed out" if getattr(agent, "_last_compression_timed_out", False) else "aborted before commit"
        warning = getattr(agent, "_last_compression_summary_warning", None)
        return f"attempt {detail}" + (f": {warning}" if warning else "")
    if getattr(agent, "_session_db", None) is None:
        return "no session_db on the hygiene agent"
    return "in-place commit did not complete"