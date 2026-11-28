"""Ticketed mid-turn corrections: ``withdraw_steer`` must settle redirects too.

A correction collected while a model request is being rebuilt lives in the
redirect slot, not the steer slot. Without ticket bookkeeping there, a gateway
hand-over that withdraws its admission after the turn ended cannot tell whether
the text reached the model, so it either loses the text or delivers it twice.
This module keeps the same ``(ticket, text)`` pairs for redirects that
``steer()`` keeps, and provides the two checkpoint appends used by the turn
loop. It holds no policy: windows and formatting are a caller's business.
"""

from __future__ import annotations

import logging

log = logging.getLogger("agent.correction_tickets")


def redirect_part(agent, ticket, text: str) -> None:
    """Caller owns the redirect lock. Track the same text/ticket pairs as native steers."""
    parts = getattr(agent, "_redirect_ticket_parts", [])
    if not getattr(agent, "_pending_redirect", None):
        parts = []
    agent._redirect_ticket_parts = [*parts, (ticket, text)]


def clear_redirect_parts(agent, *, consumed: bool) -> None:
    """Caller owns the redirect lock. ``consumed`` is each ticket's fate."""
    fates = vars(agent).setdefault("_redirect_ticket_fates", {})
    for ticket, _ in vars(agent).pop("_redirect_ticket_parts", []):
        if ticket is not None:
            fates[ticket] = consumed


def withdraw_redirect(agent, ticket):
    """None = not a redirect ticket; otherwise the native ``withdraw_steer`` contract."""
    lock = getattr(agent, "_pending_redirect_lock", None)
    if lock is None:
        return None
    with lock:
        parts = getattr(agent, "_redirect_ticket_parts", [])
        kept = [part for part in parts if part[0] != ticket]
        if len(parts) != len(kept):
            agent._redirect_ticket_parts = kept
            agent._pending_redirect = "\n\n[Additional user correction]\n".join(text for _, text in kept) or None
            return True
        fates = vars(agent).get("_redirect_ticket_fates", {})
        if ticket in fates:
            return not fates.pop(ticket)
    return None


def keep_redirect_parts(agent, *, hard_cancel: bool) -> None:
    """Caller owns the redirect lock: a turn-boundary clear must not drop collected text.

    Hand the ticketed parts to the steer slot, whose finalizer path turns an
    unconsumed ticket into the next user turn instead of silently losing it.
    ``/stop`` (``hard_cancel``) still drops everything it was told to drop.
    """
    from agent.interrupt_control import _ic_lock, _steer_parts, _set_steer_parts
    if hard_cancel:
        clear_redirect_parts(agent, consumed=False)
        return
    text = getattr(agent, "_pending_redirect", None)
    if not text:
        return
    parts = vars(agent).pop("_redirect_ticket_parts", []) or [(None, text)]
    with _ic_lock(agent, "_pending_steer_lock"):
        _set_steer_parts(agent, _steer_parts(agent) + parts)
    agent._turn_user_intervened = True


def checkpoint_input(agent, messages, *, stage: str, text: str | None = None) -> bool:
    """Append pending input at a turn seam; True when the caller must replan.

    ``stage="before_tools"`` drains the tool checkpoint's steer buffer before any
    tool side effect. ``stage="before_request"`` appends text already drained when
    the model request is rebuilt, keeping the cached transcript prefix intact.
    """
    if stage == "before_tools":
        if not getattr(agent, "_pending_steer", None):
            return False
        text = agent._drain_pending_steer()
        if not text:
            return False
    elif stage != "before_request" or not text:
        return False
    if messages and messages[-1].get("role") == "tool":
        from agent.message_metadata import append_message
        from agent.prompt_builder import steer_user_row
        append_message(messages, steer_user_row(text))
    else:
        from agent.conversation_loop import _apply_active_turn_redirect
        _apply_active_turn_redirect(agent, messages, text)
    log.info("correction checkpoint stage=%s", stage)
    return True
