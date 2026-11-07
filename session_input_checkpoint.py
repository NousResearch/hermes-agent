"""``checkpoint_input``: consume admitted input tickets at generic turn seams.

The durable half (admission row, retry alias, transfer settlement) belongs to
:mod:`session_input_primitives` and the gateway transfer module. This module only
moves already-ticketed text into the live transcript at the two seams upstream
already has — right before a tool batch runs, and when a model request is rebuilt
— and keeps the ticket bookkeeping that cancellation and ``/stop`` depend on.

Nothing here runs unless a provider bound the agent (``bind_agent``), so a
profile without the plugin keeps upstream behaviour byte for byte.
"""

from __future__ import annotations

import logging
import time

from session_input_primitives import format_part, policy_for

log = logging.getLogger("session_input_checkpoint")


def bind_agent(authority, agent) -> None:
    """Freeze this generation's policy snapshot on the agent (or clear it)."""
    agent._session_input_policy = policy_for(authority.db)


def bound(agent) -> bool:
    return getattr(agent, "_session_input_policy", None) is not None


def correction(agent, text: str, *, redirect: bool = False, arrived=None) -> str:
    """Format one ticket's text; called under the existing steer/redirect slot lock."""
    if not bound(agent):
        return text
    slot = "_pending_redirect" if redirect else "_pending_steer"
    counter = "_input_redirect_count" if redirect else "_input_steer_count"
    count = getattr(agent, counter, 0) + 1 if getattr(agent, slot, None) else 1
    setattr(agent, counter, count)
    if redirect:
        now = time.monotonic()
        if not getattr(agent, slot, None):
            agent._input_redirect_first = now
        agent._input_redirect_last = now
    log.info("input path=%s count=%d", "redirect_buffer" if redirect else "tool_checkpoint", count)
    return format_part(text, count=count, arrived=arrived)


def wait_redirect(agent) -> None:
    """The worker collects corrections while its cancelled request rebuilds.

    Never hold the slot lock while waiting: gateway callers must be able to
    append, and /stop must clear the buffer. Only bound agents use this.
    """
    policy = getattr(agent, "_session_input_policy", None)
    if policy is None:
        return
    while True:
        with agent._pending_redirect_lock:
            if not agent._pending_redirect:
                return
            first = getattr(agent, "_input_redirect_first", None)
            last = getattr(agent, "_input_redirect_last", None)
            if first is None or last is None:
                return
            delay = min(last + policy.quiet_seconds, first + policy.hard_cap_seconds) - time.monotonic()
            if delay <= 0:
                return
        time.sleep(min(delay, 0.025))


def redirect_part(agent, ticket, text: str) -> None:
    """Caller owns the redirect lock. Track the same text/ticket pairs as native steers."""
    parts = getattr(agent, "_input_redirect_parts", [])
    if not getattr(agent, "_pending_redirect", None):
        parts = []
    agent._input_redirect_parts = [*parts, (ticket, text)]


def clear_redirect_parts(agent, *, consumed: bool) -> None:
    fates = vars(agent).setdefault("_input_redirect_fates", {})
    for ticket, _ in vars(agent).pop("_input_redirect_parts", []):
        if ticket is not None:
            fates[ticket] = consumed


def withdraw_redirect(agent, ticket):
    """None = not a redirect ticket; otherwise the native ``withdraw_steer`` contract."""
    lock = getattr(agent, "_pending_redirect_lock", None)
    if lock is None:
        return None
    with lock:
        parts = getattr(agent, "_input_redirect_parts", [])
        kept = [part for part in parts if part[0] != ticket]
        if len(parts) != len(kept):
            agent._input_redirect_parts = kept
            agent._pending_redirect = "\n\n[Additional user correction]\n".join(text for _, text in kept) or None
            return True
        fates = vars(agent).get("_input_redirect_fates", {})
        if ticket in fates:
            return not fates.pop(ticket)
    return None


def keep_redirect_parts(agent, *, hard_cancel: bool) -> None:
    """Caller owns the redirect lock. Preserve ticket ownership on a turn-boundary clear."""
    from agent.interrupt_control import _ic_lock, _steer_parts, _set_steer_parts
    if hard_cancel:
        clear_redirect_parts(agent, consumed=False)
        return
    text = getattr(agent, "_pending_redirect", None)
    if not text or not bound(agent):
        return
    parts = vars(agent).pop("_input_redirect_parts", []) or [(None, text)]
    with _ic_lock(agent, "_pending_steer_lock"):
        _set_steer_parts(agent, _steer_parts(agent) + parts)
    agent._turn_user_intervened = True


def checkpoint_input(agent, messages, *, stage: str, text: str | None = None) -> bool:
    """Append pending input at a turn seam; True when the caller must replan.

    ``stage="before_tools"`` drains the tool checkpoint's steer buffer before any
    tool side effect. ``stage="before_request"`` appends text already drained when
    the model request is rebuilt, keeping the cached transcript prefix intact.
    """
    if not bound(agent):
        return False
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
    log.info("input path=%s checkpoint", stage)
    return True
