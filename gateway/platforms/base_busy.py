"""Busy-path helpers for ``BasePlatformAdapter._handle_message_while_active``.

A message that arrives while its session is busy is diverted by the adapter before it reaches the
runner's ``_handle_message``. These helpers keep that diversion consistent with the idle path.

Imports nothing from ``gateway.platforms.base``, which imports this module.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("gateway.platforms.base")


async def admit_busy_arrival(adapter: Any, event: Any) -> Optional[Any]:
    """Run the runner's ``pre_gateway_dispatch`` handler on a busy arrival; ``None`` = drop.

    Same order as the idle path (``_hm_admit_event``): the hook runs before bypass commands,
    approvals, clarify replies, steering, interrupts and queueing, so ``allow``/``None`` keeps
    ``/stop`` and ``/approve`` working mid-turn while ``skip``/``rewrite`` apply to them by
    design. The runner marks accepted events so a later cold-path drain does not run it twice.
    ``getattr``: lightweight adapter doubles in tests skip ``__init__``.
    """
    handler = getattr(adapter, "_pre_gateway_dispatch_handler", None)
    if event.internal or handler is None:
        return event
    try:
        return await handler(event)
    except Exception as e:
        logger.error("[%s] Pre-gateway dispatch failed: %s", adapter.name, e, exc_info=True)
        return None


def has_pending_text_clarify(session_key: str) -> bool:
    """Whether the busy session's agent is blocked on a clarify prompt a plain-text reply answers.

    While blocked on clarify_tool the next message must reach the text-intercept so
    numeric/exact/"Other" answers resolve it and unblock the agent; otherwise it lands in
    ``_pending_messages`` as a follow-up turn and the answer is discarded. Same shape as the
    /approve deadlock fix (PR #4926): agent thread blocked on Event.wait, message must reach the
    resolver before being a new turn.
    """
    try:
        from tools import clarify_gateway as _clarify_mod
        return _clarify_mod.get_pending_for_session(
            session_key, include_choice_prompts=True) is not None
    except Exception:
        logger.debug("Clarify probe failed for %s; treating as no pending clarify", session_key,
                     exc_info=True)
        return False
