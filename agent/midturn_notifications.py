"""Mid-turn delivery of background notifications onto the tool-result tail.

Split out of ``agent/agent_runtime_helpers.py`` (which is over its FILE_LINES cap). The feature is
the background-completion counterpart of ``/steer``: a process that finishes while a turn is running
reaches the model at the turn's next tool boundary instead of waiting for the next user turn.

The notice becomes its own ``role:"user"`` row carrying the ``[INTERNAL NOTIFICATION ...]`` marker —
**never** the steer marker, because that one grants the text user authority and a process's stdout
must not speak as the user. Same role-alternation-safe slot as a steer (after the newest tool
result), so the two are siblings rather than duplicates; see ``apply_pending_steer_to_tool_results``
in the origin module for the shape this mirrors.
"""

from __future__ import annotations

import logging

from agent.prompt_builder import STEER_DISPLAY_KIND, notification_user_row

logger = logging.getLogger(__name__)


def _requeue_pending(agent, pending) -> None:
    """Hand notices back to the agent's queue when this boundary cannot take them."""
    queue = getattr(agent, "notify_between_tool_calls", None)
    if not callable(queue):
        return
    for text in pending:
        queue(text)


def apply_pending_notifications_to_tool_results(agent, messages: list, num_tool_msgs: int) -> None:
    """Deliver queued background notifications at this turn's next break.

    Called at the same point as ``apply_pending_steer_to_tool_results`` (end of a tool-call batch,
    before the next API call). Each queued notice becomes its own ``role:"user"`` row after the
    last tool result — the one role-alternation-safe mid-turn slot — carrying the notification
    marker rather than the user steer marker. A process that finishes mid-turn therefore reaches
    the model at the next tool boundary instead of waiting for the next user turn.
    """
    if num_tool_msgs <= 0 or not messages:
        return
    from typing import Any, cast

    drain = getattr(agent, "_drain_pending_notifications", None)
    if not callable(drain):
        return
    pending = [str(t) for t in cast(Any, drain)()]
    if not pending:
        return
    tail = range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1)
    target = next((messages[j] for j in tail
                   if isinstance(messages[j], dict) and messages[j].get("role") == "tool"), None)
    if target is None:
        # No tool result in this batch: requeue so the notice still lands at the next boundary
        # rather than being lost with a batch that produced nothing to anchor against.
        _requeue_pending(agent, pending)
        return
    # A /steer appended on this same batch owns the slot: two adjacent role:"user" rows are exactly
    # what repair_message_sequence refuses to weld (a persisted steer row is never merged), and on
    # the wire the notice's frame would then ride inside a row that OPENS with the user-authority
    # steer marker — machine output speaking as the user. Requeue instead: the next boundary takes
    # it, or the finalizer hands it back as its own turn.
    last = messages[-1] if messages else None
    if (isinstance(last, dict) and last.get("role") == "user"
            and last.get("display_kind") == STEER_DISPLAY_KIND
            and not last.get("_db_persisted")):
        _requeue_pending(agent, pending)
        return
    for text in pending:
        messages.append(notification_user_row(text))
    logger.info(
        "Delivered %d background notification(s) to agent after tool batch: %s",
        len(pending), "; ".join(t[:80] for t in pending),
    )
