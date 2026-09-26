"""Deliver background delegation results while an ACP prompt RPC is still open.

The async delegation registry pushes completion events onto the shared
``process_registry.completion_queue``; CLI/gateway drains pick them up when the
session goes idle. An ACP host has no idle-drain poller — its prompt RPC is the
unit of interaction — so the session itself waits for its own delegations inside
``_finish_turn`` and ingests each completion as a same-prompt follow-up turn.
Delivery stays claim-based: a completion is dequeued only after its consumer
claimed it, and unaccepted deliveries (prompt error, cancellation) are released
and requeued so a later prompt can pick them up."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

_CONSUMER = "acp-background-followup"
_POLL_SECONDS = 0.25


def _cancelled(state: Any) -> bool:
    return bool(state.cancel_event and state.cancel_event.is_set())


def _requeue(events: list[tuple[dict, str]]) -> None:
    from tools.process_registry import process_registry

    for event, _text in events:
        process_registry.completion_queue.put(event)


async def run_background_followups(
    *,
    state: Any,
    session_id: str,
    prompt: Callable[[str], Awaitable[Any]],
) -> None:
    """Wait for this ACP session's delegations and ingest each completion once.

    Publication changes a delegation from live to complete only after its event is
    queued. When liveness first reports false, a second drain closes the window in
    which the first drain ran just before publication.
    """
    from tools.async_delegation import (
        claim_event_delivery,
        complete_event_delivery,
        has_live_for_session,
        release_event_delivery,
    )
    from tools.process_registry import process_registry

    def drain() -> list[tuple[dict, str]]:
        return process_registry.drain_notifications(
            session_key=session_id,
            owns_event=lambda event: (
                str(event.get("session_key") or "") == session_id
                or str(event.get("origin_ui_session_id") or "") == session_id
            ),
            skip_poll_observed=False,
        )

    while not _cancelled(state):
        notifications = drain()
        if not notifications:
            if has_live_for_session(
                session_key=session_id,
                origin_ui_session_id=session_id,
                parent_session_id=session_id,
            ):
                await asyncio.sleep(_POLL_SECONDS)
                continue
            # A finalizer may have queued the result between drain() and the
            # liveness check. Publication precedes the transition out of live.
            notifications = drain()
            if not notifications:
                return

        for index, (event, text) in enumerate(notifications):
            if _cancelled(state):
                _requeue(notifications[index:])
                return
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                claim = claim_event_delivery(event, _CONSUMER)
            except Exception:
                logger.warning("ACP background completion claim failed for %s", session_id, exc_info=True)
                _requeue(notifications[index:])
                return
            if claim is None:
                # Another consumer holds the claim; leave the event to it.
                continue
            try:
                response = await prompt(text)
            except Exception:
                release_event_delivery(event, claim)
                _requeue(notifications[index:])
                logger.warning("ACP background follow-up failed for %s", session_id, exc_info=True)
                return
            if _cancelled(state) or getattr(response, "stop_reason", None) == "cancelled":
                release_event_delivery(event, claim)
                _requeue(notifications[index:])
                return
            complete_event_delivery(event, claim)
