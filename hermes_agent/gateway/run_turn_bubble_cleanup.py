"""Deletion of a turn's temporary progress bubbles (``display.platforms.<p>.cleanup_progress``)."""

import asyncio
import logging
from typing import Any

from hermes_agent.gateway.turn_context import TurnContext

logger = logging.getLogger("hermes_agent.gateway.run")

# Per-delete bound: the callback runs under the post-delivery timeout (30 s), so one slow delete must
# not use it up and leave the remaining bubbles undeleted. Also caps the handoff's sender wait.
DELETE_TIMEOUT_SECONDS = 10.0


async def settle_turn_senders(*tasks: Any) -> None:
    """Stop the turn's bubble senders (progress, heartbeat) and wait for them, as the turn's
    ``finally`` does, so every bubble id they mint is tracked before a queued follow-up takes over.

    Bounded by ``DELETE_TIMEOUT_SECONDS``: a sender whose cancel-time flush edit hangs is cancelled
    again and the follow-up goes ahead (ids it mints later are lost). ``asyncio.wait`` never raises
    the senders' own cancellation, so a cancel aimed at the turn (shutdown, /stop) propagates and no
    follow-up starts."""
    live = [task for task in tasks if task is not None]
    if not live:
        return
    for task in live:
        task.cancel()
    done, pending = await asyncio.wait(live, timeout=DELETE_TIMEOUT_SECONDS)
    for task in pending:
        task.cancel()
    for task in done:
        if not task.cancelled() and task.exception() is not None:
            logger.debug("Bubble sender failed at handoff", exc_info=task.exception())


def register_bubble_cleanup(response: Any, turn_ctx: TurnContext) -> None:
    """Register deletion of the turn's tracked progress bubbles as its post-delivery callback.

    The callback is async and awaits every delete, so whoever fires it (the adapter after the final
    send, or the queued-follow-up handoff before the next turn starts) waits for the deletion.
    Failed runs keep them as breadcrumbs. Only on adapters with ``delete_message``."""
    adapter, ids, session_key = turn_ctx._cleanup_adapter, turn_ctx._cleanup_msg_ids, turn_ctx.session_key
    if not (
        turn_ctx._cleanup_progress and adapter is not None and ids and session_key
        and isinstance(response, dict) and not response.get("failed")
        and hasattr(adapter, "register_post_delivery_callback")
    ):
        return
    ids_snapshot, chat_id = list(ids), turn_ctx.source.chat_id

    async def _delete(mid: str) -> bool:
        try:
            return await asyncio.wait_for(adapter.delete_message(chat_id, mid), DELETE_TIMEOUT_SECONDS) is not False
        except Exception:  # health: allow BLE001 -- one bad or slow delete must not stop the rest; logged below
            logger.debug("Temp bubble delete failed for %s", mid, exc_info=True)
            return False

    async def _cleanup_temp_bubbles() -> None:
        failed = sum(not ok for ok in await asyncio.gather(*(_delete(mid) for mid in ids_snapshot)))
        if failed:
            logger.info("Temp bubble cleanup for session %s: %d of %d deletes failed",
                        session_key, failed, len(ids_snapshot))

    adapter.register_post_delivery_callback(session_key, _cleanup_temp_bubbles, generation=turn_ctx.run_generation)
