"""Deletion of a turn's temporary progress bubbles (``display.platforms.<p>.cleanup_progress``)."""

import logging
from typing import Any

from gateway.turn_context import TurnContext

logger = logging.getLogger("gateway.run")


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

    async def _cleanup_temp_bubbles() -> None:
        failed = 0
        for mid in ids_snapshot:
            try:
                if await adapter.delete_message(chat_id, mid) is False:
                    failed += 1
            except Exception:  # health: allow BLE001 -- one bad delete must not stop the rest; logged below
                logger.debug("Temp bubble delete failed for %s", mid, exc_info=True)
                failed += 1
        if failed:
            logger.info("Temp bubble cleanup for session %s: %d of %d deletes failed",
                        session_key, failed, len(ids_snapshot))

    adapter.register_post_delivery_callback(session_key, _cleanup_temp_bubbles, generation=turn_ctx.run_generation)
