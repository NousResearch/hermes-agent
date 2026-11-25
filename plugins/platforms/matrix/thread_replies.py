"""Matrix thread relations: the inbound thread root and the outbound reply fallback anchor."""
import asyncio
import logging
from typing import Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


def _thread_root(relates_to: dict) -> Optional[str]:
    """The m.thread root event_id an event belongs to, else None."""
    return relates_to.get("event_id") if relates_to.get("rel_type") == "m.thread" else None


async def thread_fallback_target(
    client, room_id: str, reply_to: Optional[str], metadata: Optional[dict[str, Any]],
) -> Optional[str]:
    """The event a proactive thread send's reply fallback continues from.

    None when the send is not in a thread or already has an explicit reply target (no lookup);
    otherwise the newest thread event, or the root when the lookup fails.
    """
    thread_id = str((metadata or {}).get("thread_id") or "")
    if not thread_id or reply_to:
        return None
    return await resolve_thread_reply_target(client, room_id, thread_id)


async def resolve_thread_reply_target(client, room_id: str, thread_id: str) -> str:
    """Return the newest event in a thread for reply fallback chaining.

    Matrix thread events always relate to the thread root, while the ``m.in_reply_to`` fallback
    should point at the latest message in the thread (MSC3440 backwards compatibility). Proactive
    sends have no inbound ``reply_to``, so ask the homeserver's relations API. If the lookup is
    unavailable, the root remains a valid protocol-level fallback.
    """
    if not client or not thread_id:
        return thread_id
    path = (
        f"/_matrix/client/v1/rooms/{quote(room_id, safe='')}/relations/"
        f"{quote(thread_id, safe='')}/m.thread/m.room.message"
    )
    get_method: object
    try:
        from mautrix.api import Method as MatrixMethod

        get_method = MatrixMethod.GET
    except ImportError:
        get_method = "GET"
    try:
        response = await asyncio.wait_for(
            client.api.request(get_method, path, query_params={"dir": "b", "limit": "1"}),
            timeout=10,
        )
    except Exception:  # best-effort lookup: any failure keeps the root fallback
        logger.debug("Matrix: could not resolve the latest thread event", exc_info=True)
        return thread_id
    chunk = response.get("chunk", []) if isinstance(response, dict) else []
    if chunk and isinstance(chunk[0], dict):
        event_id = str(chunk[0].get("event_id") or "")
        if event_id:
            return event_id
    return thread_id
