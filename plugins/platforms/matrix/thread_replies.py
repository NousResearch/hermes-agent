"""Resolve reply fallbacks for proactive Matrix thread delivery."""
import asyncio
import logging
from urllib.parse import quote

logger = logging.getLogger(__name__)

async def resolve_thread_reply_target(
    client,
    room_id: str,
    thread_id: str,
) -> str:
    """Return the newest event in a thread for reply fallback chaining.

    Matrix thread events always relate to the thread root, while
    ``m.in_reply_to`` should point at the message being continued. Proactive
    sends do not have an inbound ``reply_to`` value, so resolve the latest
    thread event from the homeserver. If the lookup is unavailable, the
    root remains a valid protocol-level fallback.
    """
    if not client or not thread_id:
        return thread_id

    encoded_room = quote(room_id, safe="")
    encoded_thread = quote(thread_id, safe="")
    path = (
        f"/_matrix/client/v1/rooms/{encoded_room}/relations/"
        f"{encoded_thread}/m.thread/m.room.message"
    )
    get_method: object
    try:
        from mautrix.api import Method as MatrixMethod

        get_method = MatrixMethod.GET
    except ImportError:
        get_method = "GET"
    try:
        response = await asyncio.wait_for(
            client.api.request(
                get_method,
                path,
                query_params={"dir": "b", "limit": "1"},
            ),
            timeout=10,
        )
        chunk = response.get("chunk", []) if isinstance(response, dict) else []
        if chunk and isinstance(chunk[0], dict):
            event_id = str(chunk[0].get("event_id") or "")
            if event_id:
                return event_id
    except Exception as exc:
        logger.debug(
            "Matrix: could not resolve latest event for thread %s in %s: %s",
            thread_id,
            room_id,
            exc,
        )
    return thread_id

