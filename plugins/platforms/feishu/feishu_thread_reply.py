"""Feishu topic-thread (话题) reply routing extracted from the adapter facade.

Everything about *where a reply lands relative to Feishu topics* lives here:

* ``send_threaded_reply`` — anchored reply with the ``FEISHU_THREAD_REPLY`` upgrade,
  and the plain-quote degrade when the group rejects the topic form (230071/230072).
* ``wants_thread_reply`` — the group-chat gate: DMs never thread.
* ``fetch_last_message_in_thread`` — reply-anchor lookup for topic sends without one.

The module takes the adapter as data; it must not import the facade at module level
(the facade late-imports these by topic inside its methods — see adapter.py).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Feishu rejects a topic-form reply when the group settings disallow threads (230071)
# or the anchor message is aggregated (230072) — the answer must still land, so it is
# retried once as a plain quoted reply (the pre-toggle behavior).
_THREAD_UNSUPPORTED_CODES = frozenset({230071, 230072})


async def send_threaded_reply(
    adapter: Any, *, chat_id: str, msg_type: str, payload: str, reply_to: str, metadata: Optional[dict[str, Any]],
) -> Any:
    """Reply anchored on ``reply_to``, upgraded to a topic when the toggle demands it.

    Feishu spawns the topic on the FIRST ``reply_in_thread=true`` reply to a threadless
    group message; continuation chunks anchored on the same message join it. A group that
    rejects the topic form degrades to a plain quote instead of dropping the reply.
    """
    thread_id = (metadata or {}).get("thread_id")
    reply_in_thread = bool(thread_id) or await wants_thread_reply(adapter, chat_id)
    response = await _reply(adapter, reply_to=reply_to, msg_type=msg_type, payload=payload, thread=reply_in_thread)
    if reply_in_thread and not adapter._response_succeeded(response):
        code = getattr(response, "code", None)
        if code in _THREAD_UNSUPPORTED_CODES:
            logger.info(
                "[Feishu] Topic reply to %s rejected in chat %s (code %s — group settings or aggregated "
                "anchor); falling back to plain quote", reply_to, chat_id, code,
            )
            response = await _reply(adapter, reply_to=reply_to, msg_type=msg_type, payload=payload, thread=False)
    return response


async def _reply(adapter: Any, *, reply_to: str, msg_type: str, payload: str, thread: bool) -> Any:
    body = adapter._build_reply_message_body(
        content=payload, msg_type=msg_type, reply_in_thread=thread, uuid_value=str(uuid.uuid4()),
    )
    request = adapter._build_reply_message_request(reply_to, body)
    return await adapter._run_blocking(adapter._client.im.v1.message.reply, request)


async def wants_thread_reply(adapter: Any, chat_id: str) -> bool:
    """FEISHU_THREAD_REPLY gates on group chats only — a p2p DM hosts no Feishu topic.

    The chat-info cache is warm by the time a reply goes out (the inbound that triggered
    the turn fills it via ``get_chat_info``), so this is a dict hit on the hot path; a
    cold lookup degrades to ``"dm"`` (plain quoted reply) if the API call fails.
    """
    if not getattr(adapter, "_thread_reply", False):
        return False
    if chat_id.startswith("feishu_user_id:") or chat_id.startswith("ou_"):
        return False
    info = await adapter.get_chat_info(chat_id)
    return info.get("type") in {"group", "forum"}


async def fetch_last_message_in_thread(adapter: Any, thread_id: str) -> Optional[str]:
    """Fetch the last message_id in a thread for reply-based routing."""
    if not adapter._client or not thread_id:
        return None
    try:
        from lark_oapi.api.im.v1 import ListMessageRequest
        request = ListMessageRequest.builder().container_id_type("thread").container_id(thread_id).page_size(1).build()
        response = await adapter._run_blocking(adapter._client.im.v1.message.list, request)
        if adapter._response_succeeded(response):
            items = getattr(getattr(response, "data", None), "items", None)
            if items and len(items) > 0:
                return getattr(items[0], "message_id", None)
    except Exception as exc:
        logger.debug("[Feishu] Failed to fetch last message in thread %s: %s", thread_id, exc, exc_info=True)
    return None
