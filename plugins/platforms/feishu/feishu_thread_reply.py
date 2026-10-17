"""Feishu topic-thread (话题) reply routing extracted from the adapter facade.

Two helpers share one theme: where a reply lands relative to Feishu topics.

* ``wants_thread_reply`` — the ``FEISHU_THREAD_REPLY`` gate: group chats only, DMs never.
* ``fetch_last_message_in_thread`` — reply-anchor lookup used when a topic send has no anchor.

Both take the adapter as data; this module must not import the facade at module level
(the facade late-imports these by topic inside its methods — see adapter.py).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
