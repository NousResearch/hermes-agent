"""Stable transcript ownership for accepted gateway input."""

import json
import uuid

from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def gateway_input_owner(event: MessageEvent | None, source: SessionSource) -> str:
    uid = getattr(event, "_pending_snapshot_uid", None)
    message_id = getattr(event, "message_id", None)
    identifier = f"pending:{uid}" if uid else str(message_id) if message_id else None
    if identifier is None:
        return str(uuid.uuid4())
    namespace = [source.platform.value, source.profile, source.scope_id,
                 source.chat_id, source.thread_id, identifier]
    return str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(namespace)))
