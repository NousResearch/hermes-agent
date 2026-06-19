"""WebSocket fallback for Feishu group messages.

The lark-oapi SDK's EventDispatcherHandler dispatches events by event_key
(``{schema}.{event_type}``).  Both p2p and group messages arrive as
``im.message.receive_v1`` so the registered processor *should* handle both.

However, in some SDK versions or edge cases the dispatcher may fail to route
group messages (e.g. EventException, silent drop).  This module provides a
monkey-patch for the SDK's ``_handle_data_frame`` that catches dispatch
failures and manually routes group messages to the adapter's
``_on_message_event`` handler.

The fallback is a safety-net — under normal operation the SDK handles both
p2p and group messages through the same processor.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger("gateway.platforms.feishu")


def patch_ws_client_for_group_messages(ws_client: Any, adapter: Any) -> None:
    """Monkey-patch the SDK WS client to catch dropped group messages.

    Wraps ``ws_client._handle_data_frame`` so that when the SDK's
    ``EventDispatcherHandler`` fails to process an ``im.message.receive_v1``
    event, we parse the raw payload and manually route group messages to
    ``adapter._on_message_event``.
    """
    original = getattr(ws_client, "_handle_data_frame", None)
    if original is None:
        # WS client doesn't have _handle_data_frame (e.g. test mock) — nothing to patch.
        return

    async def _handle_data_frame_with_fallback(frame: Any) -> Any:
        try:
            return await original(frame)
        except Exception as dispatch_err:
            # Dispatcher failed — attempt to extract and route group messages.
            try:
                pl = frame.payload
                payload_str = pl.decode("utf-8") if isinstance(pl, bytes) else str(pl)
                payload = json.loads(payload_str)

                header = payload.get("header", {})
                event_type = header.get("event_type", "")

                if event_type == "im.message.receive_v1":
                    event_data = payload.get("event", {})
                    message = event_data.get("message", {})
                    chat_type = message.get("chat_type", "p2p")

                    if chat_type == "group":
                        logger.info(
                            "[Feishu] Fallback captured group message: "
                            "chat_id=%s message_id=%s",
                            message.get("chat_id", ""),
                            message.get("message_id", ""),
                        )
                        data = _dict_to_namespace(payload)
                        adapter._on_message_event(data)
                        return

                # Not a group message — re-raise original error
                raise dispatch_err

            except Exception:
                # Fallback also failed — re-raise the original dispatcher error
                raise dispatch_err from None

    ws_client._handle_data_frame = _handle_data_frame_with_fallback
    logger.debug("[Feishu] Group message fallback patch applied")


def _dict_to_namespace(d: Any) -> Any:
    """Recursively convert a dict to SimpleNamespace for SDK compatibility."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(item) for item in d]
    else:
        return d
