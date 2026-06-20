"""Feishu CardKit API wrapper for streaming card support.

Provides async helper functions around the lark_oapi SDK's cardkit v1 API:
  - card.create   → create a card entity
  - card.settings → configure streaming mode
  - card_element.content → stream element content
  - card.update   → full card update

Uses sync SDK methods via ``asyncio.to_thread`` for consistency with the
rest of the Hermes feishu adapter (which does the same for IM calls).
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
from typing import Any, Optional

logger = logging.getLogger("gateway.platforms.feishu_cardkit")

# ---------------------------------------------------------------------------
# Lazy SDK imports – tolerate missing lark_oapi at import time
# ---------------------------------------------------------------------------

_CARDKIT_AVAILABLE = False

try:
    from lark_oapi.api.cardkit.v1 import (
        CreateCardRequest,
        CreateCardRequestBody,
        ContentCardElementRequest,
        ContentCardElementRequestBody,
        UpdateCardRequest,
        UpdateCardRequestBody,
        SettingsCardRequest,
        SettingsCardRequestBody,
    )
    from lark_oapi.api.cardkit.v1.model.card import Card as CardModel

    _CARDKIT_AVAILABLE = True
except ImportError:
    pass


def cardkit_available() -> bool:
    """Return True if the lark_oapi cardkit v1 API is importable."""
    return _CARDKIT_AVAILABLE


# ---------------------------------------------------------------------------
# Streaming element ID – must match the ``element_id`` field in the card JSON
# ---------------------------------------------------------------------------

STREAMING_ELEMENT_ID = "streaming_content"


# ---------------------------------------------------------------------------
# Response helper
# ---------------------------------------------------------------------------

def _is_success(response: Any) -> bool:
    """Check if a lark_oapi SDK response indicates success."""
    if response is None:
        return False
    # response.success() is a method on lark_oapi responses
    success_fn = getattr(response, "success", None)
    if callable(success_fn):
        return bool(success_fn())
    # Fallback: check code
    code = getattr(response, "code", -1)
    return code == 0


# ---------------------------------------------------------------------------
# Card JSON builder
# ---------------------------------------------------------------------------

def build_streaming_card_json(
    *,
    title: str = "Hermes",
    header_template: str = "blue",
    initial_content: str = "",
) -> str:
    """Build the Feishu interactive card JSON for a streaming card.

    Uses Card JSON 2.0 format (required by CardKit API):
      - ``schema: "2.0"``
      - elements nested under ``body.elements``
    """
    card = {
        "schema": "2.0",
        "config": {
            "wide_screen_mode": True,
            "streaming_mode": True,
        },
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": header_template,
        },
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": initial_content,
                    "element_id": STREAMING_ELEMENT_ID,
                }
            ],
        },
    }
    return json.dumps(card, ensure_ascii=False)


# ---------------------------------------------------------------------------
# API wrappers — all use sync SDK methods via asyncio.to_thread
# ---------------------------------------------------------------------------

async def create_card_entity(client: Any, card_json: str) -> Optional[str]:
    """Create a CardKit card entity and return its ``card_id``."""
    if not _CARDKIT_AVAILABLE:
        logger.warning("[cardkit] SDK not available, cannot create card")
        return None

    try:
        body = CreateCardRequestBody()
        body.type = "card_json"
        body.data = card_json

        request = CreateCardRequest.builder().request_body(body).build()

        # Use sync method via to_thread (consistent with IM calls in feishu.py)
        response = await asyncio.to_thread(client.cardkit.v1.card.create, request)

        if not _is_success(response):
            code = getattr(response, "code", "?")
            msg = getattr(response, "msg", "unknown error")
            logger.warning("[cardkit] create failed: [%s] %s", code, msg)
            return None

        data = getattr(response, "data", None)
        card_id = getattr(data, "card_id", None) if data else None
        if not card_id:
            logger.warning("[cardkit] create succeeded but no card_id returned")
            return None

        logger.info("[cardkit] card created: %s", card_id)
        return card_id

    except Exception as exc:
        logger.warning("[cardkit] create_card_entity error: %s\n%s", exc, traceback.format_exc())
        return None


async def send_card_by_id(
    client: Any,
    *,
    receive_id: str,
    card_id: str,
    reply_to_message_id: Optional[str] = None,
) -> Optional[str]:
    """Send an existing CardKit card to a chat as an interactive message."""
    try:
        from lark_oapi.api.im.v1 import (
            CreateMessageRequest,
            CreateMessageRequestBody,
            ReplyMessageRequest,
            ReplyMessageRequestBody,
        )

        # CardKit card reference payload (JSON 2.0 format)
        payload = json.dumps({"type": "card", "data": {"card_id": card_id}}, ensure_ascii=False)

        if reply_to_message_id:
            body = (
                ReplyMessageRequestBody.builder()
                .content(payload)
                .msg_type("interactive")
                .build()
            )
            request = (
                ReplyMessageRequest.builder()
                .message_id(reply_to_message_id)
                .request_body(body)
                .build()
            )
            response = await asyncio.to_thread(client.im.v1.message.reply, request)
        else:
            receive_id_type = "open_id" if receive_id.startswith("ou_") else "chat_id"
            body = (
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type("interactive")
                .content(payload)
                .uuid(str(uuid.uuid4()))
                .build()
            )
            request = (
                CreateMessageRequest.builder()
                .receive_id_type(receive_id_type)
                .request_body(body)
                .build()
            )
            response = await asyncio.to_thread(client.im.v1.message.create, request)

        if not _is_success(response):
            code = getattr(response, "code", "?")
            msg = getattr(response, "msg", "unknown error")
            logger.warning("[cardkit] send_card failed: [%s] %s", code, msg)
            return None

        data = getattr(response, "data", None)
        message_id = getattr(data, "message_id", None) if data else None
        logger.info("[cardkit] card sent to chat: msg_id=%s", message_id)
        return message_id

    except Exception as exc:
        logger.warning("[cardkit] send_card_by_id error: %s\n%s", exc, traceback.format_exc())
        return None


async def set_card_streaming_mode(
    client: Any,
    *,
    card_id: str,
    streaming: bool,
    sequence: int = 0,
) -> bool:
    """Enable or disable streaming mode on a card via the settings API."""
    if not _CARDKIT_AVAILABLE:
        return False

    try:
        settings_json = json.dumps({"streaming_mode": streaming})
        body = (
            SettingsCardRequestBody.builder()
            .settings(settings_json)
            .uuid(str(uuid.uuid4()))
            .sequence(sequence)
            .build()
        )
        request = (
            SettingsCardRequest.builder()
            .card_id(card_id)
            .request_body(body)
            .build()
        )
        response = await asyncio.to_thread(client.cardkit.v1.card.settings, request)

        if not _is_success(response):
            code = getattr(response, "code", "?")
            msg = getattr(response, "msg", "unknown error")
            logger.warning(
                "[cardkit] settings(streaming=%s) failed: [%s] %s",
                streaming, code, msg,
            )
            return False

        logger.debug(
            "[cardkit] streaming_mode=%s seq=%d card=%s",
            streaming, sequence, card_id,
        )
        return True

    except Exception as exc:
        logger.warning("[cardkit] set_card_streaming_mode error: %s", exc)
        return False


async def stream_card_content(
    client: Any,
    *,
    card_id: str,
    element_id: str,
    content: str,
    sequence: int,
) -> bool:
    """Update a single element's content on a streaming card."""
    if not _CARDKIT_AVAILABLE:
        return False

    try:
        body = (
            ContentCardElementRequestBody.builder()
            .uuid(str(uuid.uuid4()))
            .content(content)
            .sequence(sequence)
            .build()
        )
        request = (
            ContentCardElementRequest.builder()
            .card_id(card_id)
            .element_id(element_id)
            .request_body(body)
            .build()
        )
        response = await asyncio.to_thread(client.cardkit.v1.card_element.content, request)

        if not _is_success(response):
            code = getattr(response, "code", "?")
            msg = getattr(response, "msg", "unknown error")
            logger.warning(
                "[cardkit] content update failed seq=%d: [%s] %s",
                sequence, code, msg,
            )
            return False

        logger.debug(
            "[cardkit] content updated seq=%d len=%d card=%s",
            sequence, len(content), card_id,
        )
        return True

    except Exception as exc:
        logger.warning("[cardkit] stream_card_content error: %s", exc)
        return False


async def update_card_kit_card(
    client: Any,
    *,
    card_id: str,
    card_json: str,
    sequence: int,
) -> bool:
    """Replace the entire card definition (full update)."""
    if not _CARDKIT_AVAILABLE:
        return False

    try:
        card_model = CardModel.builder().type("card_json").data(card_json).build()
        body = (
            UpdateCardRequestBody.builder()
            .card(card_model)
            .uuid(str(uuid.uuid4()))
            .sequence(sequence)
            .build()
        )
        request = (
            UpdateCardRequest.builder()
            .card_id(card_id)
            .request_body(body)
            .build()
        )
        response = await asyncio.to_thread(client.cardkit.v1.card.update, request)

        if not _is_success(response):
            code = getattr(response, "code", "?")
            msg = getattr(response, "msg", "unknown error")
            logger.warning("[cardkit] update failed seq=%d: [%s] %s", sequence, code, msg)
            return False

        logger.debug("[cardkit] full card updated seq=%d card=%s", sequence, card_id)
        return True

    except Exception as exc:
        logger.warning("[cardkit] update_card_kit_card error: %s", exc)
        return False
