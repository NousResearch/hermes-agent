"""CardKit v2 client — thin executor-integrated wrapper over the Feishu SDK.

Every SDK call goes through the adapter-owned ``_run_blocking`` executor
(multiplex-safe: the worker inherits the caller's profile context), per the
platform adapter's SDK-call convention. Errors raise :class:`CardKitError`;
callers treat any exception as "fall back to native text delivery".
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Awaitable, Callable

from lark_oapi.api.cardkit.v1 import (
    BatchUpdateCardRequest,
    Card,
    BatchUpdateCardRequestBody,
    ContentCardElementRequest,
    ContentCardElementRequestBody,
    CreateCardRequest,
    CreateCardRequestBody,
    SettingsCardRequest,
    SettingsCardRequestBody,
    UpdateCardRequest,
    UpdateCardRequestBody,
)
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)


class CardKitError(RuntimeError):
    """A CardKit/IM SDK call failed (code + message from the response)."""

    def __init__(self, operation: str, code: Any, msg: str) -> None:
        super().__init__(f"{operation}: code={code} msg={msg}")
        self.operation = operation
        self.code = code
        self.msg = msg


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


class CardKitClient:
    """CardKit operations for one Feishu app, executed on the adapter's executor.

    ``run_blocking`` is the adapter's ``_run_blocking(func, *args)``; ``lark_client``
    is the adapter's configured ``lark.Client``.
    """

    def __init__(self, run_blocking: Callable[..., Awaitable[Any]], lark_client: Any) -> None:
        self._run = run_blocking
        self._lark = lark_client

    async def _checked(self, operation: str, func: Any, *args: Any) -> Any:
        response = await self._run(func, *args)
        if response is None or response.success() is False:
            code = getattr(response, "code", "unknown")
            msg = getattr(response, "msg", "no response")
            raise CardKitError(operation, code, msg)
        return response

    async def cardkit_create(self, card: dict[str, Any]) -> str:
        """Create a CardKit entity from card JSON; returns the card_id."""
        request = (
            CreateCardRequest.builder()
            .request_body(
                CreateCardRequestBody.builder().type("card_json").data(_dumps(card)).build()
            )
            .build()
        )
        resp = await self._checked("cardkit_create", self._lark.cardkit.v1.card.create, request)
        if resp.data and getattr(resp.data, "card_id", None):
            return str(resp.data.card_id)
        raise CardKitError("cardkit_create", "missing", "response missing card_id")

    async def cardkit_update(self, card_id: str, card: dict[str, Any], sequence: int = 0) -> None:
        """Full-card update (complete re-render)."""
        body = UpdateCardRequestBody.builder().card(
            Card.builder().type("card_json").data(_dumps(card)).build()
        )
        body = body.sequence(sequence)
        request = UpdateCardRequest.builder().card_id(card_id).request_body(body.build()).build()
        await self._checked("cardkit_update", self._lark.cardkit.v1.card.update, request)

    async def cardkit_stream_element(
        self, card_id: str, element_id: str, content: str, *, sequence: int = 0
    ) -> None:
        """Typewriter-update one element's content."""
        body = ContentCardElementRequestBody.builder().content(content).sequence(sequence)
        request = (
            ContentCardElementRequest.builder()
            .card_id(card_id)
            .element_id(element_id)
            .request_body(body.build())
            .build()
        )
        await self._checked(
            "cardkit_stream_element", self._lark.cardkit.v1.card_element.content, request
        )

    async def cardkit_batch_update(
        self, card_id: str, actions: list[dict[str, Any]], *, sequence: int = 0
    ) -> None:
        """Partial update (add/remove/update elements in one call)."""
        body = (
            BatchUpdateCardRequestBody.builder()
            .sequence(sequence)
            .actions(_dumps(actions))
        )
        request = (
            BatchUpdateCardRequest.builder().card_id(card_id).request_body(body.build()).build()
        )
        await self._checked(
            "cardkit_batch_update", self._lark.cardkit.v1.card.batch_update, request
        )

    async def cardkit_close_streaming(self, card_id: str, sequence: int = 0) -> None:
        """Exit streaming mode (sequence must strictly increase per call)."""
        body = (
            SettingsCardRequestBody.builder()
            .settings(_dumps({"streaming_mode": False}))
            .sequence(sequence)
        )
        request = SettingsCardRequest.builder().card_id(card_id).request_body(body.build()).build()
        await self._checked("cardkit_close_streaming", self._lark.cardkit.v1.card.settings, request)

    async def send_card_to_chat(
        self, chat_id: str, card: dict[str, Any], *, reply_to_message_id: str | None = None
    ) -> str:
        """Send a card message to a chat (optionally as a reply); returns message_id."""
        if reply_to_message_id:
            request = (
                ReplyMessageRequest.builder()
                .message_id(reply_to_message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("interactive")
                    .content(_dumps(card))
                    .uuid(uuid.uuid4().hex)
                    .build()
                )
                .build()
            )
            resp = await self._checked(
                "send_card_to_chat", self._lark.im.v1.message.areply, request
            )
        else:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("interactive")
                    .content(_dumps(card))
                    .uuid(uuid.uuid4().hex)
                    .build()
                )
                .build()
            )
            resp = await self._checked(
                "send_card_to_chat", self._lark.im.v1.message.acreate, request
            )
        if resp.data and getattr(resp.data, "message_id", None):
            return str(resp.data.message_id)
        raise CardKitError("send_card_to_chat", "missing", "response missing message_id")

    async def reply_card_by_id(self, message_id: str, card_id: str) -> str:
        """Reply a CardKit entity (by card_id) to a message; returns message_id."""
        request = (
            ReplyMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                ReplyMessageRequestBody.builder()
                .msg_type("interactive")
                .content(_dumps({"type": "card", "data": {"card_id": card_id}}))
                .uuid(uuid.uuid4().hex)
                .build()
            )
            .build()
        )
        resp = await self._checked("reply_card_by_id", self._lark.im.v1.message.areply, request)
        if resp.data and getattr(resp.data, "message_id", None):
            return str(resp.data.message_id)
        raise CardKitError("reply_card_by_id", "missing", "response missing message_id")
