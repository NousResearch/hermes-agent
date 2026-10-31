"""原版 CardKit 请求、序列号与收敛逻辑；仅服务本插件。"""

from __future__ import annotations
import asyncio
from collections import OrderedDict
import json
import logging
import re
import uuid
from typing import Any, Dict, Optional
from lark_oapi.api.cardkit.v1.model import (
    CreateCardRequest,
    CreateCardRequestBody,
    ContentCardElementRequest,
    ContentCardElementRequestBody,
    SettingsCardRequest,
    SettingsCardRequestBody,
    UpdateCardRequest,
    UpdateCardRequestBody,
    Card,
)

logger = logging.getLogger(__name__)


class CardKitMixin:
    _CARDKIT_MESSAGE_PREFIX = "cardkit:"
    _CARDKIT_STATE_CACHE_SIZE = 512
    _CARDKIT_STREAM_ELEMENT_ID = "streaming_content"

    def _init_cardkit(self):
        self._cardkit_issued_sequences = {}
        self._cardkit_confirmed_sequences = {}
        self._cardkit_terminal = set()
        self._cardkit_fallback = {}
        self._cardkit_locks = {}
        self._cardkit_state_order = OrderedDict()

    @classmethod
    def _parse_cardkit_message_id(cls, message_id: str) -> Optional[tuple[str, str]]:
        text = str(message_id or "")
        if not text.startswith(cls._CARDKIT_MESSAGE_PREFIX):
            return None
        parts = text.split(":", 2)
        if len(parts) != 3 or not parts[1] or not parts[2]:
            return None
        return parts[1], parts[2]

    @classmethod
    def _make_cardkit_message_id(cls, card_id: str, im_message_id: str) -> str:
        return f"{cls._CARDKIT_MESSAGE_PREFIX}{card_id}:{im_message_id}"

    def _cardkit_available(self) -> bool:
        return bool(
            self._client is not None
            and getattr(self._client, "cardkit", None) is not None
            and CreateCardRequest is not None
            and CreateCardRequestBody is not None
            and ContentCardElementRequest is not None
            and ContentCardElementRequestBody is not None
            and SettingsCardRequest is not None
            and SettingsCardRequestBody is not None
            and UpdateCardRequest is not None
            and UpdateCardRequestBody is not None
            and Card is not None
        )

    def _next_cardkit_sequence(self, card_id: str) -> int:
        """Allocate a strictly increasing operation sequence (gaps are allowed).

        Confirmed mutations and ambiguous reservations advance the local
        watermark; an explicit rejection does not prove remote consumption.
        """
        return int(self._cardkit_issued_sequences.get(card_id, 0) or 0) + 1

    def _commit_cardkit_sequence(self, card_id: str, sequence: int) -> None:
        self._cardkit_confirmed_sequences[card_id] = int(sequence)
        self._reserve_cardkit_sequence(card_id, sequence)

    def _reserve_cardkit_sequence(self, card_id: str, sequence: int) -> None:
        # Issued/reserved watermark is not proof of remote acknowledgement.
        self._cardkit_issued_sequences[card_id] = int(sequence)
        self._touch_cardkit_state(card_id)

    def _touch_cardkit_state(self, card_id: str) -> None:
        """Keep recently-finalized cards available for one last reconcile.

        The gateway may perform a stale-final correction after the stream
        consumer finalized the card.  Dropping the sequence at finalize made
        that correction restart from a low value and caused 300317 plus a
        duplicate normal final send.  Retain a bounded LRU instead.
        """
        self._cardkit_state_order.pop(card_id, None)
        self._cardkit_state_order[card_id] = None
        active = {turn.card_id for turn in self._card_turns.values() if not turn.final}
        attempts = len(self._cardkit_state_order)
        while len(self._cardkit_state_order) > self._CARDKIT_STATE_CACHE_SIZE and attempts > 0:
            attempts -= 1
            stale_id, _ = self._cardkit_state_order.popitem(last=False)
            stale_lock = self._cardkit_locks.get(stale_id)
            if stale_id in active or (stale_lock is not None and stale_lock.locked()):
                self._cardkit_state_order[stale_id] = None
                continue
            self._cardkit_issued_sequences.pop(stale_id, None)
            self._cardkit_confirmed_sequences.pop(stale_id, None)
            self._cardkit_locks.pop(stale_id, None)
            self._cardkit_terminal.discard(stale_id)
            self._cardkit_fallback.pop(stale_id, None)

    def _cardkit_lock(self, card_id: str) -> asyncio.Lock:
        lock = self._cardkit_locks.get(card_id)
        if lock is None:
            lock = asyncio.Lock()
            self._cardkit_locks[card_id] = lock
        self._touch_cardkit_state(card_id)
        return lock

    def _cleanup_cardkit_state(self, card_id: str) -> None:
        self._cardkit_terminal.discard(card_id)
        self._cardkit_fallback.pop(card_id, None)
        self._cardkit_issued_sequences.pop(card_id, None)
        self._cardkit_confirmed_sequences.pop(card_id, None)
        self._cardkit_locks.pop(card_id, None)
        self._cardkit_state_order.pop(card_id, None)

    @staticmethod
    def _is_cardkit_streaming_closed_error(exc: BaseException) -> bool:
        text = str(exc or "").lower()
        return bool(
            re.search(r"\b(?:200510|300309)\b", text)
            or "streaming mode is closed" in text
            or "card streaming timeout" in text
            or ("streaming" in text and "timeout" in text)
        )

    async def _create_cardkit_card(self, card: dict[str, Any]) -> str:
        if not self._cardkit_available():
            raise RuntimeError("CardKit SDK is unavailable")
        body = (
            CreateCardRequestBody.builder()
            .type("card_json")
            .data(json.dumps(card, ensure_ascii=False))
            .build()
        )
        request = CreateCardRequest.builder().request_body(body).build()
        # Use the SDK transport timeout; cancelling a 5s executor wait leaves
        # the real create running and loses the resulting card identity.
        response = await self._run_blocking(self._client.cardkit.v1.card.create, request)
        if not self._response_succeeded(response):
            raise RuntimeError(
                self._response_error_result(
                    response,
                    default_message="cardkit create failed",
                ).error
            )
        card_id = self._extract_response_field(response, "card_id")
        if not card_id:
            raise RuntimeError("CardKit create response missing card_id")
        # Creation has no sequence field. The first mutation starts at 1,
        # as in the content API example; no server-reserved sequence is assumed.
        card_id = str(card_id)
        self._cardkit_issued_sequences[card_id] = 0
        self._touch_cardkit_state(card_id)
        return card_id

    async def _stream_cardkit_content(self, card_id: str, content: str) -> None:
        if not self._cardkit_available():
            raise RuntimeError("CardKit SDK is unavailable")
        sequence = self._next_cardkit_sequence(card_id)
        body = (
            ContentCardElementRequestBody.builder()
            .content(content or " ")
            .sequence(sequence)
            .uuid(str(uuid.uuid4()))
            .build()
        )
        request = (
            ContentCardElementRequest.builder()
            .card_id(card_id)
            .element_id(self._CARDKIT_STREAM_ELEMENT_ID)
            .request_body(body)
            .build()
        )
        from .cardkit_mutations import mutate

        await mutate(
            self,
            card_id,
            self._client.cardkit.v1.card_element.content,
            request,
            "cardkit content update failed",
        )

    async def _stream_cardkit_content_with_recovery(
        self,
        card_id: str,
        content: str,
    ) -> None:
        """Reopen only a live card after an explicit streaming-window rejection.

        Transport ambiguity is handled by replaying the identical SDK request
        inside the mutation helper, not by allocating a new operation.
        """
        try:
            await self._stream_cardkit_content(card_id, content)
        except Exception as exc:
            if card_id in self._cardkit_terminal or not self._is_cardkit_streaming_closed_error(
                exc
            ):
                raise
            await self._set_cardkit_streaming_mode(card_id, True)
            await self._stream_cardkit_content(card_id, content)

    async def _set_cardkit_streaming_mode(
        self,
        card_id: str,
        streaming_mode: bool,
    ) -> None:
        if not self._cardkit_available():
            raise RuntimeError("CardKit SDK is unavailable")
        sequence = self._next_cardkit_sequence(card_id)
        body = (
            SettingsCardRequestBody.builder()
            .settings(json.dumps({"config": {"streaming_mode": bool(streaming_mode)}}))
            .sequence(sequence)
            .uuid(str(uuid.uuid4()))
            .build()
        )
        request = SettingsCardRequest.builder().card_id(card_id).request_body(body).build()
        from .cardkit_mutations import mutate

        await mutate(
            self, card_id, self._client.cardkit.v1.card.settings, request, "cardkit settings failed"
        )

    async def _update_cardkit_card(
        self,
        card_id: str,
        card: dict[str, Any],
    ) -> None:
        if not self._cardkit_available():
            raise RuntimeError("CardKit SDK is unavailable")
        sequence = self._next_cardkit_sequence(card_id)
        card_body = (
            Card.builder().type("card_json").data(json.dumps(card, ensure_ascii=False)).build()
        )
        body = (
            UpdateCardRequestBody.builder()
            .card(card_body)
            .sequence(sequence)
            .uuid(str(uuid.uuid4()))
            .build()
        )
        request = UpdateCardRequest.builder().card_id(card_id).request_body(body).build()
        from .cardkit_mutations import mutate

        await mutate(
            self, card_id, self._client.cardkit.v1.card.update, request, "cardkit update failed"
        )

    @staticmethod
    def _is_cardkit_sequence_rejected_error(exc: BaseException) -> bool:
        return bool(re.search(r"\b300317\b", str(exc or "")))

    async def _update_cardkit_card_with_seq_recovery(
        self,
        card_id: str,
        card: dict[str, Any],
    ) -> None:
        """No blind sequence probing: 300317 does not reveal server state.

        All transport retries live inside _update_cardkit_card and preserve
        request identity. Explicit rejection must remain visible to fallback.
        """
        await self._update_cardkit_card(card_id, card)

    async def _close_cardkit_card_before_fallback(
        self,
        card_id: Optional[str],
        content: str,
        *,
        cleanup: bool = True,
    ) -> None:
        if not card_id:
            return
        try:
            try:
                # Replace the card and close streaming in one sequenced
                # mutation.  Closing through ``card.settings`` first ends the
                # streaming sequence window; a following full-card update
                # then races that boundary and Feishu rejects it with 300317
                # ("sequence number compare failed").
                await self._update_cardkit_card_with_seq_recovery(
                    card_id,
                    self._build_cardkit_terminal_notice(content),
                )
            except Exception as exc:
                logger.warning("[Feishu] Failed to update orphan CardKit notice: %s", exc, exc_info=True)
                try:
                    await self._set_cardkit_streaming_mode(card_id, False)
                except Exception as close_exc:
                    logger.warning(
                        "[Feishu] Failed to stop orphan CardKit streaming: %s",
                        close_exc, exc_info=True,
                    )
        finally:
            if cleanup:
                self._cleanup_cardkit_state(card_id)
