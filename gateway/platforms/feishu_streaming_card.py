"""Feishu CardKit streaming-card helper."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from gateway.platforms.base import SendResult
from lark_oapi.api.cardkit.v1 import (
    ContentCardElementRequest,
    ContentCardElementRequestBody,
    CreateCardRequest,
    CreateCardRequestBody,
    SettingsCardRequest,
    SettingsCardRequestBody,
)


logger = logging.getLogger("gateway.feishu.streaming_card")

CARD_CONTENT_ELEMENT_ID = "content"
STREAMING_UPDATE_THROTTLE_MS = 160
STREAMING_SIGNIFICANT_DELTA_CHARS = 18
MAX_CARD_TEXT_LENGTH = 30000

_NATURAL_STREAMING_BOUNDARIES = "\n.!?;:。！？；："


CARDKIT_ASSISTANT_PROFILE = "assistant"
CARDKIT_TOOL_PROGRESS_PROFILE = "tool_progress"
CARDKIT_STATIC_PROFILE = "static"

CARDKIT_STREAMING_PROFILES: Dict[str, Dict[str, Any]] = {
    # Assistant responses should look like native LLM streaming: small steps,
    # predictable cadence, and no dependence on gateway edit-message tuning.
    CARDKIT_ASSISTANT_PROFILE: {
        "print_frequency_ms": {"default": 50},
        "print_step": {"default": 5},
        "print_strategy": "fast",
    },
    # Tool progress is already emitted as coarse lines.  A larger print step
    # avoids typewriter-dribbling command/status text while still letting
    # CardKit own the native "generating" lifecycle.
    CARDKIT_TOOL_PROGRESS_PROFILE: {
        "print_frequency_ms": {"default": 30},
        "print_step": {"default": 80},
        "print_strategy": "fast",
    },
}


def _summary_for_content(content: str, *, streaming_mode: bool) -> str:
    visible = strip_streaming_cursor(content or "").strip()
    if visible:
        return truncate_summary(visible)
    if streaming_mode:
        return "[Generating...]"
    return ""


def build_card(
    content: str = "",
    *,
    streaming_mode: bool,
    profile: str = CARDKIT_ASSISTANT_PROFILE,
) -> Dict[str, Any]:
    visible_content = strip_streaming_cursor(content or "")
    config: Dict[str, Any] = {
        "streaming_mode": bool(streaming_mode),
        "summary": {
            "content": _summary_for_content(
                visible_content,
                streaming_mode=streaming_mode,
            )
        },
    }
    if streaming_mode:
        config["streaming_config"] = copy.deepcopy(
            CARDKIT_STREAMING_PROFILES.get(profile)
            or CARDKIT_STREAMING_PROFILES[CARDKIT_ASSISTANT_PROFILE]
        )

    return {
        "schema": "2.0",
        "config": config,
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": visible_content,
                    "element_id": CARD_CONTENT_ELEMENT_ID,
                }
            ]
        },
    }


def build_streaming_card() -> Dict[str, Any]:
    return build_card("", streaming_mode=True, profile=CARDKIT_ASSISTANT_PROFILE)


def truncate_summary(text: str, max_chars: int = 50) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def has_natural_streaming_boundary(text: str) -> bool:
    return bool(text) and text[-1] in _NATURAL_STREAMING_BOUNDARIES


def should_push_streaming_update(previous_text: str, next_text: str) -> bool:
    if not previous_text:
        return True
    delta_chars = max(0, len(next_text) - len(previous_text))
    return (
        has_natural_streaming_boundary(next_text)
        or delta_chars >= STREAMING_SIGNIFICANT_DELTA_CHARS
    )


def merge_streaming_text(previous_text: Optional[str], next_text: Optional[str]) -> str:
    previous = previous_text or ""
    next_value = next_text or ""
    if not previous:
        return next_value
    if not next_value:
        return previous
    if next_value.startswith(previous):
        return next_value
    if previous.startswith(next_value):
        return previous
    if previous.endswith(next_value):
        return previous

    max_overlap = min(len(previous), len(next_value))
    for overlap in range(max_overlap, 0, -1):
        if previous.endswith(next_value[:overlap]):
            return previous + next_value[overlap:]
    return previous + next_value


def strip_streaming_cursor(text: str) -> str:
    if text.endswith(" ▉"):
        return text[:-2].rstrip()
    if text.endswith("▉"):
        return text[:-1].rstrip()
    return text


class FeishuCardKitClient:
    def __init__(self, sdk_client: Any):
        self.sdk_client = sdk_client

    @staticmethod
    def _ensure_success(response: Any, action: str) -> None:
        success = getattr(response, "success", None)
        if callable(success) and success():
            return
        code = getattr(response, "code", None)
        if code in (0, None) and not callable(success):
            return
        message = (
            getattr(response, "msg", "")
            or getattr(response, "message", "")
            or "unknown error"
        )
        raise RuntimeError(f"Feishu CardKit {action} failed: {message}")

    def _card_resource(self) -> Any:
        # Keep CardKit access behind the SDK client created by the adapter so
        # Feishu/Lark domain selection and tenant-token handling stay in one
        # place instead of reimplementing raw HTTP here.
        return self.sdk_client.cardkit.v1.card

    def _card_element_resource(self) -> Any:
        return self.sdk_client.cardkit.v1.card_element

    async def create_card(
        self,
        *,
        content: str = "",
        streaming_mode: bool = True,
        profile: str = CARDKIT_ASSISTANT_PROFILE,
    ) -> str:
        body = (
            CreateCardRequestBody.builder()
            .type("card_json")
            .data(
                json.dumps(
                    build_card(
                        content,
                        streaming_mode=streaming_mode,
                        profile=profile,
                    ),
                    ensure_ascii=False,
                )
            )
            .build()
        )
        request = CreateCardRequest.builder().request_body(body).build()
        response = await self._card_resource().acreate(request)
        self._ensure_success(response, "create card")
        card_id = getattr(getattr(response, "data", None), "card_id", None)
        if not card_id:
            raise RuntimeError("Feishu CardKit create card failed: missing card_id")
        return str(card_id)

    async def update_element_content(
        self,
        card_id: str,
        element_id: str,
        content: str,
        sequence: int,
    ) -> None:
        # Do not silently truncate CardKit content.  If the full snapshot is
        # too large for CardKit, fail setup/update explicitly so the Feishu
        # adapter can fall back to the standard chunking send path when this is
        # still a send setup failure.
        if len(content) > MAX_CARD_TEXT_LENGTH:
            raise ValueError(
                f"CardKit content exceeds {MAX_CARD_TEXT_LENGTH} characters"
            )
        body = (
            ContentCardElementRequestBody.builder()
            .content(content)
            .sequence(sequence)
            .uuid(f"s_{card_id}_{sequence}")
            .build()
        )
        request = (
            ContentCardElementRequest.builder()
            .card_id(card_id)
            .element_id(element_id)
            .request_body(body)
            .build()
        )
        response = await self._card_element_resource().acontent(request)
        self._ensure_success(response, "update card")

    async def close_card(self, card_id: str, final_text: str, sequence: int) -> None:
        settings = {
            "config": {
                "streaming_mode": False,
                "summary": {"content": truncate_summary(final_text)},
            }
        }
        body = (
            SettingsCardRequestBody.builder()
            .settings(json.dumps(settings, ensure_ascii=False))
            .sequence(sequence)
            .uuid(f"c_{card_id}_{sequence}")
            .build()
        )
        request = (
            SettingsCardRequest.builder()
            .card_id(card_id)
            .request_body(body)
            .build()
        )
        response = await self._card_resource().asettings(request)
        self._ensure_success(response, "close card")


SendCardReference = Callable[..., Awaitable[SendResult]]


class FeishuStreamingCardSession:
    def __init__(
        self,
        *,
        client: Any,
        chat_id: str,
        send_card_reference: SendCardReference,
        profile: str = CARDKIT_ASSISTANT_PROFILE,
    ):
        self.client = client
        self.chat_id = chat_id
        self.send_card_reference = send_card_reference
        self.profile = profile
        self.card_id: Optional[str] = None
        self.message_id: Optional[str] = None
        self.sequence = 1
        self.current_text = ""
        self.pending_text: Optional[str] = None
        self.closed = False
        self.last_update_time = 0.0
        self._pending_flush_task: Optional[asyncio.Task] = None

    async def start(
        self,
        initial_text: str,
        *,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> SendResult:
        if self.card_id and self.message_id:
            return SendResult(success=True, message_id=self.message_id)
        self.card_id = await self.client.create_card(
            content=strip_streaming_cursor(initial_text),
            streaming_mode=True,
            profile=self.profile,
        )
        send_result = await self.send_card_reference(
            card_id=self.card_id,
            chat_id=self.chat_id,
            reply_to=reply_to,
            metadata=metadata,
        )
        if not send_result.success:
            return send_result
        self.message_id = send_result.message_id
        text = strip_streaming_cursor(initial_text)
        if text:
            # The create-card payload seeds initial content for clients that
            # render the card immediately, but CardKit streaming only becomes
            # visibly current after an element-content update.  Send that first
            # forced snapshot before the gateway starts issuing edit updates.
            try:
                await self._push_update(text, force=True)
            except Exception as exc:
                logger.warning(
                    "[Feishu] CardKit initial update failed for %s: %s",
                    self.card_id,
                    exc,
                    exc_info=True,
                )
                await self.close(None)
                return SendResult(
                    success=False,
                    message_id=self.message_id,
                    error=str(exc),
                    raw_response=getattr(send_result, "raw_response", None),
                )
        return SendResult(
            success=True,
            message_id=self.message_id,
            raw_response=getattr(send_result, "raw_response", None),
        )

    async def update(self, text: str) -> SendResult:
        if self.closed or not self.card_id or not self.message_id:
            return SendResult(success=False, error="CardKit session is not active")
        # CardKit receives full visible markdown snapshots, not deltas.  The
        # merge step protects continuity across gateway cursor stripping,
        # throttled updates, and providers that resend overlapping text.
        next_text = merge_streaming_text(
            self.pending_text or self.current_text,
            strip_streaming_cursor(text),
        )
        if not next_text or next_text == self.current_text:
            return SendResult(success=True, message_id=self.message_id)
        self.pending_text = next_text
        if not should_push_streaming_update(
            self.current_text,
            next_text,
        ):
            self._schedule_pending_flush()
            return SendResult(success=True, message_id=self.message_id)
        now_ms = time.monotonic() * 1000
        if now_ms - self.last_update_time < STREAMING_UPDATE_THROTTLE_MS:
            self._schedule_pending_flush(now_ms=now_ms)
            return SendResult(success=True, message_id=self.message_id)
        await self._push_update(next_text, force=True)
        return SendResult(success=True, message_id=self.message_id)

    async def close(self, final_text: Optional[str] = None) -> SendResult:
        self._cancel_pending_flush()
        if self.closed:
            return SendResult(success=True, message_id=self.message_id)
        if not self.card_id or not self.message_id:
            return SendResult(success=False, error="CardKit session is not active")
        merged = merge_streaming_text(self.current_text, self.pending_text)
        if final_text:
            merged = merge_streaming_text(merged, strip_streaming_cursor(final_text))
        if merged and merged != self.current_text:
            try:
                await self._push_update(merged, force=True)
            except Exception as exc:
                logger.warning(
                    "[Feishu] CardKit final update failed for %s: %s",
                    self.card_id,
                    exc,
                    exc_info=True,
                )
                return SendResult(
                    success=False,
                    message_id=self.message_id,
                    error=str(exc),
                )
        self.sequence += 1
        try:
            await self.client.close_card(
                self.card_id,
                merged or self.current_text,
                self.sequence,
            )
        except Exception as exc:
            logger.warning(
                "[Feishu] CardKit close failed for %s: %s",
                self.card_id,
                exc,
                exc_info=True,
            )
            return SendResult(success=False, message_id=self.message_id, error=str(exc))
        self.closed = True
        return SendResult(success=True, message_id=self.message_id)

    def _schedule_pending_flush(self, *, now_ms: Optional[float] = None) -> None:
        if self.closed or not self.pending_text:
            return
        if self._pending_flush_task and not self._pending_flush_task.done():
            return
        if now_ms is None:
            now_ms = time.monotonic() * 1000
        elapsed_ms = now_ms - self.last_update_time
        delay_ms = max(0.0, STREAMING_UPDATE_THROTTLE_MS - elapsed_ms)
        self._pending_flush_task = asyncio.create_task(
            self._flush_pending_after_delay(delay_ms / 1000)
        )

    def _cancel_pending_flush(self) -> None:
        task = self._pending_flush_task
        self._pending_flush_task = None
        if task and not task.done():
            task.cancel()

    async def _flush_pending_after_delay(self, delay_seconds: float) -> None:
        try:
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            if self.closed or not self.card_id or not self.message_id:
                return
            text = self.pending_text
            if not text or text == self.current_text:
                return
            await self._push_update(text, force=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "[Feishu] CardKit pending flush failed for %s: %s",
                self.card_id,
                exc,
                exc_info=True,
            )
        finally:
            task = asyncio.current_task()
            if self._pending_flush_task is task:
                self._pending_flush_task = None

    async def _push_update(self, text: str, *, force: bool) -> None:
        if not self.card_id:
            return
        if not force and text == self.current_text:
            return
        self.sequence += 1
        await self.client.update_element_content(
            self.card_id,
            CARD_CONTENT_ELEMENT_ID,
            text,
            self.sequence,
        )
        self.current_text = text
        self.pending_text = None
        self.last_update_time = time.monotonic() * 1000
