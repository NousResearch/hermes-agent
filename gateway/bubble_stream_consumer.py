"""Discrete Multi-Bubble Streaming Consumer for Hermes Agent (iMessage / Photon).

Bridges incremental LLM token streaming to discrete message bubbles sent via
PhotonAdapter.send() with pacing, code fence state preservation, and turn-final suppression.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
from typing import Any, Callable, List, Optional

from gateway.bubble_parser import BubbleParser
from gateway.bubble_queue import BubbleDeliveryState, StreamingBubbleQueue

logger = logging.getLogger("gateway.bubble_stream_consumer")

_STREAM_END = object()


class BubbleStreamConsumer:
    """Async stream consumer that delivers discrete message bubbles to non-editable platforms."""

    def __init__(
        self,
        adapter: Any,
        chat_id: str,
        reply_to: Optional[str] = None,
        run_still_current: Optional[Callable[[], bool]] = None,
        config: Optional[Any] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.adapter = adapter
        self.chat_id = chat_id
        self.reply_to = reply_to
        self._run_still_current = run_still_current or (lambda: True)
        self.config = config
        self.metadata = metadata or {}

        self.stream_deltas_enabled = True
        self._final_content_delivered = False
        self._final_response_sent = False
        self.message_id: Optional[str] = None
        self._turn_split_delivery = True
        self.media_already_delivered = False

        self._parser = BubbleParser(paragraph_fallback=True, isolate_urls=True)
        self._bubble_queue = StreamingBubbleQueue(
            chat_id=self.chat_id,
            send_func=self._send_bubble,
            min_interval_s=0.850,
            in_flight_timeout_s=12.0,
            typing_func=self._send_typing_heartbeat,
        )
        self._inbound_queue: queue.Queue = queue.Queue()
        self._finished = False
        self._delivered_bubbles: list[str] = []

    @property
    def final_content_delivered(self) -> bool:
        if self._final_content_delivered:
            return True
        return bool(self._bubble_queue and (self._bubble_queue.sent_count > 0 or self._bubble_queue.unknown_delivery_count > 0))

    @final_content_delivered.setter
    def final_content_delivered(self, val: bool) -> None:
        self._final_content_delivered = bool(val)

    @property
    def final_response_sent(self) -> bool:
        if self._final_response_sent:
            return True
        return bool(self._bubble_queue and (self._bubble_queue.sent_count > 0 or self._bubble_queue.unknown_delivery_count > 0))

    @final_response_sent.setter
    def final_response_sent(self, val: bool) -> None:
        self._final_response_sent = bool(val)

    async def _send_typing_heartbeat(self) -> None:
        """Pulse iOS/iMessage typing indicator during inter-bubble cognitive pause."""
        if hasattr(self.adapter, "send_typing"):
            try:
                await self.adapter.send_typing(self.chat_id)
            except (OSError, RuntimeError):  # health: allow S110 -- non-fatal typing heartbeat
                pass

    async def _send_media_bubble(self, text: str) -> dict[str, Any]:
        """Dispatch image or file attachments to platform adapter."""
        if text.startswith("IMAGE:"):
            try:
                data = json.loads(text[len("IMAGE:"):])
                img_path = data.get("path") or ""
                caption = data.get("caption") or ""
                delay = float(data.get("delay", 1.5))
                is_valid_img = await asyncio.to_thread(lambda p: os.path.isfile(p) and os.path.getsize(p) > 500, img_path)
                if hasattr(self.adapter, "send_image_file") and is_valid_img:
                    logger.info("[BubbleStreamConsumer] Delivering image %s to %s", img_path, self.chat_id)
                    res = await self.adapter.send_image_file(self.chat_id, img_path, caption=caption)
                    self.media_already_delivered = True
                    await self._send_typing_heartbeat()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    return {"ok": getattr(res, "success", True), "message_id": getattr(res, "message_id", None)}
                elif hasattr(self.adapter, "send_image") and img_path and (img_path.startswith("http://") or img_path.startswith("https://")):
                    logger.info("[BubbleStreamConsumer] Delivering image URL %s to %s", img_path, self.chat_id)
                    res = await self.adapter.send_image(self.chat_id, img_path, caption=caption)
                    self.media_already_delivered = True
                    await self._send_typing_heartbeat()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    return {"ok": getattr(res, "success", True), "message_id": getattr(res, "message_id", None)}
                else:
                    logger.warning("[BubbleStreamConsumer] Image not found or invalid URL: %s, skipping safely", img_path)
                    return {"ok": True}
            except Exception as e:  # health: allow BLE001 -- non-fatal media send failure
                logger.warning("[BubbleStreamConsumer] Delivering image failed (soft): %s", e, exc_info=True)
            return {"ok": True}

        if text.startswith("FILE:"):
            file_path = text[len("FILE:"):].strip()
            file_exists = await asyncio.to_thread(os.path.exists, file_path)
            if hasattr(self.adapter, "send_document") and file_exists:
                file_name = os.path.basename(file_path)
                logger.info("[BubbleStreamConsumer] Delivering document %s (%s) to %s", file_path, file_name, self.chat_id)
                try:
                    res = await self.adapter.send_document(self.chat_id, file_path, file_name=file_name)
                except TypeError:
                    res = await self.adapter.send_document(self.chat_id, file_path)
                self.media_already_delivered = True
                return {"ok": getattr(res, "success", True), "message_id": getattr(res, "message_id", None)}
        return {"ok": True}

    async def _send_bubble(self, text: str) -> Any:
        """Send an individual bubble or tapback reaction through the platform adapter."""
        if text.startswith("TAPBACK:"):
            emoji = text[len("TAPBACK:"):].strip()
            if hasattr(self.adapter, "add_reaction"):
                logger.info(
                    "[BubbleStreamConsumer] Delivering tapback reaction %s to chat %s (reply_to=%s)",
                    emoji, self.chat_id, self.reply_to,
                )
                try:
                    await self.adapter.add_reaction(self.chat_id, emoji, message_id=self.reply_to)
                except Exception as e:  # health: allow BLE001 -- non-fatal tapback failure
                    logger.warning("[BubbleStreamConsumer] add_reaction failed (soft): %s", e, exc_info=True)
            return {"ok": True}

        if text.startswith("POLL:"):
            import json
            try:
                data = json.loads(text[len("POLL:"):])
                title = data.get("title") or "请选择"
                options = data.get("options") or []
                if hasattr(self.adapter, "_sidecar_send_poll"):
                    logger.info(
                        "[BubbleStreamConsumer] Delivering native poll '%s' (%d options) to %s",
                        title, len(options), self.chat_id,
                    )
                    res = await self.adapter._sidecar_send_poll(self.chat_id, title, options)
                    return {"ok": getattr(res, "success", True), "message_id": getattr(res, "message_id", None)}
            except Exception as e:  # health: allow BLE001 -- non-fatal poll delivery fallback
                logger.warning("[BubbleStreamConsumer] _sidecar_send_poll failed (soft): %s", e, exc_info=True)
            return {"ok": True}

        if text.startswith("EFFECT:"):
            import json
            try:
                data = json.loads(text[len("EFFECT:"):])
                eff = data.get("effect") or "confetti"
                eff_text = data.get("text") or ""
                if hasattr(self.adapter, "_sidecar_send_effect"):
                    logger.info("[BubbleStreamConsumer] Delivering effect '%s' to %s", eff, self.chat_id)
                    res = await self.adapter._sidecar_send_effect(self.chat_id, eff_text, eff)
                    return {"ok": getattr(res, "success", True), "message_id": getattr(res, "message_id", None)}
                return await self.adapter.send(self.chat_id, eff_text, reply_to=self.reply_to, metadata=self.metadata)
            except Exception as e:  # health: allow BLE001 -- non-fatal effect delivery fallback
                logger.warning("[BubbleStreamConsumer] _sidecar_send_effect failed (soft): %s", e, exc_info=True)
            return {"ok": True}

        if text.startswith(("IMAGE:", "FILE:")):
            return await self._send_media_bubble(text)

        return await self.adapter.send(
            self.chat_id,
            text,
            reply_to=self.reply_to,
            metadata=self.metadata,
        )

    def on_delta(self, text: Optional[str]) -> None:
        """Synchronous entry point called by the agent execution thread."""
        if text is not None and not self._finished:
            self._inbound_queue.put(text)

    def on_segment_break(self) -> None:
        """Called between tool executions or turn segment boundaries."""
        # A segment break allows any complete ready bubbles to be flushed
        if not self._finished:
            self._inbound_queue.put("")

    def on_commentary(self, text: str) -> None:
        """Called when interim commentary is produced."""
        if text and not self._finished:
            self._inbound_queue.put(text)

    def finish(self, final_text: Optional[str] = None) -> None:
        """Signal stream completion from agent execution thread (synchronous)."""
        if not self._finished:
            self._finished = True
            self._inbound_queue.put(_STREAM_END)

    def has_durably_delivered_text(self, final_text: str) -> bool:
        """Check for gateway duplicate suppression."""
        return self.final_content_delivered

    def delivered_final_matches(self, final_text: str) -> bool:
        """Check for gateway duplicate suppression."""
        return self.final_content_delivered

    async def run(self) -> None:
        """Async worker loop running on the gateway event loop."""
        self._bubble_queue.start()

        while True:
            if not self._run_still_current():
                logger.info("[BubbleStreamConsumer] Turn cancelled or superseded; halting queue")
                await self._bubble_queue.cancel(reason="turn_superseded")
                break

            # Non-blocking poll from thread-safe queue with small sleep
            try:
                item = self._inbound_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.015)
                continue

            if item is _STREAM_END:
                # Flush parser and finalize stream
                rem_bubbles = self._parser.finish()
                for b in rem_bubbles:
                    await self._bubble_queue.enqueue(b)
                    self._delivered_bubbles.append(b)

                await self._bubble_queue.finish_stream()
                try:
                    await self._bubble_queue.wait_finished(timeout=25.0)
                except Exception as e:  # health: allow BLE001 -- non-fatal queue wait timeout
                    logger.warning("[BubbleStreamConsumer] wait_finished error: %s", e, exc_info=True)
                finally:
                    # Update delivery confirmation flags for turn suppression
                    if self._bubble_queue.sent_count > 0 or self._bubble_queue.unknown_delivery_count > 0:
                        self.final_content_delivered = True
                        self.final_response_sent = True
                        ids = self._bubble_queue.delivered_message_ids
                        if ids:
                            self.message_id = ids[-1]
                        logger.info(
                            "[BubbleStreamConsumer] Stream finished successfully. "
                            "Delivered %d bubbles (unknown=%d, failed=%d) to %s",
                            self._bubble_queue.sent_count,
                            self._bubble_queue.unknown_delivery_count,
                            self._bubble_queue.failed_count,
                            self.chat_id,
                        )
                break

            if isinstance(item, str) and item:
                new_bubbles = self._parser.feed(item)
                for b in new_bubbles:
                    await self._bubble_queue.enqueue(b)
                    self._delivered_bubbles.append(b)


__all__ = ["BubbleStreamConsumer"]
