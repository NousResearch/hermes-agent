"""Discrete Multi-Bubble Streaming Consumer for Hermes Agent (iMessage / Photon).

Bridges incremental LLM token streaming to discrete message bubbles sent via
PhotonAdapter.send() with pacing, code fence state preservation, and turn-final suppression.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
from typing import Any, Callable, List, Optional

from gateway.bubble_actions import (
    BubbleAction,
    EffectAction,
    FileAttachmentAction,
    ImageAttachmentAction,
    PollAction,
    TapbackAction,
    TextBubble,
)
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
        self.metadata = metadata or {}

        if config is None:
            if hasattr(adapter, "airhermes_config"):
                self.config = adapter.airhermes_config
            elif hasattr(adapter, "photon_config"):
                self.config = adapter.photon_config
            else:
                from gateway.airhermes_config import AirHermesConfig
                self.config = AirHermesConfig.from_dict({})
        elif not hasattr(config, "bubble_min_interval"):
            from gateway.airhermes_config import AirHermesConfig
            self.config = AirHermesConfig.from_dict(config if isinstance(config, dict) else {})
        else:
            self.config = config

        self.stream_deltas_enabled = True
        self._final_content_delivered = False
        self._final_response_sent = False
        self.message_id: Optional[str] = None
        self._turn_split_delivery = True
        self.media_already_delivered = False

        self._parser = BubbleParser(
            paragraph_fallback=True,
            isolate_urls=True,
            config=self.config,
            chat_id=self.chat_id,
            turn_id=self.metadata.get("turn_id"),
        )
        self._bubble_queue = StreamingBubbleQueue(
            chat_id=self.chat_id,
            send_func=self._send_bubble,
            min_interval_s=getattr(self.config, "bubble_min_interval", 0.850),
            adaptive=getattr(self.config, "bubble_adaptive_pacing", True),
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

    async def _send_raw_text_bubble(self, content: str, reply_to: Optional[str] = None) -> Any:
        """Deliver plain text safely without triggering privileged control markers or re-parsing."""
        if hasattr(self.adapter, "send_raw_text"):
            return await self.adapter.send_raw_text(
                self.chat_id,
                content,
                reply_to=reply_to,
                metadata=self.metadata,
            )
        return await self.adapter.send(
            self.chat_id,
            content,
            reply_to=reply_to,
            metadata=self.metadata,
        )

    async def _dispatch_tapback_action(self, item: TapbackAction) -> Any:
        if not getattr(self.config, "enable_tapbacks", True):
            logger.info("[BubbleStreamConsumer] Tapbacks disabled by configuration, ignoring reaction")
            return {"ok": True}
        target_id = item.target_message_id or self.reply_to
        if hasattr(self.adapter, "add_reaction"):
            logger.info(
                "[BubbleStreamConsumer] Delivering tapback reaction %s to chat %s (reply_to=%s)",
                item.reaction, self.chat_id, target_id,
            )
            try:
                await self.adapter.add_reaction(self.chat_id, item.reaction, message_id=target_id)
            except Exception as e:  # health: allow BLE001 -- non-fatal tapback failure
                logger.warning("[BubbleStreamConsumer] add_reaction failed (soft): %s", e, exc_info=True)
        return {"ok": True}

    async def _dispatch_poll_action(self, item: PollAction) -> Any:
        if not getattr(self.config, "enable_polls", True):
            logger.info("[BubbleStreamConsumer] Native polls disabled by configuration")
            opt_lines = "\n".join(f"{idx+1}. {opt}" for idx, opt in enumerate(item.options))
            return await self._send_raw_text_bubble(f"📊 投票: {item.title}\n{opt_lines}", reply_to=self.reply_to)

        if hasattr(self.adapter, "_sidecar_send_poll"):
            logger.info(
                "[BubbleStreamConsumer] Delivering native poll '%s' (%d options) to %s",
                item.title, len(item.options), self.chat_id,
            )
            try:
                res = await self.adapter._sidecar_send_poll(self.chat_id, item.title, list(item.options))
                return {"ok": getattr(res, "success", True), "message_id": getattr(res, "message_id", None)}
            except Exception as e:  # health: allow BLE001 -- non-fatal poll delivery fallback
                logger.warning("[BubbleStreamConsumer] _sidecar_send_poll failed (soft): %s", e, exc_info=True)
                opt_lines = "\n".join(f"{idx+1}. {opt}" for idx, opt in enumerate(item.options))
                return await self._send_raw_text_bubble(f"📊 投票: {item.title}\n{opt_lines}", reply_to=self.reply_to)
        return {"ok": True}

    async def _dispatch_effect_action(self, item: EffectAction) -> Any:
        if not getattr(self.config, "enable_effects", True):
            logger.info("[BubbleStreamConsumer] Visual effects disabled by configuration")
            return await self._send_raw_text_bubble(item.content, reply_to=self.reply_to)

        if hasattr(self.adapter, "_sidecar_send_effect"):
            logger.info("[BubbleStreamConsumer] Delivering effect '%s' to %s", item.effect_type, self.chat_id)
            try:
                res = await self.adapter._sidecar_send_effect(self.chat_id, item.content, item.effect_type)
                return {"ok": getattr(res, "success", True), "message_id": getattr(res, "message_id", None)}
            except Exception as e:  # health: allow BLE001 -- non-fatal effect delivery fallback
                logger.warning("[BubbleStreamConsumer] _sidecar_send_effect failed (soft): %s", e, exc_info=True)
        return await self._send_raw_text_bubble(item.content, reply_to=self.reply_to)

    async def _dispatch_file_attachment_action(self, item: FileAttachmentAction) -> Any:
        if hasattr(self.adapter, "send_document"):
            logger.info(
                "[BubbleStreamConsumer] Delivering authorized file attachment %s to %s",
                item.staged_path, self.chat_id,
            )
            try:
                res = await self.adapter.send_document(
                    self.chat_id,
                    item.staged_path,
                    file_name=item.file_name,
                    caption=item.caption,
                    metadata=self.metadata,
                )
                return {"ok": getattr(res, "success", True), "message_id": getattr(res, "message_id", None)}
            except Exception as e:  # health: allow BLE001 -- non-fatal attachment delivery error
                logger.warning("[BubbleStreamConsumer] send_document failed (soft): %s", e, exc_info=True)
                return {"ok": False, "error": str(e)}
        logger.warning("[BubbleStreamConsumer] Adapter does not support document attachments")
        return {"ok": False, "error": "Adapter does not support document delivery"}

    async def _dispatch_image_attachment_action(self, item: ImageAttachmentAction) -> Any:
        if hasattr(self.adapter, "send_image_file"):
            logger.info(
                "[BubbleStreamConsumer] Delivering authorized image attachment %s to %s",
                item.staged_path, self.chat_id,
            )
            try:
                res = await self.adapter.send_image_file(
                    self.chat_id,
                    item.staged_path,
                    caption=item.caption,
                    metadata=self.metadata,
                )
                return {"ok": getattr(res, "success", True), "message_id": getattr(res, "message_id", None)}
            except Exception as e:  # health: allow BLE001 -- non-fatal attachment delivery error
                logger.warning("[BubbleStreamConsumer] send_image_file failed (soft): %s", e, exc_info=True)
                return {"ok": False, "error": str(e)}
        logger.warning("[BubbleStreamConsumer] Adapter does not support image file attachments")
        return {"ok": False, "error": "Adapter does not support image file delivery"}

    async def _send_bubble(self, item: Any) -> Any:
        """Send an individual typed action or text bubble through the platform adapter."""
        if isinstance(item, str):
            item = TextBubble(content=item)

        if isinstance(item, TapbackAction):
            return await self._dispatch_tapback_action(item)
        if isinstance(item, PollAction):
            return await self._dispatch_poll_action(item)
        if isinstance(item, EffectAction):
            return await self._dispatch_effect_action(item)
        if isinstance(item, FileAttachmentAction):
            return await self._dispatch_file_attachment_action(item)
        if isinstance(item, ImageAttachmentAction):
            return await self._dispatch_image_attachment_action(item)

        content = item.content if isinstance(item, TextBubble) else str(item)
        target_reply = (item.reply_to_id if isinstance(item, TextBubble) and item.reply_to_id else None) or self.reply_to
        return await self._send_raw_text_bubble(content, reply_to=target_reply)

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

        try:
            while True:
                if not self._run_still_current():
                    logger.info("[BubbleStreamConsumer] Turn cancelled or superseded; halting queue")
                    await self._bubble_queue.cancel(reason="turn_superseded")
                    break

                # Non-blocking poll from thread-safe queue with small sleep
                try:
                    item = self._inbound_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.015)  # health: allow HX013 -- consumer queue poll delay
                    continue

                if item is _STREAM_END:
                    # Flush parser and finalize stream
                    rem_bubbles = self._parser.finish_actions()
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
                        try:
                            from gateway.attachment_registry import registry
                            registry.cleanup_turn(self.chat_id, self.metadata.get("turn_id"))
                        except Exception as e:  # health: allow BLE001 -- non-fatal turn cleanup error
                            logger.debug("[BubbleStreamConsumer] Turn cleanup error: %s", e)
                    break

                if isinstance(item, str) and item:
                    new_bubbles = self._parser.feed_actions(item)
                    for b in new_bubbles:
                        await self._bubble_queue.enqueue(b)
                        self._delivered_bubbles.append(b)
        except asyncio.CancelledError:
            logger.info("[BubbleStreamConsumer] Consumer task cancelled; halting owned bubble queue")
            await self._bubble_queue.cancel(reason="consumer_cancelled")
            raise
        finally:
            if not self._bubble_queue._finished_event.is_set():
                with contextlib.suppress(Exception):  # health: allow BLE001 -- consumer exit cancel suppression
                    await self._bubble_queue.cancel(reason="consumer_exited")


__all__ = ["BubbleStreamConsumer"]
