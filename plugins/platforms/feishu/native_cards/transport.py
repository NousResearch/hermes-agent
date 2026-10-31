"""单卡发布与终态收敛。未知远端写入保留身份，不追加猜测性消息。"""

import asyncio
import json
import logging
import time

from gateway.platforms.base import SendResult

from .cardkit_mutations import await_settled, _ambiguous

logger = logging.getLogger(__name__)


class CardTransportMixin:
    def _stream_payload(self, turn):
        with turn.lock:
            text = turn.text
        composed = self._compose_unified_stream_content(
            text, {"__hermes_stream_status": turn.status()}
        )
        return "\n".join(self._split_streaming_card_content(composed))

    async def _patch_card_message(self, message_id, payload):
        request = self._build_patch_message_request(
            message_id=message_id,
            request_body=self._build_patch_message_body(
                content=json.dumps(payload, ensure_ascii=False)
            ),
        )
        response = await self._run_blocking(self._client.im.v1.message.patch, request)
        if not self._response_succeeded(response):
            raise RuntimeError(
                self._response_error_result(response, default_message="卡片更新失败").error
            )

    async def _ensure_card(self, turn):
        if turn.uncertain:
            raise RuntimeError("卡片发送确认未知；拒绝重复发送")
        if turn.message_id:
            return
        if not turn.card_id and not turn.fallback:
            try:
                turn.card_id = await self._create_cardkit_card(self._build_cardkit_initial_card())
            except Exception as exc:
                # CardKit 尚无可见消息，显式 API 失败可退到普通交互卡。
                if _ambiguous(exc):
                    turn.uncertain = True
                    raise
                logger.warning("CardKit unavailable, using an interactive card: %s", exc)
                turn.fallback = True
        payload = (
            json.loads(self._build_streaming_card_payload(self._stream_payload(turn)))
            if turn.fallback
            else {"type": "card", "data": {"card_id": turn.card_id}}
        )
        try:
            response = await self._feishu_send_with_retry(
                chat_id=turn.chat_id,
                msg_type="interactive",
                payload=json.dumps(payload, ensure_ascii=False),
                reply_to=turn.reply_to or None,
                metadata=turn.metadata,
            )
            result = self._finalize_send_result(response, "卡片引用发送失败")
            if not result.success or not result.message_id:
                raise RuntimeError(result.error or "消息发送响应缺少消息身份")
            turn.message_id = str(result.message_id)
        except Exception as exc:
            # 无消息 ID 的超时也可能已送达，不能再创建第二张卡。
            turn.uncertain = True
            turn.error = str(exc)
            raise

    async def _write_card(self, turn, *, finalize=False):
        await self._ensure_card(turn)
        if finalize:
            status = turn.status()
            moa = status.get("orchestration", {})
            if moa:
                moa["phase"] = (
                    "degraded"
                    if any(r["status"] == "failed" for r in moa["references"])
                    else "done"
                )
            payload = self._build_final_cardkit_card(
                turn.text,
                tool_status=status,
                elapsed_seconds=time.monotonic() - turn.started,
                model_name=turn.model,
            )
        else:
            payload = None
        if turn.fallback:
            if payload is None:
                payload = json.loads(self._build_streaming_card_payload(self._stream_payload(turn)))
            await self._patch_card_message(turn.message_id, payload)
            return
        try:
            if finalize:
                await self._update_cardkit_card_with_seq_recovery(turn.card_id, payload)
            else:
                await self._stream_cardkit_content_with_recovery(
                    turn.card_id, self._stream_payload(turn)
                )
        except Exception:
            logger.warning("CardKit write failed; patching the original IM message", exc_info=True)
            # 保留同一 IM 消息；即使原 CardKit 已失效也不能新增结果气泡。
            turn.fallback = True
            if payload is None:
                payload = json.loads(self._build_streaming_card_payload(self._stream_payload(turn)))
            await self._patch_card_message(turn.message_id, payload)

    async def _refresh_turn(self, turn, *, finalize=False):
        async with turn.io_lock:
            if (turn.final or turn.cancelled) and not finalize:
                return SendResult(success=True, message_id=turn.message_id or None)
            try:
                # 保护整个写入及本地身份登记，不取消仍在运行的 SDK 线程。
                await await_settled(self._write_card(turn, finalize=finalize))
                if finalize:
                    turn.final = True
                    if turn.card_id:
                        self._cardkit_terminal.add(turn.card_id)
                turn.error = ""
                return SendResult(success=True, message_id=turn.message_id)
            except asyncio.CancelledError:
                turn.cancelled = True
                if turn.message_id and not turn.final:

                    async def seal_cancelled():
                        with turn.lock:
                            turn.text += "\n\n本轮已中断。"
                        await self._write_card(turn, finalize=True)
                        turn.final = True
                        if turn.card_id:
                            self._cardkit_terminal.add(turn.card_id)

                    try:
                        await await_settled(seal_cancelled())
                    except Exception:
                        logger.warning("Cancelled card could not be sealed", exc_info=True)
                raise
            except Exception as exc:
                turn.error = str(exc)
                logger.warning("Card delivery failed for an anchored turn: %s", exc, exc_info=True)
                return SendResult(success=False, message_id=turn.message_id or None, error=str(exc))

    def schedule_refresh(self, turn):
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def schedule():
            if self._cards_closing or turn.final or turn.cancelled:
                return
            task = self._card_refresh_tasks.get(turn.key)
            if task is not None and not task.done():
                self._card_dirty.add(turn.key)
                return
            task = asyncio.create_task(self._drain_refresh(turn))
            self._card_refresh_tasks[turn.key] = task
            task.add_done_callback(self._refresh_done)

        loop.call_soon_threadsafe(schedule)

    def _refresh_done(self, task):
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error(
                "Card refresh task failed", exc_info=(type(error), error, error.__traceback__)
            )

    async def _drain_refresh(self, turn):
        try:
            while True:
                self._card_dirty.discard(turn.key)
                result = await self._refresh_turn(turn)
                if not result.success or turn.key not in self._card_dirty:
                    return
        finally:
            self._card_refresh_tasks.pop(turn.key, None)

    async def send_draft(self, chat_id, draft_id, content, metadata=None):
        turn = self._find_turn(chat_id, metadata=metadata)
        if turn is None:
            return SendResult(success=False, error="流式消息缺少入站锚点")
        with turn.lock:
            if turn.draft_id not in (None, draft_id):
                return SendResult(success=False, error="拒绝过期流式写入")
            turn.draft_id = draft_id
            if turn.final or turn.cancelled:
                return SendResult(success=True, message_id=turn.message_id or None)
            turn.text = content
        return await self._refresh_turn(turn)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        meta = metadata or {}
        turn = self._find_turn(chat_id, reply_to=reply_to, metadata=meta)
        if turn is None or not (meta.get("notify") or reply_to or meta.get("reply_to_message_id")):
            return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
        if meta.get("_interim_send"):
            if content and not turn.final:
                with turn.lock:
                    turn.text = content
                return await self._refresh_turn(turn)
            return SendResult(success=True, message_id=turn.message_id or None)
        with turn.lock:
            turn.text = content
        return await self._refresh_turn(turn, finalize=not bool(meta.get("expect_edits")))

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        turn = next(
            (
                item
                for item in self._card_turns.values()
                if item.chat_id == chat_id and item.message_id == message_id
            ),
            None,
        )
        if turn is None:
            return await super().edit_message(
                chat_id, message_id, content, finalize=finalize, metadata=metadata
            )
        with turn.lock:
            if turn.final and not finalize:
                return SendResult(success=True, message_id=message_id)
            turn.text = content
        return await self._refresh_turn(turn, finalize=finalize)

    async def abandon_open_draft(self, chat_id, content, metadata=None):
        turn = self._find_turn(chat_id, metadata=metadata)
        if turn is None or turn.final:
            return
        with turn.lock:
            turn.text = (content or turn.text) + "\n\n本轮已中断。"
            turn.cancelled = True
        await self._refresh_turn(turn, finalize=True)

    async def send_busy_ack(self, chat_id, content, *, metadata=None):
        turn = self._find_turn(chat_id, metadata=metadata)
        if turn is None or turn.final:
            return False
        result = await self._send_busy_card(turn, content)
        return bool(result and result.success)
