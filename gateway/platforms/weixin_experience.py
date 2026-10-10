"""Native iLink tool progress, local diagnostics and expired-token request guards."""

from __future__ import annotations

import time
import uuid
import logging
import asyncio

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)


class WeixinSessionPausedError(RuntimeError):
    """A stale bot credential pauses all requests until the cooldown expires or it is replaced."""


class WeixinExperienceMixin:
    native_task_card_task_limit = 0  # Every lifecycle item must reach iLink, including a cleanup drain.

    def _init_weixin_experience(self, extra):
        from gateway.platforms.weixin import _coerce_bool

        self._reply_progress_messages = _coerce_bool(extra.get("reply_progress_messages"), default=True)
        self._weixin_progress = {}
        self._weixin_reply_run_ids = {}
        self._session_pause_until = 0.0
        self._weixin_debug = False
        self._weixin_received_at = {}
        self._weixin_quote_gc_task = None

    async def _weixin_quote_gc(self):
        while self._running and self._quote_store.enabled:
            await asyncio.sleep(300)
            await asyncio.to_thread(self._quote_store.sweep)

    def _pause_weixin_session(self):
        self._session_pause_until = time.monotonic() + 3600

    def _weixin_pause_remaining(self):
        return max(0.0, self._session_pause_until - time.monotonic())

    def _assert_weixin_session_active(self):
        remaining = self._weixin_pause_remaining()
        if remaining:
            raise WeixinSessionPausedError(f"Weixin bot token expired (errcode -14); requests paused for {remaining / 60:.0f} min. Reconnect with hermes gateway setup.")

    def native_task_cards_enabled(self):
        return self._reply_progress_messages

    def format_tool_event(self, event, *, mode="all", preview_max_len=40):
        # iLink uses native tool items; turning them off must not replace them with text bubbles.
        return None

    async def send_native_task_card_progress(self, chat_id, tasks, title=None, reply_to=None, metadata=None, fallback_text=None):
        from gateway.platforms import weixin

        if not self._send_session or not self._token:
            return SendResult(success=False, error="Not connected")
        key = (chat_id, reply_to)
        state = self._weixin_progress.setdefault(key, {"run_id": uuid.uuid4().hex, "tasks": {}})
        self._weixin_reply_run_ids[chat_id] = state["run_id"]
        try:
            for task in tasks:
                identifier, status = task["id"], task["status"]
                if state["tasks"].get(identifier) == status:
                    continue
                started = status == "in_progress"
                tool_name = task["title"].split(" - ", 1)[0]
                payload = {"tool_name": tool_name, "tool_call_id": identifier}
                if not started:
                    payload["status"] = "failed" if status == "error" else "completed" if status == "complete" else "unknown"
                item = {"type": 11 if started else 12, "create_time_ms": int(time.time() * 1000),
                        "is_completed": not started, "tool_call_start_item" if started else "tool_call_result_item": payload}
                response = await weixin._send_items(
                    self._send_session, base_url=self._base_url, token=self._token, to=chat_id, item_list=[item],
                    context_token=self._token_store.get(self._account_id, chat_id), client_id=f"hermes-weixin-{uuid.uuid4().hex}",
                    run_id=state["run_id"],
                )
                if response.get("ret") not in (None, 0) or response.get("errcode") not in (None, 0):
                    raise RuntimeError(f"iLink tool progress failed: ret={response.get('ret')} errcode={response.get('errcode')}")
                state["tasks"][identifier] = status
            return SendResult(success=True)
        except Exception as exc:
            # A progress failure must not prevent the normal answer; the shared rail uses text fallback.
            logger.warning("Weixin native tool progress failed", exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def stop_native_task_card_progress(self, chat_id, reply_to=None, metadata=None):
        self._weixin_progress.pop((chat_id, reply_to), None)

    async def handle_local_command(self, event):
        """Called only after the gateway's canonical authorization gate, including pairing."""
        command, _, args = event.text.strip().partition(" ")
        handlers = {"/echo": self._weixin_echo, "/toggle-debug": self._weixin_toggle_debug}
        handler = handlers.get(command.lower())
        if handler is None:
            return False
        await handler(event, args)
        return True

    async def _weixin_echo(self, event, args):
        started = time.time()
        if args.strip():
            await self.send(event.source.chat_id, args.strip())
        timestamp = (event.raw_message or {}).get("create_time_ms")
        delay = f"{int(started * 1000) - timestamp}ms" if isinstance(timestamp, (int, float)) else "N/A"
        await self.send(event.source.chat_id, f"⏱ 通道耗时\n├ 平台→Hermes: {delay}\n└ Hermes 处理: {int((time.time() - started) * 1000)}ms")

    async def _weixin_toggle_debug(self, event, args):
        self._weixin_debug = not self._weixin_debug
        await self.send(event.source.chat_id, "Debug 模式已开启" if self._weixin_debug else "Debug 模式已关闭")
