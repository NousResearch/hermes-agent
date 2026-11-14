"""原版卡片适配层；只通过子类与插件钩子扩展官方适配器。"""

import asyncio
from collections import OrderedDict
from functools import wraps
from dataclasses import replace
import hashlib
import logging
import threading

from .busy import BusyCardMixin
from .cardkit import CardKitMixin
from .clarify import FeishuClarifyMixin
from .prompts import PromptMixin
from .notices import NoticeCardMixin
from .rendering import CardRenderer
from .state import CardTurn
from .sending import MessageSendMixin
from .transport import CardTransportMixin
from .webhook import WebhookMixin

logger = logging.getLogger(__name__)


def build_adapter_class(base, hub):
    class FeishuCardsAdapter(
        BusyCardMixin,
        CardTransportMixin,
        NoticeCardMixin,
        PromptMixin,
        FeishuClarifyMixin,
        WebhookMixin,
        MessageSendMixin,
        CardRenderer,
        CardKitMixin,
        base,
    ):
        draft_stream_is_message = True

        def __init__(self, config):
            super().__init__(config)
            self._init_cardkit()
            self._hub = hub
            self._card_turns = OrderedDict()
            self._notice_cards = OrderedDict()
            self._card_refresh_tasks = {}
            self._card_dirty = set()
            self._cards_closing = False
            self._card_action_lock = threading.RLock()
            self._clarify_state = {}
            self._clarify_tasks = {}

        def supports_draft_streaming(self, chat_type=None, metadata=None, chat_id=None):
            return True

        def streaming_overflow_limit(self):
            return self.MAX_STREAMING_CARD_CONTENT_LENGTH + 10_000

        def message_len_fn_for_chat(self, chat_id):
            # 计量实际有界卡片的可见内容；超长输出由渲染器保留头尾并明确提示，
            # 不让共享消费者先把完整文本拆成多个独立消息。
            return lambda text: sum(
                len(chunk) for chunk in self._split_streaming_card_content(text)
            )

        async def connect(self, *, is_reconnect=False):
            self._cards_closing = False
            return await super().connect(is_reconnect=is_reconnect)

        def _build_lark_client(self, domain):
            import lark_oapi as lark

            return (
                lark.Client.builder()
                .app_id(self._app_id)
                .app_secret(self._app_secret)
                .domain(domain)
                .timeout(15)
                .log_level(lark.LogLevel.WARNING)
                .build()
            )

        def _find_turn(self, chat_id, *, reply_to=None, metadata=None):
            meta = metadata or {}
            anchor = str(meta.get("reply_to_message_id") or reply_to or "")
            key = (str(chat_id), str(meta.get("thread_id") or ""), anchor)
            if anchor:
                return self._card_turns.get(key)
            # 仅接受同一执行上下文，不按“唯一活跃会话”猜测。
            bound = hub.current.get()
            if bound is not None and bound[0] is self:
                turn = bound[1]
                if turn.chat_id == chat_id and turn.thread_id == key[1]:
                    return turn
            return None

        def _turn_for_event(self, event):
            source = event.source
            anchor = str(event.message_id or "")
            if not anchor or str(event.text or "").lstrip().startswith("/"):
                return None
            key = (str(source.chat_id), str(source.thread_id or ""), anchor)
            turn = self._card_turns.get(key)
            if turn is None:
                owners = tuple(
                    dict.fromkeys(
                        str(value)
                        for value in (source.user_id, getattr(source, "user_id_alt", None))
                        if value
                    )
                )
                turn = CardTurn(
                    *key,
                    owner_ids=owners,
                    chat_type=source.chat_type,
                    session_key=self._event_session_key(event),
                    metadata={"thread_id": source.thread_id} if source.thread_id else {},
                )
                self._card_turns[key] = turn
            # 不驱逐仍在运行的卡片或远端结果未知的卡片。
            while len(self._card_turns) > 512:
                expired = next(
                    (
                        k
                        for k, v in self._card_turns.items()
                        if v.final and k not in self._card_refresh_tasks
                    ),
                    None,
                )
                if expired is None:
                    break
                self._card_turns.pop(expired)
            return turn

        def set_message_handler(self, handler):
            @wraps(handler)
            async def bound_handler(event):
                turn = self._turn_for_event(event)
                token = hub.current.set((self, turn) if turn is not None else None)
                try:
                    return await handler(event)
                finally:
                    hub.current.reset(token)

            # base 会读取 handler.__self__ 获取官方 runner 的会话/历史路由。
            if hasattr(handler, "__self__"):
                bound_handler.__self__ = handler.__self__
            super().set_message_handler(bound_handler)

        async def handle_message(self, event):
            # 官方在 handler 之前消费文字澄清答复：在该入口保留原生卡片的答题人绑定。
            self._canonicalize(event.source)
            session_key = self._event_session_key(event)
            for clarify_id, state in list(self._clarify_state.items()):
                if state["session_key"] == session_key and not self.can_answer_clarify(
                    clarify_id, event.source
                ):
                    return
            return await super().handle_message(event)

        def render_message_event(self, event, sink):
            from gateway.stream_events import Commentary

            turn = self._find_turn(sink.chat_id, metadata=sink.metadata)
            if turn is not None and isinstance(event, Commentary):
                with turn.lock:
                    turn.text = event.text
                self.schedule_refresh(turn)
                return
            return super().render_message_event(event, sink)

        def format_tool_event(self, event, *, mode="all", preview_max_len=40):
            if hub.current.get() is not None:
                return None  # 结构化 pre/post_tool_call 驱动单卡进度，不制造独立工具气泡。
            return super().format_tool_event(event, mode=mode, preview_max_len=preview_max_len)

        def _prompt_metadata(self, metadata):
            meta = dict(metadata or {})
            bound = hub.current.get()
            if bound is not None and bound[0] is self:
                turn = bound[1]
                meta.setdefault("__hermes_prompt_user_ids", turn.owner_ids)
                meta.setdefault("__hermes_prompt_chat_type", turn.chat_type)
            return meta

        async def _send_interactive_card(self, chat_id, card, metadata, failure_message, **kwargs):
            from gateway.platforms.base import SendResult

            meta = self._prompt_metadata(metadata)
            if not meta.get("__hermes_prompt_user_ids"):
                return SendResult(success=False, error="交互卡片缺少发起人绑定")
            if kwargs.get("state_map") is self._approval_state and not meta.get(
                "__hermes_approval_request_id"
            ):
                return SendResult(success=False, error="审批请求身份未知，拒绝创建可操作卡片")
            from .cardkit_mutations import await_settled

            return await await_settled(
                super()._send_interactive_card(chat_id, card, meta, failure_message, **kwargs)
            )

        async def _send_exec_approval_prompt(self, prompt):
            from gateway.platforms.base import SendResult
            from tools.approval import list_gateway_approvals

            meta = self._prompt_metadata(prompt.metadata)
            bound = hub.current.get()
            if (
                not meta.get("__hermes_approval_request_id")
                and bound is not None
                and bound[0] is self
            ):
                turn = bound[1]
                pending = {
                    item.get("request_id") for item in list_gateway_approvals(prompt.session_key)
                }
                digest = hashlib.sha256(prompt.command.encode()).hexdigest()
                with turn.lock:
                    candidates = [
                        (key, data)
                        for key, data in turn.approvals.items()
                        if key in pending and data[0] == prompt.session_key
                    ]
                exact = [key for key, data in candidates if data[1] == digest]
                selected = exact if exact else [key for key, data in candidates]
                if len(selected) == 1:
                    meta["__hermes_approval_request_id"] = selected[0]
            if not meta.get("__hermes_approval_request_id"):
                return SendResult(success=False, error="无法唯一确定当前审批请求")
            return await super()._send_exec_approval_prompt(replace(prompt, metadata=meta))

        async def send_clarify(
            self, chat_id, question, choices, clarify_id, session_key, metadata=None
        ):
            return await super().send_clarify(
                chat_id, question, choices, clarify_id, session_key, self._prompt_metadata(metadata)
            )

        async def disconnect(self):
            self._cards_closing = True
            tasks = list(self._card_refresh_tasks.values()) + list(self._clarify_tasks.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            for clarify_id in list(self._clarify_state):
                try:
                    await self.retire_clarify_card(clarify_id, "网关已停止，此确认已失效。")
                except Exception:
                    logger.warning(
                        "Clarify card retirement during disconnect failed", exc_info=True
                    )
            for turn in list(self._card_turns.values()):
                if turn.message_id and not turn.final:
                    await self.abandon_open_draft(
                        turn.chat_id,
                        turn.text,
                        metadata={**turn.metadata, "reply_to_message_id": turn.reply_to},
                    )
            self._clarify_state.clear()
            await super().disconnect()

    return FeishuCardsAdapter
