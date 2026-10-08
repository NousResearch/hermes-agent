"""Bot API 10.3 controls bound to one live draft, never a chat-wide /stop."""

import html
import re
from dataclasses import dataclass
from typing import Awaitable, Callable
from agent.i18n import t

from gateway.platforms.helpers import bounded_put
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id


@dataclass
class _Generation:
    draft_id: int
    cancel: Callable[[], Awaitable[None]]
    current: Callable[[], bool]


class TelegramGenerationMixin:
    def supports_thinking_drafts(self):
        return bool(getattr(self, "_rich_messages_enabled", False)
                    and getattr(self, "_rich_drafts_enabled", False)
                    and not getattr(self, "_rich_draft_disabled", False))

    def _thinking_draft_payload(self, content, metadata):
        status = (metadata or {}).get("telegram_thinking_status")
        if not content.strip() or status:
            # This is presentation, not a model reasoning transcript. Never persist the tag.
            label = str(status or t("platform.slack.status.thinking"))[:500]
            suffix = "\n\n<tg-thinking>" + html.escape(label) + "</tg-thinking>"
            return {"markdown": content + suffix}
        return self._rich_message_payload(content)

    @staticmethod
    def _generation_key(chat_id, metadata=None):
        thread = (metadata or {}).get("thread_id")
        thread = None if str(thread) in {"", "1", "None"} else str(thread)
        return (str(normalize_telegram_chat_id(chat_id)), thread)

    def bind_generation_control(self, chat_id, draft_id, metadata, cancel, current):
        if not hasattr(self, "_generation_controls"):
            self._generation_controls = {}
        bounded_put(self._generation_controls, self._generation_key(chat_id, metadata),
                    _Generation(draft_id, cancel, current), 4096)

    def finish_generation_control(self, chat_id, draft_id, metadata=None):
        controls = getattr(self, "_generation_controls", {})
        key = self._generation_key(chat_id, metadata)
        control = controls.get(key)
        if control is not None and control.draft_id == draft_id:
            controls.pop(key, None)

    def _generation_draft_options(self, chat_id, draft_id, metadata=None):
        control = getattr(self, "_generation_controls", {}).get(self._generation_key(chat_id, metadata))
        if control is not None and control.draft_id == draft_id and control.current():
            # Keeping a stopped preview is temporary, not persistence. Never claim final delivery.
            return {"can_stop": True, "keep_on_stop": True}
        return {}

    async def _handle_generation_stopped(self, update, context):
        stopped = getattr(update, "stopped_message_generation", None)
        if stopped is None:
            stopped = (getattr(update, "api_kwargs", None) or {}).get("stopped_message_generation")
        if not stopped:
            return
        if not isinstance(stopped, dict):
            stopped = stopped.to_dict()
        chat = stopped.get("chat")
        if not isinstance(chat, dict) or chat.get("type") != "private":
            return
        chat_id, draft_id = chat.get("id"), stopped.get("draft_id")
        # Telegram's server emits this int64 as a JSON string (td::to_string),
        # retained verbatim in PTB api_kwargs. Keep numeric updates supported too.
        if isinstance(draft_id, str) and re.fullmatch(r"-?[0-9]{1,19}", draft_id):
            draft_id = int(draft_id)
        if type(chat_id) is not int or type(draft_id) is not int or draft_id == 0:
            return
        thread = stopped.get("message_thread_id")
        if thread is not None and type(thread) is not int:
            return
        controls = getattr(self, "_generation_controls", {})
        key = self._generation_key(chat_id, {"thread_id": thread})
        control = controls.get(key)
        if control is None or control.draft_id != draft_id:
            return
        # Consume before awaiting: duplicate updates and an old turn's unwind cannot hit a successor.
        controls.pop(key, None)
        if control.current():
            self._accept_update()
            await control.cancel()
