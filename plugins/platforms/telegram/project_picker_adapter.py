"""Inline project picker transport state and callback handling for the Telegram adapter."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from agent.i18n import t
from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramProjectPickerMixin:
    """Opaque, expiring Telegram project-picker callbacks bound to one sent message."""

    def _project_picker_states(self) -> dict[str, dict]:
        return self.__dict__.setdefault("_project_picker_state", {})

    def _project_picker_keyboard(
        self, token: str, choices: list[dict], controls: list[dict]
    ):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        rows = [
            [
                InlineKeyboardButton(
                    str(item["label"])[:60],
                    callback_data=f"pj:{token}:{item['action']}",
                )
            ]
            for item in choices
        ]
        if controls:
            rows.append(
                [
                    InlineKeyboardButton(
                        str(item["label"]), callback_data=f"pj:{token}:{item['action']}"
                    )
                    for item in controls
                ]
            )
        return InlineKeyboardMarkup(rows)

    async def send_project_picker(
        self,
        *,
        chat_id: str,
        token: str,
        title: str,
        choices: list[dict],
        controls: list[dict],
        metadata: Optional[dict[str, Any]],
        thread_id: Optional[str],
        state: dict,
    ) -> SendResult:
        """Send a picker card; callback_data contains only an opaque nonce and action."""
        keyboard = self._project_picker_keyboard(token, choices, controls)

        def remember(message):
            state["message_id"] = str(message.message_id)
            state["expires_at"] = time.monotonic() + 900
            state["busy"] = False
            states = self._project_picker_states()
            for prior, current in list(states.items()):
                if current.get("expires_at", 0) <= time.monotonic() or all(
                    current.get(key) == state.get(key)
                    for key in ("chat_id", "thread_id", "user_id", "profile_home")
                ):
                    states.pop(prior, None)
            states[token] = state
            while len(states) > 256:
                oldest = min(states, key=lambda key: states[key].get("expires_at", 0))
                states.pop(oldest, None)

        return await self._send_prompt(
            "send_project_picker",
            chat_id,
            metadata,
            lambda: (self.format_message(title), keyboard, remember),
            thread_id=thread_id,
            reply_to_mode=self._reply_to_mode,
        )

    async def edit_project_picker(
        self, query, title: str, choices: list[dict], controls: list[dict]
    ) -> None:
        from telegram.constants import ParseMode

        token = str(query.data).split(":", 2)[1]
        keyboard = self._project_picker_keyboard(token, choices, controls)
        await query.edit_message_text(
            text=self.format_message(title),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard,
        )

    async def _handle_project_picker_callback(
        self, query, data: str, _chat_id: str
    ) -> None:
        """Authenticate and serialize a `pj:<nonce>:<action>` tap before calling its owner."""
        cb = self._callback_ctx(query)
        parts = data.split(":", 2)
        if len(parts) != 3:
            await query.answer(text=t("platform.telegram.picker.invalid_selection"))
            return
        _, token, action = parts
        states = self._project_picker_states()
        state = states.get(token)
        message = getattr(query, "message", None)
        user_id = str(getattr(getattr(query, "from_user", None), "id", ""))
        expired = bool(state and time.monotonic() >= state.get("expires_at", 0))
        valid = bool(
            state
            and not expired
            and not state.get("busy")
            and str(cb.get("chat_id") or "") == state.get("chat_id")
            and str(cb.get("thread_id") or "") == state.get("thread_id", "")
            and user_id == state.get("user_id")
            and str(getattr(message, "message_id", "")) == state.get("message_id")
        )
        if not valid:
            if expired:
                states.pop(token, None)
            await query.answer(text=t("platform.telegram.picker.expired"))
            return
        state["busy"] = True
        toast = ""
        try:
            try:
                result = await state["handler"](query, action, state)
            except Exception:
                logger.exception("Telegram project picker callback failed")
                result = {
                    "text": "Project selection failed. Your conversation was not changed.",
                    "done": False,
                }
            toast = result.get("toast", "")
            if result.get("done"):
                states.pop(token, None)
                await self._edit_result_text(query, result.get("text", ""))
            elif result.get("choices") is not None:
                await self.edit_project_picker(
                    query, result.get("title", ""), result["choices"], result.get("controls", []))
            elif result.get("text"):
                await self._edit_result_text(query, result["text"])
        except Exception:
            logger.exception("Telegram project picker could not update its message")
        finally:
            state["busy"] = False
            await query.answer(text=toast[:200] or None)
