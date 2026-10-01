"""Native approval prompt rendering and request callbacks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from gateway.platforms.base import BasePlatformAdapter, ExecApprovalPrompt, SendResult


class TelegramApprovalMixin(BasePlatformAdapter):
    if TYPE_CHECKING:
        _approval_counter: Any

        _approval_state: Dict[int, str]

        async def _claim_callback_state(self, query, cb: Dict[str, Any], state: dict, key, denial: str, resolved: str, *, pop: bool=...):
            ...

        async def _edit_md_quiet(self, query, text_md: str) -> None:
            ...

        @classmethod
        def _metadata_thread_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
            ...

        _reply_to_mode: str

        @staticmethod
        def _rows_of_two(buttons: list) -> list:
            ...

        async def _send_prompt(self, what: str, chat_id: str, metadata: Optional[Dict[str, Any]], build, *, parse_mode: Any=..., thread_id: Any=..., reply_to_mode: Any=...) -> SendResult:
            ...

        resume_typing_for_chat: Any

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        """Inline-keyboard approval prompt; buttons call ``resolve_gateway_approval()`` like the
        text ``/approve`` flow."""
        from .adapter import InlineKeyboardButton, InlineKeyboardMarkup, ParseMode

        def build():
            # Short monotonic ids in callback_data map back to session_key.
            import itertools
            if not hasattr(self, "_approval_counter"):
                self._approval_counter = itertools.count(1)
            approval_id = next(self._approval_counter)
            buttons = [InlineKeyboardButton(label, callback_data=f"ea:{choice}:{approval_id}")
                       for label, choice, _ in prompt.actions]
            return prompt.text, InlineKeyboardMarkup(self._rows_of_two(buttons)), (
                lambda msg: self._approval_state.__setitem__(approval_id, prompt.session_key))
        return await self._send_prompt(
            "send_exec_approval", prompt.chat_id, prompt.metadata, build, parse_mode=ParseMode.HTML,
            thread_id=self._metadata_thread_id(prompt.metadata), reply_to_mode=self._reply_to_mode)

    async def _handle_exec_approval_callback(self, query, data: str, cb: dict[str, Any]) -> None:
        """``ea:<choice>:<approval_id>`` — resolve a pending exec approval."""
        from .adapter import _TOAST_LIMIT, _toast, _unauthorized, logger, t

        parts = data.split(":", 2)
        if len(parts) != 3:
            return
        choice = parts[1]  # once, session, always, deny
        try:
            approval_id = int(parts[2])
        except (ValueError, IndexError):
            await query.answer(text=_toast("platform.telegram.approval.toast_invalid_data"))
            return
        session_key = await self._claim_callback_state(
            query, cb, self._approval_state, approval_id, _unauthorized(),
            _toast("platform.telegram.approval.toast_already_resolved"))
        if not session_key:
            return
        user_display = getattr(query.from_user, "first_name", None) or t("platform.telegram.user_fallback")
        # Resolve FIRST (unblocks the agent thread), render after: a tap landing after the wait timed out
        # (count == 0) must NOT claim "Approved" — the command was already denied.
        try:
            # Rendering happens after so the message reflects what actually occurred: a tap that lands after
            # the approval wait timed out (count == 0) must NOT claim "Approved" — the command was already
            # denied and will not run (#63501 regression follow-up: 60s waits made stale taps common).
            from tools.approval import resolve_gateway_approval
            count = resolve_gateway_approval(session_key, choice)
            logger.info(
                "Telegram button resolved %d approval(s) for session %s (choice=%s, user=%s)", count, session_key, choice, user_display)
        except Exception as exc:
            logger.error("Failed to resolve gateway approval from Telegram button: %s", exc)
            count = 0
        if count:
            label_key = {"once": "resolved_once", "session": "resolved_session", "always": "resolved_always", "deny": "resolved_deny"}.get(
                choice, "resolved_generic")
            label = t(f"platform.telegram.approval.{label_key}")
            edit_text = t("platform.telegram.approval.resolved_by_user", label=label, user=user_display)
        else:
            label = t("platform.telegram.approval.expired")
            edit_text = t("platform.telegram.approval.expired_detail", label=label)
        await query.answer(text=label[:_TOAST_LIMIT])
        await self._edit_md_quiet(query, edit_text)
        # Typing was paused when the approval was sent; the text /approve and /deny paths resume it too.
        if count and cb["chat_id"] is not None:
            self.resume_typing_for_chat(str(cb["chat_id"]))

