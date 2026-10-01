"""Native approval prompt rendering and request callbacks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from gateway.platforms.base import BasePlatformAdapter, ExecApprovalPrompt, SendResult


class FeishuApprovalMixin(BasePlatformAdapter):
    if TYPE_CHECKING:
        _EA_CARD_ACTIONS: Any

        _approval_counter: Any

        _approval_state: Dict[int, Dict[str, str]]

        _client: Optional[Any]

        async def _feishu_send_with_retry(self, *, chat_id: str, msg_type: str, payload: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]]) -> Any:
            ...

        def _finalize_send_result(self, response: Any, default_message: str) -> SendResult:
            ...

        def _pop_validated_prompt_state(self, *, states: Dict[int, Dict[str, str]], ident: Any, label: str, open_id: str, chat_id: str, unauthorized_fmt: str, operator_repr: str) -> Optional[Dict[str, str]]:
            ...

        async def send(self, chat_id: str, content: str, reply_to: Optional[str]=..., metadata: Optional[Dict[str, Any]]=...) -> SendResult:
            ...

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        """Approval-button card; ``hermes_action`` in each button value lets the click callback
        route to ``resolve_gateway_approval()`` and unblock the waiting agent thread."""
        from .adapter import SendResult, _card, _card_button, logger, t

        if not self._client:
            return SendResult(success=False, error="Not connected")
        try:
            approval_id = next(self._approval_counter)
            actions = [
                _card_button(label, style or "default",
                             {"hermes_action": self._EA_CARD_ACTIONS[choice], "approval_id": approval_id})
                for label, choice, style in prompt.actions]
            card = _card(f"⚠️ {t('gateway.exec_approval.header')}", "orange", prompt.text, actions=actions)
            return await self._send_interactive_card(
                prompt.chat_id, card, prompt.metadata, "send_exec_approval failed",
                state_map=self._approval_state, state_id=approval_id, session_key=prompt.session_key,
                request_id=prompt.request_id,
            )
        except Exception as exc:
            logger.warning("[Feishu] send_exec_approval failed: %s", exc)
            return SendResult(success=False, error=str(exc))

    async def _send_interactive_card(
        self, chat_id: str, card: Dict[str, Any], metadata: Optional[Dict[str, Any]], failure_message: str, *,
        state_map: Dict[int, Dict[str, str]], state_id: int, session_key: str,
        request_id: Optional[str] = None,
    ) -> SendResult:
        """Send a button card and, on success, remember where it went so a click can be validated."""
        from .adapter import json

        response = await self._feishu_send_with_retry(
            chat_id=chat_id, msg_type="interactive", payload=json.dumps(card, ensure_ascii=False),
            reply_to=None, metadata=metadata,
        )
        result = self._finalize_send_result(response, failure_message)
        if result.success:
            state_map[state_id] = {
                "session_key": session_key,
                "message_id": result.message_id or "",
                "chat_id": chat_id,
                "request_id": request_id or "",
            }
        return result

    async def _resolve_approval(
        self, approval_id: Any, choice: str, user_name: str, *, open_id: str = "", chat_id: str = "",
    ) -> None:
        """Pop approval state and unblock the waiting agent thread."""
        from .adapter import logger, t

        state = self._pop_validated_prompt_state(
            states=self._approval_state, ident=approval_id, label="Approval", open_id=open_id, chat_id=chat_id,
            unauthorized_fmt="[Feishu] Unauthorized approval click by %s for approval %s",
            operator_repr=open_id or "<unknown>",
        )
        if not state:
            return
        try:
            from tools.approval import resolve_gateway_approval
            request_id = state.get("request_id")
            count = (
                resolve_gateway_approval(state["session_key"], choice, request_id=request_id)
                if request_id else 0
            )
            logger.info(
                "Feishu button resolved %d approval(s) for session %s (choice=%s, user=%s)",
                count, state["session_key"], choice, user_name,
            )
            if not count and choice != "deny":
                # The card already reads "Approved" (synchronous callback), but nothing was
                # waiting — the wait timed out (fail-closed deny) or was resolved via /approve.
                # Correct the record so the user doesn't believe the command ran.
                _chat = str(state.get("chat_id", "") or chat_id or "")
                if _chat:
                    try:
                        await self.send(
                            _chat,
                            t("platform.shared.approval_expired"),
                        )
                    except Exception:
                        logger.debug("[Feishu] expired-approval notice failed", exc_info=True)
        except Exception as exc:
            logger.error("Failed to resolve gateway approval from Feishu button: %s", exc)
