"""Native approval prompt rendering and request callbacks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from gateway.platforms.base import BasePlatformAdapter, ExecApprovalPrompt, SendResult


class RelayApprovalMixin(BasePlatformAdapter):
    if TYPE_CHECKING:
        _PROMPT_UNAVAILABLE: Any

        async def _mint_and_send_prompt(self, kind: str, state: Dict[str, Any], chat_id: str, *, prompt_kind: str, text: str, options: list, metadata: Optional[Dict[str, Any]]) -> Optional[SendResult]:
            ...

        def _send_lifecycle_ack(self, chat_id: str, text: str, metadata: Dict[str, Any]) -> None:
            ...

        resume_typing_for_chat: Any

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        """Native-button exec approval over the relay (the press resolves via
        tools.approval.resolve_gateway_approval). When the lane is unavailable the send FAILS
        (success=False) so run.py's button→text fallback runs."""
        options = [{"id": choice, "label": label, **({"style": style} if style else {})}
                   for label, choice, style in prompt.actions]
        result = await self._mint_and_send_prompt(
            "exec_approval", {"session_key": prompt.session_key, "request_id": prompt.request_id},
            prompt.chat_id, prompt_kind="approval",
            text=prompt.text, options=options, metadata=prompt.metadata,
        )
        return result if result is not None else self._PROMPT_UNAVAILABLE

    async def _resolve_exec_approval(self, state, option_id, chat_id, ack_meta) -> None:
        from .adapter import _EXEC_APPROVAL_LABELS

        from tools.approval import resolve_gateway_approval

        choice = option_id if option_id in _EXEC_APPROVAL_LABELS else "deny"
        count = (resolve_gateway_approval(str(state.get("session_key") or ""), choice,
                                          request_id=state.get("request_id"))
                 if state.get("request_id") else 0)
        label = _EXEC_APPROVAL_LABELS[choice] if count else "⌛ Approval expired — no command was waiting."
        # In-channel ack preserves the audit trail the native edit gives (the
        # connector's prompt message can't be edited cross-platform yet).
        self._send_lifecycle_ack(chat_id, label, ack_meta)
        if count:
            self.resume_typing_for_chat(chat_id)
