"""Native approval prompt rendering and request callbacks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.platforms.base import ExecApprovalPrompt, SendResult


class BaseApprovalPromptMixin:
    if TYPE_CHECKING:
        def _exec_approval_actions(self, *, allow_permanent: bool, allow_session: bool, smart_denied: bool) -> List[Tuple[str, str, str]]:
            ...

        def _format_exec_approval(self, command: str, description: Optional[str]=..., smart_denied: bool=...) -> str:
            ...

        async def _send_exec_approval_prompt(self, prompt: 'ExecApprovalPrompt') -> SendResult:
            ...

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str, description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, allow_permanent: bool = True, allow_session: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Interactive exec-approval prompt; a press resolves via
        ``tools.approval.resolve_gateway_approval``. Text and choice set are shared; adapters
        render them natively in ``_send_exec_approval_prompt``."""
        from gateway.platforms.base import ExecApprovalPrompt, ea_default_reason_text

        if description is None:
            description = ea_default_reason_text()
        prompt = ExecApprovalPrompt(
            chat_id=chat_id, session_key=session_key, metadata=metadata, command=str(command or ""),
            description=description, smart_denied=smart_denied,
            text=self._format_exec_approval(command, description, smart_denied),
            actions=self._exec_approval_actions(
                allow_permanent=allow_permanent, allow_session=allow_session, smart_denied=smart_denied))
        return await self._send_exec_approval_prompt(prompt)

