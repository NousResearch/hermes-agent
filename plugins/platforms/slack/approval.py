"""Native approval prompt rendering and request callbacks."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, TYPE_CHECKING

from gateway.platforms.base import BasePlatformAdapter, ExecApprovalPrompt, SendResult


class SlackApprovalMixin(BasePlatformAdapter):
    if TYPE_CHECKING:
        _APPROVAL_CHOICES: ClassVar[Dict[str, str]]

        _APPROVAL_DECISION_KEYS: ClassVar[Dict[str, str]]

        _APPROVAL_RESOLVED_MAX: Any

        _EA_ACTION_IDS: Any

        _approval_resolved: Dict[Any, bool]

        async def _begin_interaction(self, ack, body: dict, action: dict, kind: str, *, team_scoped: bool=...) -> Optional[Tuple[str, str, str, dict, str, str, str, str]]:
            ...

        @staticmethod
        def _button(text: str, action_id: str, value: str, *, style: str=..., emoji: bool=...) -> dict:
            ...

        async def _finalize_interactive_message(self, channel_id: str, msg_ts: str, original_text: str, decision_text: str, placeholder: str, label: str, team_id: Optional[str]=..., sanitize: bool=...) -> None:
            ...

        @staticmethod
        def _section_text(message: dict, limit: Optional[int]=...) -> str:
            ...

        async def _send_interactive_prompt(self, chat_id: str, metadata: Optional[Dict[str, Any]], build: Callable[[], Tuple[str, list]], label: str, *, resolved: Optional[Dict[Any, bool]]=..., resolved_max: int=..., team_scoped_key: bool=..., sanitize: bool=...) -> SendResult:
            ...

        @staticmethod
        def _workspace_message_marker(team_id: str, message_id: str) -> Any:
            ...

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        """Block Kit approval prompt; the buttons call ``resolve_gateway_approval()`` to unblock the
        waiting agent thread — same mechanism as the text ``/approve`` flow."""
        from .adapter import Tuple, t


        def _build() -> tuple[str, list]:
            actions = [
                self._button(label, self._EA_ACTION_IDS[choice], prompt.session_key, style=style)
                for label, choice, style in prompt.actions]
            blocks = [
                {"type": "section", "text": {"type": "mrkdwn", "text": prompt.text}},
                {"type": "actions", "elements": actions}]
            return t("platform.slack.approval.fallback_text", command=prompt.command[:100]), blocks

        return await self._send_interactive_prompt(
            prompt.chat_id, prompt.metadata, _build, "send_exec_approval",
            resolved=self._approval_resolved, resolved_max=self._APPROVAL_RESOLVED_MAX)

    async def _handle_approval_action(self, ack, body, action) -> None:
        """Handle an approval button click from Block Kit."""
        from .adapter import logger, t

        started = await self._begin_interaction(ack, body, action, "approval")
        if started is None:
            return
        team_id, action_id, session_key, message, msg_ts, channel_id, user_name, _user_id = started
        choice = self._APPROVAL_CHOICES.get(action_id, "deny")
        # Double-click guard (atomic pop). Also accept the bare ts: the approval may
        # have been stored without a team id while the click carries one.
        approval_key = self._workspace_message_marker(team_id, msg_ts)
        if msg_ts in self._approval_resolved:
            approval_key = msg_ts
        if self._approval_resolved.pop(approval_key, True):
            return
        # Resolve FIRST (unblocks the agent); render after so a click past the
        # timeout (count == 0) shows "expired", not "approved".
        try:
            from tools.approval import resolve_gateway_approval
            count = resolve_gateway_approval(session_key, choice)
            logger.info(
                "Slack button resolved %d approval(s) for session %s (choice=%s, user=%s)", count,
                session_key, choice, user_name)
        except Exception as exc:
            logger.error("Failed to resolve gateway approval from Slack button: %s", exc)
            count = 0
        decision_text = t(self._APPROVAL_DECISION_KEYS[choice], user=user_name)
        if not count:
            decision_text = t("platform.shared.approval_expired")
        await self._finalize_interactive_message(
            channel_id, msg_ts, self._section_text(message), decision_text,
            "Command approval request", "approval", team_id or None)

