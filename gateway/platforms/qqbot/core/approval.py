# -*- coding: utf-8 -*-
"""QQ Bot approval message sender.

Builds and sends interactive approval messages with inline keyboard buttons
to QQ C2C users or group chats.

Approval flow::

    1. hermes tools/approval.py blocks the agent thread, waiting for a decision.
    2. gateway/run.py calls adapter.send_approval_request() which delegates here.
    3. ApprovalSender sends a markdown message with an InlineKeyboard:
         [✅ 允许一次]  [⭐ 始终允许]  [❌ 拒绝]
    4. User clicks a button → INTERACTION_CREATE dispatched to QQWebSocket.
    5. adapter._on_interaction() ACKs the interaction, parses button_data,
       and calls tools.approval.resolve_gateway_approval().

button_data format (mirrors TS approval-handler.ts)::

    approve:<session_key>:<decision>
    decision: allow-once | allow-always | deny

Zero hermes dependencies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .api_client import QQApiClient
from .dto import (
    InlineKeyboard,
    KeyboardButton,
    KeyboardButtonAction,
    KeyboardButtonPermission,
    KeyboardButtonRenderData,
    KeyboardContent,
    KeyboardRow,
)

logger = logging.getLogger(__name__)

# button_data prefix used to identify approval button clicks
APPROVAL_BUTTON_PREFIX = "approve:"

# button_data prefix for update-prompt Yes/No clicks
UPDATE_PROMPT_PREFIX = "update_prompt:"

# Pattern: approve:<session_key>:<decision>
# session_key may contain colons (e.g. agent:main:qqbot:c2c:OPENID)
_APPROVAL_DATA_RE = re.compile(
    r"^approve:(.+):(allow-once|allow-session|allow-always|deny)$"
)

# Pattern: update_prompt:<answer>  (answer = y | n)
_UPDATE_PROMPT_RE = re.compile(r"^update_prompt:(y|n)$")


@dataclass
class ApprovalRequest:
    """Structured approval request for rendering.

    :param session_key: Hermes session key used to route the decision.
    :param title: Short title shown at the top of the message.
    :param description: Optional longer description.
    :param command_preview: Command text to display (exec approvals).
    :param cwd: Working directory (exec approvals).
    :param tool_name: Tool name (plugin approvals).
    :param severity: 'critical' | 'info' | other (affects icon).
    :param timeout_sec: Seconds until the approval expires.
    """

    session_key: str
    title: str
    description: str = ""
    command_preview: str = ""
    cwd: str = ""
    tool_name: str = ""
    severity: str = ""
    timeout_sec: int = 120


def parse_approval_button_data(
    button_data: str,
) -> Optional[tuple[str, str]]:
    """Parse approval button_data into ``(session_key, decision)``.

    :param button_data: Raw ``data.resolved.button_data`` from INTERACTION_CREATE.
    :returns: ``(session_key, decision)`` tuple, or ``None`` if not an approval button.
    """
    m = _APPROVAL_DATA_RE.match(button_data)
    if not m:
        return None
    return m.group(1), m.group(2)


def parse_update_prompt_button_data(button_data: str) -> Optional[str]:
    """Parse update-prompt button_data into answer (``'y'`` or ``'n'``).

    :param button_data: Raw ``data.resolved.button_data`` from INTERACTION_CREATE.
    :returns: ``'y'`` or ``'n'``, or ``None`` if not an update-prompt button.
    """
    m = _UPDATE_PROMPT_RE.match(button_data)
    if not m:
        return None
    return m.group(1)


def build_update_prompt_keyboard() -> InlineKeyboard:
    """Build a Yes/No inline keyboard for a hermes update confirmation.

    button_data format: ``update_prompt:<answer>``  (answer = y | n)

    Buttons share ``group_id="update_prompt"`` so clicking one greys out the other.

    :returns: :class:`~dto.InlineKeyboard` ready to send.
    """
    def _make_button(btn_id: str, label: str, visited_label: str, answer: str, style: int) -> KeyboardButton:
        return KeyboardButton(
            id=btn_id,
            render_data=KeyboardButtonRenderData(
                label=label,
                visited_label=visited_label,
                style=style,
            ),
            action=KeyboardButtonAction(
                type=1,
                data=f"{UPDATE_PROMPT_PREFIX}{answer}",
                permission=KeyboardButtonPermission(type=2),
                click_limit=1,
            ),
            group_id="update_prompt",
        )

    return InlineKeyboard(
        content=KeyboardContent(
            rows=[
                KeyboardRow(buttons=[
                    _make_button("yes", "✓ 确认", "已确认", "y", 1),
                    _make_button("no",  "✗ 取消", "已取消", "n", 0),
                ]),
            ]
        )
    )


def build_approval_keyboard(session_key: str) -> InlineKeyboard:
    """Build a 3-button inline keyboard for an approval request.

    button_data format: ``approve:<session_key>:<decision>``

    Buttons share ``group_id="approval"`` so clicking one greys out the rest.

    Row 1: [✅ 允许一次]  [⭐ 始终允许]  [❌ 拒绝]

    :param session_key: Hermes session key embedded in the button payload.
    :returns: :class:`~dto.InlineKeyboard` ready to send.
    """
    def _make_button(
        btn_id: str,
        label: str,
        visited_label: str,
        decision: str,
        style: int,
    ) -> KeyboardButton:
        return KeyboardButton(
            id=btn_id,
            render_data=KeyboardButtonRenderData(
                label=label,
                visited_label=visited_label,
                style=style,
            ),
            action=KeyboardButtonAction(
                type=1,  # Callback → triggers INTERACTION_CREATE
                data=f"{APPROVAL_BUTTON_PREFIX}{session_key}:{decision}",
                permission=KeyboardButtonPermission(type=2),
                click_limit=1,
            ),
            group_id="approval",
        )

    return InlineKeyboard(
        content=KeyboardContent(
            rows=[
                KeyboardRow(buttons=[
                    _make_button("allow",  "✅ 允许一次",  "已允许",     "allow-once",   1),
                    _make_button("always", "⭐ 始终允许",  "已始终允许", "allow-always", 1),
                    _make_button("deny",   "❌ 拒绝",      "已拒绝",     "deny",         0),
                ]),
            ]
        )
    )


def build_approval_text(req: ApprovalRequest) -> str:
    """Build the markdown message text for an approval request.

    :param req: :class:`ApprovalRequest` with display data.
    :returns: Markdown-formatted string.
    """
    if req.command_preview or req.cwd:
        return _build_exec_text(req)
    return _build_plugin_text(req)


def _build_exec_text(req: ApprovalRequest) -> str:
    lines: List[str] = ["🔐 **命令执行审批**", ""]
    if req.command_preview:
        preview = req.command_preview[:300]
        lines.append(f"```\n{preview}\n```")
    if req.cwd:
        lines.append(f"📁 目录: {req.cwd}")
    if req.title and req.title != req.command_preview:
        lines.append(f"📋 {req.title}")
    if req.description:
        lines.append(f"📝 {req.description}")
    lines.append("")
    lines.append(f"⏱️ 超时: {req.timeout_sec} 秒")
    return "\n".join(lines)


def _build_plugin_text(req: ApprovalRequest) -> str:
    severity_icon = (
        "🔴" if req.severity == "critical"
        else "🔵" if req.severity == "info"
        else "🟡"
    )
    lines: List[str] = [f"{severity_icon} **审批请求**", ""]
    lines.append(f"📋 {req.title}")
    if req.description:
        lines.append(f"📝 {req.description}")
    if req.tool_name:
        lines.append(f"🔧 工具: {req.tool_name}")
    lines.append("")
    lines.append(f"⏱️ 超时: {req.timeout_sec} 秒")
    return "\n".join(lines)


class ApprovalSender:
    """Send approval messages with inline keyboard to QQ chats.

    :param api: Authenticated :class:`~api_client.QQApiClient`.
    :param log_tag: Log prefix.
    """

    def __init__(self, api: QQApiClient, log_tag: str = "QQBot") -> None:
        self._api = api
        self._log_tag = log_tag

    async def send(
        self,
        chat_type: str,
        chat_id: str,
        req: ApprovalRequest,
        msg_id: Optional[str] = None,
    ) -> bool:
        """Send an approval message with buttons to the specified chat.

        :param chat_type: ``'c2c'`` or ``'group'``.
        :param chat_id: User openid or group openid.
        :param req: :class:`ApprovalRequest` with display data.
        :param msg_id: Reply-to message ID (passive message context).
        :returns: ``True`` on success, ``False`` on failure.
        """
        text = build_approval_text(req)
        keyboard = build_approval_keyboard(req.session_key)

        logger.info(
            "[%s] Sending approval request to %s:%s (session=%s)",
            self._log_tag, chat_type, chat_id, req.session_key[:20],
        )

        msg = self._api.build_text_body(text, reply_to=msg_id, markdown=True)
        try:
            if chat_type == "c2c":
                await self._api.post_c2c_message(chat_id, msg, keyboard=keyboard)
            elif chat_type == "group":
                await self._api.post_group_message(chat_id, msg, keyboard=keyboard)
            else:
                logger.warning(
                    "[%s] Approval: unsupported chat_type %r", self._log_tag, chat_type
                )
                return False
            logger.info(
                "[%s] Approval message sent to %s:%s", self._log_tag, chat_type, chat_id
            )
            return True
        except Exception as exc:
            logger.error(
                "[%s] Failed to send approval message to %s:%s: %s",
                self._log_tag, chat_type, chat_id, exc,
            )
            return False
