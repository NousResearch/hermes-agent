"""QQ Bot inline keyboards + approval / update-prompt senders.

QQ Bot v2 supports attaching inline keyboards to outbound messages. When a
user clicks a button, the platform dispatches an ``INTERACTION_CREATE``
gateway event containing the button's ``data`` payload. The bot must ACK the
interaction promptly via ``PUT /interactions/{id}`` or the user sees an
error indicator on the button.

This module provides:

- :class:`InlineKeyboard` + button dataclasses — serialized into the
  ``keyboard`` field of the outbound message body.
- :func:`build_approval_keyboard` — 3-button ✅ once / ⭐ always / ❌ deny
  keyboard for tool-approval flows.
- :func:`build_update_prompt_keyboard` — Yes/No keyboard for update confirms.
- :func:`parse_approval_button_data` / :func:`parse_update_prompt_button_data`
  — decode the ``button_data`` payload from ``INTERACTION_CREATE``.
- :class:`ApprovalRequest` + :class:`ApprovalSender` — high-level helper that
  builds an approval message with keyboard and posts it to a c2c / group chat.

``button_data`` formats::

    approve:<session_key>:<decision>      # decision = allow-once|allow-always|deny
    update_prompt:<answer>                # answer = y|n

Ported from WideLee's qqbot-agent-sdk v1.2.2 (``approval.py`` + ``dto.py``
keyboard types). Authorship preserved via Co-authored-by.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── button_data prefixes + patterns ──────────────────────────────────

APPROVAL_BUTTON_PREFIX = "approve:"
UPDATE_PROMPT_PREFIX = "update_prompt:"

# Pattern: approve:<session_key>:<decision>
# session_key may itself contain colons (e.g. agent:main:qqbot:c2c:OPENID),
# so the session_key group is greedy but trails the decision.
_APPROVAL_DATA_RE = re.compile(
    r"^approve:(.+):(allow-once|allow-always|deny)$"
)

# Pattern: update_prompt:y | update_prompt:n
_UPDATE_PROMPT_RE = re.compile(r"^update_prompt:(y|n)$")


# ── Keyboard dataclasses ─────────────────────────────────────────────

@dataclass
class KeyboardButtonPermission:
    """Button permission metadata. ``type=2`` means all users can click."""
    type: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type}


@dataclass
class KeyboardButtonAction:
    """What happens when the button is clicked.

    :param type: ``1`` (Callback — triggers ``INTERACTION_CREATE``) or
        ``2`` (Link — opens a URL).
    :param data: Payload delivered in ``data.resolved.button_data`` when
        ``type=1``.
    :param permission: :class:`KeyboardButtonPermission`.
    :param click_limit: Max clicks per user (``1`` = single-use).
    """
    type: int
    data: str
    permission: KeyboardButtonPermission = field(
        default_factory=KeyboardButtonPermission
    )
    click_limit: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "data": self.data,
            "permission": self.permission.to_dict(),
            "click_limit": self.click_limit,
        }


@dataclass
class KeyboardButtonRenderData:
    """Visual rendering of a button.

    :param label: Pre-click label.
    :param visited_label: Post-click label (button stays greyed in place).
    :param style: ``0`` = grey, ``1`` = blue.
    """
    label: str
    visited_label: str
    style: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "visited_label": self.visited_label,
            "style": self.style,
        }


@dataclass
class KeyboardButton:
    """One button in a keyboard.

    :param group_id: Buttons sharing a ``group_id`` are mutually exclusive —
        clicking one greys the rest.
    """
    id: str
    render_data: KeyboardButtonRenderData
    action: KeyboardButtonAction
    group_id: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "render_data": self.render_data.to_dict(),
            "action": self.action.to_dict(),
            "group_id": self.group_id,
        }


@dataclass
class KeyboardRow:
    buttons: List[KeyboardButton] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"buttons": [b.to_dict() for b in self.buttons]}


@dataclass
class KeyboardContent:
    rows: List[KeyboardRow] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": [r.to_dict() for r in self.rows]}


@dataclass
class InlineKeyboard:
    """Top-level keyboard payload — goes into ``MessageToCreate.keyboard``."""
    content: KeyboardContent = field(default_factory=KeyboardContent)

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content.to_dict()}


# ── INTERACTION_CREATE parsing ───────────────────────────────────────

def parse_approval_button_data(button_data: str) -> Optional[tuple[str, str]]:
    """Parse approval ``button_data`` into ``(session_key, decision)``.

    :param button_data: Raw ``data.resolved.button_data`` from
        ``INTERACTION_CREATE``.
    :returns: ``(session_key, decision)`` or ``None`` if not an approval button.
    """
    m = _APPROVAL_DATA_RE.match(button_data or "")
    if not m:
        return None
    return m.group(1), m.group(2)


def parse_update_prompt_button_data(button_data: str) -> Optional[str]:
    """Parse update-prompt ``button_data`` into ``'y'`` or ``'n'``."""
    m = _UPDATE_PROMPT_RE.match(button_data or "")
    if not m:
        return None
    return m.group(1)


# ── Keyboard builders ────────────────────────────────────────────────

def _make_callback_button(
    btn_id: str,
    label: str,
    visited_label: str,
    data: str,
    style: int,
    group_id: str,
) -> KeyboardButton:
    return KeyboardButton(
        id=btn_id,
        render_data=KeyboardButtonRenderData(
            label=label,
            visited_label=visited_label,
            style=style,
        ),
        action=KeyboardButtonAction(type=1, data=data),
        group_id=group_id,
    )


def build_approval_keyboard(session_key: str) -> InlineKeyboard:
    """Build the 3-button approval keyboard.

    Layout: ``[✅ 允许一次] [⭐ 始终允许] [❌ 拒绝]`` — all three share
    ``group_id='approval'`` so clicking one greys out the rest.

    :param session_key: Embedded into ``button_data`` so the decision
        routes back to the right pending approval.
    """
    return InlineKeyboard(
        content=KeyboardContent(
            rows=[
                KeyboardRow(buttons=[
                    _make_callback_button(
                        btn_id="allow",
                        label="✅ 允许一次",
                        visited_label="已允许",
                        data=f"{APPROVAL_BUTTON_PREFIX}{session_key}:allow-once",
                        style=1,
                        group_id="approval",
                    ),
                    _make_callback_button(
                        btn_id="always",
                        label="⭐ 始终允许",
                        visited_label="已始终允许",
                        data=f"{APPROVAL_BUTTON_PREFIX}{session_key}:allow-always",
                        style=1,
                        group_id="approval",
                    ),
                    _make_callback_button(
                        btn_id="deny",
                        label="❌ 拒绝",
                        visited_label="已拒绝",
                        data=f"{APPROVAL_BUTTON_PREFIX}{session_key}:deny",
                        style=0,
                        group_id="approval",
                    ),
                ]),
            ]
        )
    )


def build_update_prompt_keyboard() -> InlineKeyboard:
    """Build a Yes/No keyboard for update confirmation prompts."""
    return InlineKeyboard(
        content=KeyboardContent(
            rows=[
                KeyboardRow(buttons=[
                    _make_callback_button(
                        btn_id="yes",
                        label="✓ 确认",
                        visited_label="已确认",
                        data=f"{UPDATE_PROMPT_PREFIX}y",
                        style=1,
                        group_id="update_prompt",
                    ),
                    _make_callback_button(
                        btn_id="no",
                        label="✗ 取消",
                        visited_label="已取消",
                        data=f"{UPDATE_PROMPT_PREFIX}n",
                        style=0,
                        group_id="update_prompt",
                    ),
                ]),
            ]
        )
    )


# ── ApprovalRequest + text builder ───────────────────────────────────

@dataclass
class ApprovalRequest:
    """Structured approval-request display data.

    :param session_key: Routes the decision back to the waiting caller.
    :param title: Short title at the top.
    :param description: Optional longer description.
    :param command_preview: Command text (exec approvals).
    :param cwd: Working directory (exec approvals).
    :param tool_name: Tool name (plugin approvals).
    :param severity: ``'critical' | 'info' | ''``.
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


def build_approval_text(req: ApprovalRequest) -> str:
    """Render an :class:`ApprovalRequest` into the message body (markdown)."""
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
    icon = (
        "🔴" if req.severity == "critical"
        else "🔵" if req.severity == "info"
        else "🟡"
    )
    lines: List[str] = [f"{icon} **审批请求**", ""]
    lines.append(f"📋 {req.title}")
    if req.description:
        lines.append(f"📝 {req.description}")
    if req.tool_name:
        lines.append(f"🔧 工具: {req.tool_name}")
    lines.append("")
    lines.append(f"⏱️ 超时: {req.timeout_sec} 秒")
    return "\n".join(lines)


# ── ApprovalSender ───────────────────────────────────────────────────

PostMessageFn = Callable[..., Awaitable[Dict[str, Any]]]
"""Signature of an async POST to ``/v2/{users|groups}/{id}/messages``.

Implementations accept a body dict and return the raw API response.
"""


class ApprovalSender:
    """Send an approval-request message with an inline keyboard.

    Decoupled from the adapter via callables so it can be unit-tested in
    isolation. Pass the adapter's ``_send_message_with_keyboard`` helper
    (or any equivalent) as ``post_message``.
    """

    def __init__(
        self,
        post_c2c: PostMessageFn,
        post_group: PostMessageFn,
        log_tag: str = "QQBot",
    ) -> None:
        self._post_c2c = post_c2c
        self._post_group = post_group
        self._log_tag = log_tag

    async def send(
        self,
        chat_type: str,
        chat_id: str,
        req: ApprovalRequest,
        msg_id: Optional[str] = None,
    ) -> bool:
        """Send an approval message to *chat_id*.

        :param chat_type: ``'c2c'`` or ``'group'``.
        :param chat_id: User openid or group openid.
        :param req: :class:`ApprovalRequest`.
        :param msg_id: Reply-to message id (required for passive messages).
        :returns: ``True`` on success, ``False`` on failure.
        """
        text = build_approval_text(req)
        keyboard = build_approval_keyboard(req.session_key)

        logger.info(
            "[%s] Sending approval request to %s:%s (session=%.20s…)",
            self._log_tag, chat_type, chat_id, req.session_key,
        )

        try:
            if chat_type == "c2c":
                await self._post_c2c(chat_id, text, msg_id, keyboard)
            elif chat_type == "group":
                await self._post_group(chat_id, text, msg_id, keyboard)
            else:
                logger.warning(
                    "[%s] Approval: unsupported chat_type %r",
                    self._log_tag, chat_type,
                )
                return False
            logger.info(
                "[%s] Approval message sent to %s:%s",
                self._log_tag, chat_type, chat_id,
            )
            return True
        except Exception as exc:
            logger.error(
                "[%s] Failed to send approval message to %s:%s: %s",
                self._log_tag, chat_type, chat_id, exc,
            )
            return False


# ── INTERACTION_CREATE event shape ───────────────────────────────────

@dataclass
class InteractionEvent:
    """Parsed ``INTERACTION_CREATE`` event payload.

    See https://bot.q.qq.com/wiki/develop/api-v2/dev-prepare/interface-framework/event-emit.html
    """
    id: str = ""
    """Interaction event id — required for the ``PUT /interactions/{id}`` ACK."""

    type: int = 0
    """Event type code (``11`` = message button)."""

    chat_type: int = 0
    """``0`` = guild, ``1`` = group, ``2`` = c2c."""

    scene: str = ""
    """``'guild'`` | ``'group'`` | ``'c2c'`` — human-readable scene."""

    group_openid: str = ""
    group_member_openid: str = ""
    user_openid: str = ""
    channel_id: str = ""
    guild_id: str = ""

    button_data: str = ""
    button_id: str = ""
    resolver_user_id: str = ""

    @property
    def operator_openid(self) -> str:
        """Best available operator openid (group → member; c2c → user)."""
        return (
            self.group_member_openid
            or self.user_openid
            or self.resolver_user_id
        )


def parse_interaction_event(raw: Dict[str, Any]) -> InteractionEvent:
    """Parse a raw ``INTERACTION_CREATE`` dispatch payload (``d``)."""
    data_raw = raw.get("data") or {}
    resolved = data_raw.get("resolved") or {}
    scene_code = int(raw.get("chat_type", 0) or 0)
    scene = {0: "guild", 1: "group", 2: "c2c"}.get(scene_code, "")
    return InteractionEvent(
        id=str(raw.get("id", "")),
        type=int(data_raw.get("type", 0) or 0),
        chat_type=scene_code,
        scene=scene,
        group_openid=str(raw.get("group_openid", "")),
        group_member_openid=str(raw.get("group_member_openid", "")),
        user_openid=str(raw.get("user_openid", "")),
        channel_id=str(raw.get("channel_id", "")),
        guild_id=str(raw.get("guild_id", "")),
        button_data=str(resolved.get("button_data", "")),
        button_id=str(resolved.get("button_id", "")),
        resolver_user_id=str(resolved.get("user_id", "")),
    )
