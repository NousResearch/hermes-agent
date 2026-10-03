"""QQ Bot inline keyboards + approval / update-prompt helpers. A button click dispatches an
``INTERACTION_CREATE`` event carrying the button's ``data``; the bot must ACK promptly via
``PUT /interactions/{id}`` or the user sees an error indicator. ``button_data`` formats:
``approve:<session_key>:<decision>`` (allow-once|allow-always|deny) and ``update_prompt:<answer>`` (y|n).
Ported from WideLee's qqbot-agent-sdk v1.2.2 (authorship via Co-authored-by)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

APPROVAL_BUTTON_PREFIX = "approve:"
UPDATE_PROMPT_PREFIX = "update_prompt:"

# session_key may itself contain colons (agent:main:qqbot:c2c:OPENID): greedy group, decision trails.
_APPROVAL_DATA_RE = re.compile(r"^approve:(.+):(allow-once|allow-always|deny)$")
_UPDATE_PROMPT_RE = re.compile(r"^update_prompt:(y|n)$")

def _to_dict(value: Any) -> Any:
    """Serialize a dataclass tree in field-declaration order (the wire shape)."""
    if is_dataclass(value):
        return {f.name: _to_dict(getattr(value, f.name)) for f in fields(value)}
    return [_to_dict(v) for v in value] if isinstance(value, list) else value


class _Serializable:
    to_dict = _to_dict


@dataclass
class KeyboardButtonPermission(_Serializable):
    """Button permission metadata. ``type=2`` means all users can click."""
    type: int = 2


@dataclass
class KeyboardButtonAction(_Serializable):
    """Click behaviour: ``type`` 1 = Callback (INTERACTION_CREATE with ``data``), 2 = Link;
    ``click_limit=1`` = single-use."""
    type: int
    data: str
    permission: KeyboardButtonPermission = field(default_factory=KeyboardButtonPermission)
    click_limit: int = 1


@dataclass
class KeyboardButtonRenderData(_Serializable):
    """Visual rendering: pre/post-click labels; ``style`` 0 = grey, 1 = blue."""
    label: str
    visited_label: str
    style: int = 1


@dataclass
class KeyboardButton(_Serializable):
    """One button; buttons sharing a ``group_id`` are mutually exclusive."""
    id: str
    render_data: KeyboardButtonRenderData
    action: KeyboardButtonAction
    group_id: str = "default"


@dataclass
class KeyboardRow(_Serializable):
    buttons: List[KeyboardButton] = field(default_factory=list)


@dataclass
class KeyboardContent(_Serializable):
    rows: List[KeyboardRow] = field(default_factory=list)


@dataclass
class InlineKeyboard(_Serializable):
    """Top-level keyboard payload — goes into ``MessageToCreate.keyboard``."""
    content: KeyboardContent = field(default_factory=KeyboardContent)


def parse_approval_button_data(button_data: str) -> Optional[tuple[str, str]]:
    """Parse approval ``button_data`` into ``(session_key, decision)`` or ``None``."""
    return m.groups() if (m := _APPROVAL_DATA_RE.match(button_data or "")) else None


def parse_update_prompt_button_data(button_data: str) -> Optional[str]:
    """Parse update-prompt ``button_data`` into ``'y'`` / ``'n'`` or ``None``."""
    return m.group(1) if (m := _UPDATE_PROMPT_RE.match(button_data or "")) else None


def _single_row_keyboard(group_id: str, *buttons: tuple) -> InlineKeyboard:
    """One row of callback buttons from ``(id, label, visited_label, data, style)`` tuples."""
    row = KeyboardRow(buttons=[
        KeyboardButton(id=btn_id, group_id=group_id, action=KeyboardButtonAction(type=1, data=data),
                       render_data=KeyboardButtonRenderData(label=label, visited_label=visited, style=style))
        for btn_id, label, visited, data, style in buttons])
    return InlineKeyboard(content=KeyboardContent(rows=[row]))


def build_approval_keyboard(session_key: str, *, allow_permanent: bool = True) -> InlineKeyboard:
    """Build ``[✅ 允许一次] [⭐ 始终允许] [❌ 拒绝]`` (one group, so a click greys the rest). ⭐ is hidden when
    persistent scope is unavailable; *session_key* rides in ``button_data`` so the decision routes correctly."""
    prefix = f"{APPROVAL_BUTTON_PREFIX}{session_key}"
    buttons = [("allow", "✅ 允许一次", "已允许", f"{prefix}:allow-once", 1)]
    if allow_permanent:
        buttons.append(("always", "⭐ 始终允许", "已始终允许", f"{prefix}:allow-always", 1))
    buttons.append(("deny", "❌ 拒绝", "已拒绝", f"{prefix}:deny", 0))
    return _single_row_keyboard("approval", *buttons)


def build_update_prompt_keyboard() -> InlineKeyboard:
    """Build a Yes/No keyboard for update confirmation prompts."""
    return _single_row_keyboard("update_prompt", ("yes", "✓ 确认", "已确认", f"{UPDATE_PROMPT_PREFIX}y", 1),
                                ("no", "✗ 取消", "已取消", f"{UPDATE_PROMPT_PREFIX}n", 0))


@dataclass
class ApprovalRequest:
    """Approval-request display data. ``command_preview`` / ``cwd`` are set for exec approvals, ``tool_name``
    for plugin approvals; ``severity`` is ``'critical' | 'info' | ''``."""
    session_key: str
    title: str
    description: str = ""
    command_preview: str = ""
    cwd: str = ""
    tool_name: str = ""
    severity: str = ""
    timeout_sec: int = 120
    allow_permanent: bool = True


_SEVERITY_ICONS = {"critical": "🔴", "info": "🔵"}


def build_approval_text(req: ApprovalRequest) -> str:
    """Render an :class:`ApprovalRequest` into the message body (markdown)."""
    if req.command_preview or req.cwd:
        lines = ["🔐 **命令执行审批**", ""]
        if req.command_preview:
            lines.append(f"```\n{req.command_preview[:300]}\n```")
        if req.cwd:
            lines.append(f"📁 目录: {req.cwd}")
        if req.title and req.title != req.command_preview:
            lines.append(f"📋 {req.title}")
        if req.description:
            lines.append(f"📝 {req.description}")
    else:
        lines = [f"{_SEVERITY_ICONS.get(req.severity, '🟡')} **审批请求**", "", f"📋 {req.title}"]
        if req.description:
            lines.append(f"📝 {req.description}")
        if req.tool_name:
            lines.append(f"🔧 工具: {req.tool_name}")
    lines += ["", f"⏱️ 超时: {req.timeout_sec} 秒"]
    return "\n".join(lines)


@dataclass
class InteractionEvent:
    """Parsed ``INTERACTION_CREATE`` payload (api-v2 event-emit docs)."""
    id: str = ""            # required for the ``PUT /interactions/{id}`` ACK
    type: int = 0           # event type code (11 = message button)
    chat_type: int = 0      # 0 = guild, 1 = group, 2 = c2c
    scene: str = ""         # 'guild' | 'group' | 'c2c'
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
        return self.group_member_openid or self.user_openid or self.resolver_user_id


_SCENE_NAMES = {0: "guild", 1: "group", 2: "c2c"}


def parse_interaction_event(raw: Dict[str, Any]) -> InteractionEvent:
    """Parse a raw ``INTERACTION_CREATE`` dispatch payload (``d``)."""
    data_raw = raw.get("data") or {}
    resolved, scene_code = data_raw.get("resolved") or {}, int(raw.get("chat_type", 0) or 0)
    return InteractionEvent(
        id=str(raw.get("id", "")), type=int(data_raw.get("type", 0) or 0), chat_type=scene_code,
        scene=_SCENE_NAMES.get(scene_code, ""), group_openid=str(raw.get("group_openid", "")),
        group_member_openid=str(raw.get("group_member_openid", "")), user_openid=str(raw.get("user_openid", "")),
        channel_id=str(raw.get("channel_id", "")), guild_id=str(raw.get("guild_id", "")),
        button_data=str(resolved.get("button_data", "")), button_id=str(resolved.get("button_id", "")),
        resolver_user_id=str(resolved.get("user_id", "")))
