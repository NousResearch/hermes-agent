"""Policy helpers for optional WhatsApp service conversations.

This module is intentionally conservative.  It models explicit approved service
chats and safe defaults without granting broad autonomy to arbitrary WhatsApp
contacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from gateway.config import GatewayConfig, Platform


_ALLOWED_MODES = {"draft_first", "approved_followup", "paused", "revoked"}


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _coerce_int(value: Any, default: int, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def normalize_whatsapp_chat_id(value: Any) -> Optional[str]:
    """Normalize phone numbers / JIDs into canonical WhatsApp chat IDs.

    - ``+34600111222`` -> ``34600111222@s.whatsapp.net``
    - ``34600111222`` -> ``34600111222@s.whatsapp.net``
    - ``34600111222@s.whatsapp.net`` -> unchanged
    - ``12345678901234@lid`` -> unchanged
    - ``12345@g.us`` -> unchanged
    - human labels (e.g. ``Movistar (dm)``) -> ``None``
    """
    raw = str(value or "").strip()
    if not raw:
        return None

    normalized = raw.replace(":", "@", 1)
    if normalized.endswith("@lid") or normalized.endswith("@s.whatsapp.net") or normalized.endswith("@g.us"):
        return normalized

    digits = "".join(ch for ch in normalized if ch.isdigit())
    if len(digits) >= 7:
        return f"{digits}@s.whatsapp.net"

    return None


@dataclass
class WhatsAppServiceConversationConfig:
    enabled: bool = False
    default_mode: str = "draft_first"
    approved_chats: list[str] = field(default_factory=list)
    allow_agent_initiated_service_chats: bool = False
    auto_promote_replied_to_chats: bool = False
    max_new_service_chats_per_day: int = 3
    require_explicit_approval_for_first_contact: bool = True
    allow_inbound_media: bool = False
    allow_outbound_media: bool = False
    pending_first_contact_ttl_minutes: int = 1440
    max_auto_sends_per_chat_per_hour: int = 3
    max_consecutive_agent_sends_without_reply: int = 2

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "WhatsAppServiceConversationConfig":
        data = data or {}
        mode = str(data.get("default_mode", "draft_first") or "draft_first").strip().lower()
        if mode not in _ALLOWED_MODES:
            mode = "draft_first"

        approved = []
        for candidate in data.get("approved_chats", []) or []:
            normalized = normalize_whatsapp_chat_id(candidate)
            if normalized and normalized not in approved:
                approved.append(normalized)

        return cls(
            enabled=_coerce_bool(data.get("enabled"), False),
            default_mode=mode,
            approved_chats=approved,
            allow_agent_initiated_service_chats=_coerce_bool(data.get("allow_agent_initiated_service_chats"), False),
            auto_promote_replied_to_chats=_coerce_bool(data.get("auto_promote_replied_to_chats"), False),
            max_new_service_chats_per_day=_coerce_int(data.get("max_new_service_chats_per_day"), 3, minimum=0),
            require_explicit_approval_for_first_contact=_coerce_bool(data.get("require_explicit_approval_for_first_contact"), True),
            allow_inbound_media=_coerce_bool(data.get("allow_inbound_media"), False),
            allow_outbound_media=_coerce_bool(data.get("allow_outbound_media"), False),
            pending_first_contact_ttl_minutes=_coerce_int(data.get("pending_first_contact_ttl_minutes"), 1440, minimum=1),
            max_auto_sends_per_chat_per_hour=_coerce_int(data.get("max_auto_sends_per_chat_per_hour"), 3, minimum=0),
            max_consecutive_agent_sends_without_reply=_coerce_int(data.get("max_consecutive_agent_sends_without_reply"), 2, minimum=0),
        )

    @property
    def approved_chat_set(self) -> set[str]:
        return set(self.approved_chats)


@dataclass
class WhatsAppServiceConversationPolicy:
    config: WhatsAppServiceConversationConfig

    @classmethod
    def disabled(cls) -> "WhatsAppServiceConversationPolicy":
        return cls(WhatsAppServiceConversationConfig())

    @classmethod
    def from_gateway_config(cls, gateway_config: Optional[GatewayConfig]) -> "WhatsAppServiceConversationPolicy":
        if not gateway_config:
            return cls.disabled()
        platform_cfg = gateway_config.platforms.get(Platform.WHATSAPP)
        extra = getattr(platform_cfg, "extra", {}) or {}
        return cls(WhatsAppServiceConversationConfig.from_dict(extra.get("service_conversations")))

    def normalize_chat_id(self, value: Any) -> Optional[str]:
        return normalize_whatsapp_chat_id(value)

    def is_enabled(self) -> bool:
        return self.config.enabled

    def is_approved_chat(self, chat_id: Any) -> bool:
        normalized = self.normalize_chat_id(chat_id)
        if not normalized:
            return False
        return normalized in self.config.approved_chat_set

    def can_accept_inbound(self, chat_id: Any) -> bool:
        return self.is_enabled() and self.is_approved_chat(chat_id)

    def requires_first_contact_approval(self) -> bool:
        return self.config.require_explicit_approval_for_first_contact

    def can_agent_initiate(self) -> bool:
        return self.is_enabled() and self.config.allow_agent_initiated_service_chats

    def allows_inbound_media(self) -> bool:
        return self.is_enabled() and self.config.allow_inbound_media

    def allows_outbound_media(self) -> bool:
        return self.is_enabled() and self.config.allow_outbound_media

    def is_operator_command_allowed(self, from_approved_service_chat: bool) -> bool:
        # Approved provider/service chats are allowed as conversational inputs only.
        # They must never inherit trusted-operator command permissions.
        return not from_approved_service_chat

    def normalize_targets(self, values: Iterable[Any]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            chat_id = self.normalize_chat_id(value)
            if chat_id and chat_id not in normalized:
                normalized.append(chat_id)
        return normalized
