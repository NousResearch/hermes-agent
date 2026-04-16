from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)


DEFAULT_MSTEAMS_STATE_PATH = get_hermes_home() / "gateway" / "msteams_conversations.json"


def default_msteams_state_path() -> Path:
    return DEFAULT_MSTEAMS_STATE_PATH


@dataclass
class ConversationRef:
    conversation_id: str
    service_url: str
    conversation_type: str
    chat_type: str
    tenant_id: Optional[str] = None
    team_id: Optional[str] = None
    channel_id: Optional[str] = None
    activity_id: Optional[str] = None
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    chat_name: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    last_inbound_activity_id: Optional[str] = None
    last_sent_activity_id: Optional[str] = None
    reply_style: Optional[str] = None
    require_mention: Optional[bool] = None
    sent_activity_ids: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["sent_activity_ids"] = sorted(self.sent_activity_ids)
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConversationRef":
        data = dict(payload or {})
        sent_activity_ids = data.get("sent_activity_ids") or []
        if not isinstance(sent_activity_ids, (list, tuple, set)):
            sent_activity_ids = []
        data["sent_activity_ids"] = {str(item) for item in sent_activity_ids if str(item).strip()}
        return cls(**data)


class ConversationRegistry:
    """Registry for Teams conversation references and reply state."""

    def __init__(self) -> None:
        self._by_conversation: Dict[str, ConversationRef] = {}

    def remember(self, ref: ConversationRef) -> ConversationRef:
        existing = self._by_conversation.get(ref.conversation_id)
        if existing is None:
            self._by_conversation[ref.conversation_id] = ref
            return ref

        existing.service_url = ref.service_url or existing.service_url
        existing.conversation_type = ref.conversation_type or existing.conversation_type
        existing.chat_type = ref.chat_type or existing.chat_type
        existing.tenant_id = ref.tenant_id or existing.tenant_id
        existing.team_id = ref.team_id or existing.team_id
        existing.channel_id = ref.channel_id or existing.channel_id
        existing.user_id = ref.user_id or existing.user_id
        existing.user_name = ref.user_name or existing.user_name
        existing.chat_name = ref.chat_name or existing.chat_name
        existing.raw = ref.raw or existing.raw
        if ref.activity_id and (ref.activity_id not in existing.sent_activity_ids or not existing.activity_id):
            existing.activity_id = ref.activity_id
        if ref.last_inbound_activity_id:
            existing.last_inbound_activity_id = ref.last_inbound_activity_id
        if ref.last_sent_activity_id:
            existing.last_sent_activity_id = ref.last_sent_activity_id
        if ref.reply_style:
            existing.reply_style = ref.reply_style
        if ref.require_mention is not None:
            existing.require_mention = ref.require_mention
        if ref.sent_activity_ids:
            existing.sent_activity_ids.update(ref.sent_activity_ids)
        return existing

    def get(self, conversation_id: str) -> Optional[ConversationRef]:
        return self._by_conversation.get(conversation_id)

    def update_activity(self, conversation_id: str, activity_id: Optional[str]) -> None:
        if not activity_id:
            return
        ref = self._by_conversation.get(conversation_id)
        if ref:
            ref.activity_id = activity_id
            ref.last_inbound_activity_id = activity_id

    def register_sent_message(self, conversation_id: str, activity_id: Optional[str]) -> None:
        if not activity_id:
            return
        ref = self._by_conversation.get(conversation_id)
        if ref is None:
            return
        ref.last_sent_activity_id = activity_id
        ref.sent_activity_ids.add(activity_id)

    def has_sent_activity(self, conversation_id: str, activity_id: Optional[str]) -> bool:
        if not activity_id:
            return False
        ref = self._by_conversation.get(conversation_id)
        if ref is None:
            return False
        return activity_id in ref.sent_activity_ids

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {conversation_id: ref.to_dict() for conversation_id, ref in self._by_conversation.items()}

    def save_to_path(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else default_msteams_state_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.flush()
        tmp_path.replace(target)
        return target

    @classmethod
    def load_from_path(cls, path: str | Path | None = None) -> "ConversationRegistry":
        target = Path(path) if path else default_msteams_state_path()
        registry = cls()
        if not target.exists():
            return registry
        try:
            with target.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            logger.warning("[MSTeams] Failed to load conversation state from %s: %s", target, exc)
            return registry
        if not isinstance(payload, dict):
            logger.warning("[MSTeams] Ignoring invalid conversation state payload at %s", target)
            return registry
        for conversation_id, ref_payload in payload.items():
            if not isinstance(ref_payload, dict):
                continue
            try:
                ref = ConversationRef.from_dict(ref_payload)
            except Exception as exc:
                logger.warning("[MSTeams] Skipping invalid conversation state for %s: %s", conversation_id, exc)
                continue
            registry._by_conversation[str(conversation_id)] = ref
        return registry
