"""EntryEvent model — canonical inbound event for all Hermes entrypoints.

Feishu, Discord, Web Console, CLI, and future Mac App all produce EntryEvents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datetime import datetime, timezone
from uuid import uuid4

from .workspace import DEFAULT_WORKSPACE_ID, Entrypoint
from .session import DEFAULT_SESSION_ID


@dataclass(frozen=True, slots=True)
class EntryEvent:
    event_id: str
    entrypoint: Entrypoint
    external_source_id: str | None = None
    external_channel_id: str | None = None
    external_thread_id: str | None = None
    external_user_id: str | None = None
    workspace_id: str = DEFAULT_WORKSPACE_ID
    session_id: str = DEFAULT_SESSION_ID
    message: str = ""
    intent: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    origin_entrypoint: Entrypoint | None = None
    dedupe_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "entrypoint": self.entrypoint,
            "external_source_id": self.external_source_id,
            "external_channel_id": self.external_channel_id,
            "external_thread_id": self.external_thread_id,
            "external_user_id": self.external_user_id,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "message": self.message,
            "intent": self.intent,
            "created_at": self.created_at,
            "origin_entrypoint": self.origin_entrypoint,
            "dedupe_key": self.dedupe_key,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> EntryEvent:
        return EntryEvent(
            event_id=str(data.get("event_id") or uuid4().hex),
            entrypoint=str(data.get("entrypoint") or "cli"),
            external_source_id=data.get("external_source_id") or None,
            external_channel_id=data.get("external_channel_id") or None,
            external_thread_id=data.get("external_thread_id") or None,
            external_user_id=data.get("external_user_id") or None,
            workspace_id=str(data.get("workspace_id") or DEFAULT_WORKSPACE_ID),
            session_id=str(data.get("session_id") or DEFAULT_SESSION_ID),
            message=str(data.get("message") or ""),
            intent=data.get("intent") or None,
            created_at=str(data.get("created_at") or ""),
            origin_entrypoint=data.get("origin_entrypoint") or None,
            dedupe_key=data.get("dedupe_key") or None,
        )
