"""Feishu EntryAdapter wrapper.

Normalizes Feishu inbound payloads into Hermes EntryEvent.
Does NOT modify gateway/platforms/feishu.py transport.
Does NOT call agents, create tasks, route, or write ledger directly.

Expected raw payload shape (from Feishu message event):
    {
        "tenant_id": "tenant-123",           # optional
        "chat_id": "oc_xxx",                 # required
        "message_id": "om_xxx",              # required
        "open_id": "ou_xxx",                 # required (sender)
        "user_id": "u_xxx",                  # optional (employee id)
        "content": "hello",                  # required
        "thread_id": "om_xxx",               # optional (reply thread)
        "root_id": "om_xxx",                 # optional (topic root)
        "session_key": "feishu:...",         # optional (existing session key)
        "timestamp": 1717200000,             # optional (epoch seconds)
        "message_type": "text",              # optional
    }
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from .entry_adapter import EntryAdapter, EntryAdapterRegistry
from .entry_event import EntryEvent
from .workspace import Workspace, DEFAULT_WORKSPACE_ID
from .session import Session, DEFAULT_SESSION_ID

# Required top-level keys in a valid Feishu inbound payload.
_REQUIRED_FIELDS = frozenset({"chat_id", "message_id", "open_id", "content"})


class FeishuEntryAdapter:
    """Concrete EntryAdapter for Feishu inbound message events.

    Stateless — workspace/session resolution is pure mapping, no I/O.
    Does NOT call agents, create tasks, route, or write ledger directly.
    """

    entrypoint = "feishu"

    def __init__(self, app_id: str | None = None) -> None:
        self.app_id = app_id

    # -- EntryAdapter protocol ----------------------------------------------

    def normalize_event(self, raw: dict[str, Any]) -> EntryEvent:
        """Convert a Feishu message event payload to EntryEvent.

        Raises ValueError if required Feishu fields are missing.
        """
        _validate_feishu_payload(raw)

        tenant_id = str(raw.get("tenant_id") or "")
        chat_id = str(raw["chat_id"])
        message_id = str(raw["message_id"])
        open_id = str(raw["open_id"])
        user_id = str(raw.get("user_id") or "")
        content = str(raw["content"])
        thread_id = raw.get("thread_id") or raw.get("root_id")
        session_key = str(raw.get("session_key") or "")
        timestamp = raw.get("timestamp")

        workspace_id = _workspace_for(tenant_id, chat_id)
        session_id = _session_for(chat_id, thread_id, session_key)

        return EntryEvent(
            event_id=message_id,
            entrypoint="feishu",
            external_source_id=tenant_id or chat_id,
            external_channel_id=chat_id,
            external_thread_id=str(thread_id) if thread_id else None,
            external_user_id=user_id or open_id,
            workspace_id=workspace_id,
            session_id=session_id,
            message=content,
            intent=_detect_intent(content),
            created_at=_format_timestamp(timestamp) if timestamp else None,
            origin_entrypoint="feishu",
            dedupe_key=f"feishu:{message_id}",
        )

    def resolve_workspace(self, event: EntryEvent) -> Workspace | None:
        """Map Feishu tenant/chat to Workspace.

        Returns a Workspace with id `ws-feishu-{chat_id}` if chat_id is
        available; otherwise returns None (caller should use default).
        """
        wid = event.workspace_id
        if wid == DEFAULT_WORKSPACE_ID:
            return None
        return Workspace(
            workspace_id=wid,
            name=f"Feishu:{wid}",
            entrypoint="feishu",
            external_source_id=event.external_source_id,
        )

    def resolve_session(
        self, event: EntryEvent, workspace: Workspace
    ) -> Session | None:
        """Map Feishu chat/thread to Session.

        Returns a Session with id `ses-feishu-{chat_id}` (or thread variant).
        """
        sid = event.session_id
        if sid == DEFAULT_SESSION_ID:
            return None
        return Session(
            session_id=sid,
            workspace_id=workspace.workspace_id,
            name=f"Feishu:{sid}",
            entrypoint="feishu",
            external_channel_id=event.external_channel_id,
            external_thread_id=event.external_thread_id,
        )

    def health(self) -> dict[str, Any]:
        """Report adapter health.

        Configured = app_id is set.  Otherwise "unconfigured".
        """
        if self.app_id:
            return {"entrypoint": "feishu", "status": "configured", "app_id": self.app_id}
        return {"entrypoint": "feishu", "status": "unconfigured", "reason": "app_id not set"}


    def resolve_session_with_ambiguity(
        self,
        event: "EntryEvent",
        *,
        active_sessions: tuple[str, ...] | None = None,
    ):
        """Resolve a Feishu EntryEvent's session using the full priority chain.

        Delegates to feishu_session_resolver.resolve_feishu_session().
        Returns a ResolutionResult indicating whether ambiguity was detected
        and whether an interactive card should be sent.

        Args:
            event: The EntryEvent to resolve.
            active_sessions: Optional tuple of active session IDs for
                ambiguity detection.

        Returns:
            ResolutionResult with workspace_id, session_id, source, and
            ambiguity info.
        """
        from .feishu_session_resolver import resolve_feishu_session
        return resolve_feishu_session(event, active_sessions=active_sessions)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _validate_feishu_payload(raw: dict[str, Any]) -> None:
    missing = _REQUIRED_FIELDS - raw.keys()
    if missing:
        raise ValueError(f"Feishu payload missing required fields: {sorted(missing)}")


def _workspace_for(tenant_id: str, chat_id: str) -> str:
    """Feishu tenant + chat -> workspace; chat only -> chat workspace."""
    if tenant_id:
        return f"ws-feishu-{tenant_id}"
    return f"ws-feishu-{chat_id}"


def _session_for(chat_id: str, thread_id: str | None, session_key: str) -> str:
    """Feishu thread -> sub-session; session_key -> explicit session; chat -> session."""
    if session_key:
        return session_key
    if thread_id:
        return f"ses-feishu-thread-{thread_id}"
    return f"ses-feishu-{chat_id}"


def _detect_intent(content: str) -> str | None:
    """Detect intent from Feishu message content."""
    stripped = content.strip()
    if stripped.startswith("/"):
        return "command"
    if "@" in stripped:
        return "mention"
    return None


def _format_timestamp(ts: Any) -> str:
    """Format Feishu epoch timestamp to ISO 8601."""
    from datetime import datetime, timezone
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return str(ts)


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_feishu_entry_adapter(
    registry: EntryAdapterRegistry,
    app_id: str | None = None,
) -> FeishuEntryAdapter:
    """Create and register a FeishuEntryAdapter in the given registry."""
    adapter = FeishuEntryAdapter(app_id=app_id)
    registry.register(adapter)
    return adapter
