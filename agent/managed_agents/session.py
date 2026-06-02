"""Session model and store.

A session groups related agent runs (discord channel, feishu thread, web route, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .workspace import DEFAULT_WORKSPACE_ID, Entrypoint


DEFAULT_SESSION_ID = "hermes-legacy"
DEFAULT_SESSION_NAME = "Default Session"


@dataclass(frozen=True, slots=True)
class Session:
    session_id: str
    workspace_id: str = DEFAULT_WORKSPACE_ID
    name: str = DEFAULT_SESSION_NAME
    entrypoint: Entrypoint = "cli"
    external_channel_id: str | None = None
    external_thread_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "name": self.name,
            "entrypoint": self.entrypoint,
            "external_channel_id": self.external_channel_id,
            "external_thread_id": self.external_thread_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Session:
        return Session(
            session_id=str(data.get("session_id") or DEFAULT_SESSION_ID),
            workspace_id=str(data.get("workspace_id") or DEFAULT_WORKSPACE_ID),
            name=str(data.get("name") or DEFAULT_SESSION_NAME),
            entrypoint=str(data.get("entrypoint") or "cli"),
            external_channel_id=data.get("external_channel_id") or None,
            external_thread_id=data.get("external_thread_id") or None,
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )

    @staticmethod
    def make_default() -> Session:
        """Return the singleton default session for legacy backward compatibility."""
        return Session(
            session_id=DEFAULT_SESSION_ID,
            workspace_id=DEFAULT_WORKSPACE_ID,
            name=DEFAULT_SESSION_NAME,
            entrypoint="cli",
        )


_session_store: dict[str, Session] = {}
_session_lock = threading.Lock()


def _session_store_path() -> Path:
    from hermes_cli.config import get_hermes_home
    return get_hermes_home() / "data" / "sessions.json"


def load_sessions() -> dict[str, Session]:
    path = _session_store_path()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Session.from_dict(v) for k, v in raw.items()}


def save_sessions(sessions: dict[str, Session]) -> None:
    path = _session_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: v.to_dict() for k, v in sessions.items()}, f, ensure_ascii=False, indent=2)


def get_session(session_id: str) -> Session | None:
    with _session_lock:
        if not _session_store:
            _session_store.update(load_sessions())
        return _session_store.get(session_id)


def put_session(session: Session) -> None:
    with _session_lock:
        _session_store[session.session_id] = session
        save_sessions(_session_store)


def resolve_session(
    session_id: str | None,
    fallback_workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> Session:
    """Resolve a session by id, falling back to default."""
    if session_id:
        existing = get_session(session_id)
        if existing:
            return existing
    return Session.make_default()
