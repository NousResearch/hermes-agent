"""Session binding store — maps external source IDs to Hermes sessions.

Used by entry adapters (Feishu, Discord, Web, CLI, Mac App) to resolve
inbound events to the correct workspace/session.  Supports both legacy
2-tuple format and the v2.10 SessionBindingValue format with source tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import threading
from pathlib import Path
from typing import Any, Literal

from .workspace import DEFAULT_WORKSPACE_ID
from .session import DEFAULT_SESSION_ID

# v2.10 BindingSource — tracks WHY a binding exists.
# "card" = user chose via interactive card
# "thread" = derived from Feishu thread_id
# "alias" = command-based explicit selection
# "default" = automatic fallback
BindingSource = Literal["card", "thread", "alias", "default"]


@dataclass(frozen=True, slots=True)
class SessionBindingValue:
    """Tracks not just where a message goes, but WHY it goes there."""

    workspace_id: str
    session_id: str
    source: BindingSource = "default"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "source": self.source,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any] | list) -> SessionBindingValue:
        # Backward compat: old format is a 2-list [workspace_id, session_id]
        if isinstance(data, list):
            return SessionBindingValue(
                workspace_id=str(data[0] or DEFAULT_WORKSPACE_ID),
                session_id=str(data[1] or DEFAULT_SESSION_ID),
                source="default",
                created_at="",
            )
        return SessionBindingValue(
            workspace_id=str(data.get("workspace_id") or DEFAULT_WORKSPACE_ID),
            session_id=str(data.get("session_id") or DEFAULT_SESSION_ID),
            source=str(data.get("source") or "default"),
            created_at=str(data.get("created_at") or ""),
        )


# Key: entrypoint + external_channel_id + external_thread_id
# Value: (workspace_id, session_id) — legacy format kept for backward compat
_bindings: dict[str, tuple[str, str]] = {}
# v2.10 format with source tracking
_binding_values: dict[str, SessionBindingValue] = {}
_bindings_lock = threading.Lock()


def _binding_key(
    entrypoint: str,
    external_channel_id: str | None,
    external_thread_id: str | None,
) -> str:
    return f"{entrypoint}:{external_channel_id or ''}:{external_thread_id or ''}"


def _bindings_store_path() -> Path:
    from hermes_cli.config import get_hermes_home
    return get_hermes_home() / "data" / "session_bindings.json"


def load_bindings() -> dict[str, tuple[str, str]]:
    path = _bindings_store_path()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result: dict[str, tuple[str, str]] = {}
    for k, v in raw.items():
        if isinstance(v, list) and len(v) >= 2:
            result[k] = (str(v[0]), str(v[1]))
        elif isinstance(v, dict):
            result[k] = (str(v["workspace_id"]), str(v["session_id"]))
    return result


def load_binding_values() -> dict[str, SessionBindingValue]:
    """Load bindings in the v2.10 SessionBindingValue format."""
    path = _bindings_store_path()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: SessionBindingValue.from_dict(v) for k, v in raw.items()}


def _serialize_bindings(values: dict[str, SessionBindingValue]) -> dict[str, Any]:
    return {k: v.to_dict() for k, v in values.items()}


def save_bindings(
    bindings: dict[str, tuple[str, str]] | None = None,
    *,
    values: dict[str, SessionBindingValue] | None = None,
) -> None:
    path = _bindings_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    if values:
        out.update(_serialize_bindings(values))
    if bindings:
        for k, v in bindings.items():
            if k not in out:
                out[k] = list(v)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def get_binding(
    entrypoint: str,
    external_channel_id: str | None,
    external_thread_id: str | None,
) -> tuple[str, str] | None:
    key = _binding_key(entrypoint, external_channel_id, external_thread_id)
    with _bindings_lock:
        if not _bindings:
            _bindings.update(load_bindings())
        return _bindings.get(key)


def get_binding_value(
    entrypoint: str,
    external_channel_id: str | None,
    external_thread_id: str | None,
) -> SessionBindingValue | None:
    """Look up a binding in the v2.10 SessionBindingValue format."""
    key = _binding_key(entrypoint, external_channel_id, external_thread_id)
    with _bindings_lock:
        if not _binding_values:
            _binding_values.update(load_binding_values())
        return _binding_values.get(key)


def put_binding(
    entrypoint: str,
    external_channel_id: str | None,
    external_thread_id: str | None,
    workspace_id: str,
    session_id: str,
    source: BindingSource = "default",
) -> None:
    key = _binding_key(entrypoint, external_channel_id, external_thread_id)
    with _bindings_lock:
        now = datetime.now(timezone.utc).isoformat()
        val = SessionBindingValue(
            workspace_id=workspace_id,
            session_id=session_id,
            source=source,
            created_at=now,
        )
        _bindings[key] = (workspace_id, session_id)
        _binding_values[key] = val
        save_bindings(values=_binding_values)


def resolve_binding(
    entrypoint: str,
    external_channel_id: str | None,
    external_thread_id: str | None,
) -> tuple[str, str]:
    """Resolve session binding; fall back to defaults if unmapped.

    Resolution order:
      1. Explicit card binding (source="card")
      2. Thread-derived binding (source="thread")
      3. Alias binding (source="alias")
      4. Default-derived binding (source="default")
      5. Fall back to (DEFAULT_WORKSPACE_ID, DEFAULT_SESSION_ID)
    """
    val = get_binding_value(entrypoint, external_channel_id, external_thread_id)
    if val:
        return (val.workspace_id, val.session_id)
    # Legacy fallback
    bound = get_binding(entrypoint, external_channel_id, external_thread_id)
    if bound:
        return bound
    return (DEFAULT_WORKSPACE_ID, DEFAULT_SESSION_ID)


def resolve_binding_with_source(
    entrypoint: str,
    external_channel_id: str | None,
    external_thread_id: str | None,
) -> SessionBindingValue | None:
    """Resolve binding with source metadata, or None if no binding exists."""
    return get_binding_value(entrypoint, external_channel_id, external_thread_id)
