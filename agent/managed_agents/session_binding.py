"""Session binding store — maps external source IDs to Hermes sessions.

Used by entry adapters (Feishu, Discord, Web, CLI, Mac App) to resolve
inbound events to the correct workspace/session.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .workspace import DEFAULT_WORKSPACE_ID
from .session import DEFAULT_SESSION_ID

# Key: entrypoint + external_channel_id + external_thread_id
# Value: (workspace_id, session_id)
_bindings: dict[str, tuple[str, str]] = {}
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
    return {k: tuple(v) for k, v in raw.items()}


def save_bindings(bindings: dict[str, tuple[str, str]]) -> None:
    path = _bindings_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: list(v) for k, v in bindings.items()}, f, ensure_ascii=False, indent=2)


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


def put_binding(
    entrypoint: str,
    external_channel_id: str | None,
    external_thread_id: str | None,
    workspace_id: str,
    session_id: str,
) -> None:
    key = _binding_key(entrypoint, external_channel_id, external_thread_id)
    with _bindings_lock:
        _bindings[key] = (workspace_id, session_id)
        save_bindings(_bindings)


def resolve_binding(
    entrypoint: str,
    external_channel_id: str | None,
    external_thread_id: str | None,
) -> tuple[str, str]:
    """Resolve session binding; fall back to defaults if unmapped."""
    bound = get_binding(entrypoint, external_channel_id, external_thread_id)
    if bound:
        return bound
    return (DEFAULT_WORKSPACE_ID, DEFAULT_SESSION_ID)
