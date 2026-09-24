"""Backend-neutral durable storage for session slash-control state.

The legacy SQLite representation remains ``state_meta`` JSON.  When the selected
state store is PostgreSQL this facade opens the selected store directly; it never
constructs ``SessionDB`` as a fallback.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()
_PREFIXES = ("goal:", "heartbeat:", "loop:")


def _kind_for_key(key: str) -> tuple[str, str]:
    for prefix in _PREFIXES:
        if key.startswith(prefix) and key[len(prefix):]:
            return prefix[:-1], key[len(prefix):]
    raise ValueError(f"unsupported session control key: {key!r}")


class PostgreSQLSessionControlStore:
    """Compatibility-shaped facade over the focused PostgreSQL control table."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def get_meta(self, key: str) -> str | None:
        kind, session_id = _kind_for_key(key)
        value = self._store.get_session_control_state(session_id, kind)
        return None if value is None else json.dumps(value["payload"], ensure_ascii=False)

    def set_meta(self, key: str, value: str) -> None:
        kind, session_id = _kind_for_key(key)
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("session control payload must be an object")
        status = str(payload.get("status", "active"))
        self._store.ensure_session(session_id, source="session-control")
        self._store.put_session_control_state(session_id, kind, status, payload)

    def list_meta_prefix(self, prefix: str) -> list[tuple[str, str]]:
        kind = prefix[:-1] if prefix.endswith(":") else ""
        if kind not in {"goal", "heartbeat", "loop"}:
            return []
        return [
            (f"{kind}:{row['session_id']}", json.dumps(row["payload"], ensure_ascii=False))
            for row in self._store.list_session_control_states(kind)
        ]

    def transfer_to_session(self, parent_session_id: str, child_session_id: str) -> bool:
        return bool(self._store.transfer_session_control_states(parent_session_id, child_session_id))

    def close(self) -> None:
        self._store.close()


def _config() -> dict[str, Any]:
    from hermes_cli.config import load_config
    return load_config() or {}


def get_session_control_store() -> Any:
    """Resolve the active profile backend; selected PostgreSQL is fail-closed."""
    from hermes_constants import get_hermes_home
    from state_store import resolve_state_store_config

    home = str(get_hermes_home())
    with _LOCK:
        cached = _CACHE.get(home)
        if cached is not None:
            return cached
        config = _config()
        resolved = resolve_state_store_config(config)
        if resolved.backend == "sqlite":
            from hermes_state_registry import acquire
            store = acquire(Path(home) / "state.db")
        else:
            from state_store import open_state_store
            store = PostgreSQLSessionControlStore(open_state_store(config))
        _CACHE[home] = store
        return store


def clear_session_control_store_cache() -> None:
    """Test/process teardown helper; production caches one store per profile."""
    with _LOCK:
        stores = list(_CACHE.values())
        _CACHE.clear()
    for store in stores:
        close = getattr(store, "close", None)
        if callable(close):
            close()
