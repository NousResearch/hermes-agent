"""Exclusive conversation-store provider discovery.

SQLite is built into SessionDB and represented by None here. Third-party
stores are selected by sessions.store and discovered through the
hermes_agent.conversation_stores pip entry-point group. A named external
store fails closed when it cannot be loaded or is unavailable.
"""

from __future__ import annotations

import importlib.metadata
import logging
from pathlib import Path
from typing import Optional

from conversation_store import ConversationStore, ConversationStoreUnavailableError
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

BUILTIN_STORE = "sqlite"
ENTRY_POINTS_GROUP = "hermes_agent.conversation_stores"


def _iter_entry_points():
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            return list(eps.select(group=ENTRY_POINTS_GROUP))
        if isinstance(eps, dict):
            return list(eps.get(ENTRY_POINTS_GROUP, []))
        return [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]
    except Exception as exc:
        logger.debug("Conversation-store entry-point scan failed: %s", exc)
        return []


def list_conversation_store_names() -> list[str]:
    """Installed store names without importing third-party code."""
    return sorted({BUILTIN_STORE, *(ep.name for ep in _iter_entry_points())})


def find_conversation_store_entry_point(name: str):
    return next((ep for ep in _iter_entry_points() if ep.name == name), None)


class _StoreCollector:
    """Minimal plugin context for an exclusive conversation-store provider."""

    def __init__(self) -> None:
        self.store: Optional[ConversationStore] = None

    def register_conversation_store(self, store: ConversationStore) -> None:
        if not isinstance(store, ConversationStore):
            raise TypeError("register_conversation_store requires ConversationStore")
        if self.store is not None:
            raise ValueError("only one conversation store may be registered")
        self.store = store


def _from_loaded(loaded) -> Optional[ConversationStore]:
    if isinstance(loaded, ConversationStore):
        return loaded
    if isinstance(loaded, type) and issubclass(loaded, ConversationStore):
        return loaded()
    register = getattr(loaded, "register", None)
    if callable(register):
        collector = _StoreCollector()
        register(collector)
        return collector.store
    if callable(loaded):
        try:
            candidate = loaded()
        except TypeError:
            collector = _StoreCollector()
            loaded(collector)
            return collector.store
        return candidate if isinstance(candidate, ConversationStore) else None
    return None


def load_conversation_store(name: str) -> Optional[ConversationStore]:
    """Load one named external store; sqlite returns the built-in sentinel."""
    normalized = (name or BUILTIN_STORE).strip().lower()
    if normalized == BUILTIN_STORE:
        return None

    entry_point = find_conversation_store_entry_point(normalized)
    if entry_point is None:
        raise ConversationStoreUnavailableError(
            f"Configured conversation store {normalized!r} is not installed"
        )
    try:
        store = _from_loaded(entry_point.load())
    except ConversationStoreUnavailableError:
        raise
    except Exception as exc:
        raise ConversationStoreUnavailableError(
            f"Failed to load conversation store {normalized!r}: {exc}"
        ) from exc
    if store is None:
        raise ConversationStoreUnavailableError(
            f"Conversation-store entry point {normalized!r} registered no ConversationStore"
        )
    if store.name.strip().lower() != normalized:
        raise ConversationStoreUnavailableError(
            f"Conversation store {normalized!r} loaded provider named {store.name!r}"
        )
    if not store.is_available():
        reason = store.unavailable_reason().strip()
        suffix = f": {reason}" if reason else ""
        raise ConversationStoreUnavailableError(
            f"Conversation store {normalized!r} is unavailable{suffix}"
        )
    return store


def configured_conversation_store_name() -> str:
    """Read the active profile store selector from config.yaml."""
    from hermes_cli.config import cfg_get, load_config

    value = cfg_get(load_config(), "sessions", "store", default=BUILTIN_STORE)
    return str(value or BUILTIN_STORE).strip().lower()


def load_configured_conversation_store() -> Optional[ConversationStore]:
    """Load and initialize the active profile selected external store."""
    store = load_conversation_store(configured_conversation_store_name())
    if store is None:
        return None
    home = Path(get_hermes_home())
    try:
        store.initialize(hermes_home=home)
    except Exception as exc:
        try:
            store.close()
        except Exception:
            logger.debug("Conversation-store cleanup after init failure failed", exc_info=True)
        raise ConversationStoreUnavailableError(
            f"Conversation store {store.name!r} failed to initialize: {exc}"
        ) from exc
    return store
