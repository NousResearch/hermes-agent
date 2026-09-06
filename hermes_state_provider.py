"""Abstract base class for pluggable session-storage providers (RFC #23717, Phase 1).

Mirrors the MemoryProvider pattern (``agent/memory_provider.py``): ONE provider active at a
time, selected via the ``sessiondb.provider`` config key (default ``sqlite``; future
backends arrive default-off behind config.yaml, never a new env var). The bundled SQLite
engine (``SessionDB``) is the reference implementation; Phase 1 ships no other backend.
Kanban and the gateway's other SQLite stores are explicitly out of scope (RFC Decision 5).

The contract is synchronous by design (RFC Decision 3): every call site today is sync, and
the ABC carries the minimum lifecycle/CRUD/messages/search contract — deliberately NOT an
exhaustive mirror of SessionDB's full surface (the failure mode that sank PR #71945).
Abstract signatures copy SessionDB's exactly (LSP: an implementation must accept everything
the contract declares), including the house-style ``str = None`` annotations.

``SQLiteSessionProvider`` is a lazy alias for ``SessionDB`` (PEP 562 ``__getattr__``):
this module cannot import the facade at module level because the facade imports this ABC.
"""

from __future__ import annotations

import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Collection, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionDBProvider(ABC):
    """Abstract base for pluggable session storage backends (RFC #23717)."""

    # Set by hermes_state_registry.acquire() on shared instances: close() becomes a refcount
    # release and the registry owns the connection lifecycle.
    _shared_registry_owned = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider (e.g. 'sqlite')."""

    @abstractmethod
    def is_available(self) -> bool:
        """Configured and ready to serve? False once a handle is quarantined."""

    # ── Lifecycle ──

    @abstractmethod
    def initialize(self, **kwargs: Any) -> None:
        """One-time startup (pools, threads). SQLite connects eagerly in __init__."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release connections/resources; alias of close() for lifecycle drivers."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying connection(s)."""

    # ── Session CRUD ──

    @abstractmethod
    def create_session(self, session_id: str, source: str, **kwargs) -> str:
        """Create (upsert) a session record; returns the session_id."""

    @abstractmethod
    def ensure_session(self, session_id: str, source: str = "unknown", model: str = None,
                       **kwargs) -> str:
        """Ensure a session row exists (upsert)."""

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session row by ID, or None."""

    @abstractmethod
    def end_session(self, session_id: str, end_reason: str) -> None:
        """Mark a session ended; the first end_reason wins."""

    @abstractmethod
    def reopen_session(self, session_id: str) -> None:
        """Clear the ended marker so a session can be resumed."""

    @abstractmethod
    def delete_session(
        self, session_id: str, sessions_dir: Optional[Path] = None,
        expected_delete_ids: Optional[List[str]] = None,
    ) -> bool:
        """Delete a session and its messages."""

    @abstractmethod
    def update_session_meta(self, session_id: str, model_config_json: str,
                            model: Optional[str] = None) -> None:
        """Persist model/config metadata for a session."""

    # ── Titles ──

    @abstractmethod
    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set a session title on the user's behalf."""

    @abstractmethod
    def get_session_title(self, session_id: str) -> Optional[str]:
        """Get the title for a session, or None."""

    # ── Messages ──

    @abstractmethod
    def append_message(
        self, session_id: str, role: str, content: str = None, tool_name: str = None,
        tool_calls: Any = None, tool_call_id: str = None, token_count: int = None,
        finish_reason: str = None, reasoning: str = None, reasoning_content: str = None,
        reasoning_details: Any = None, codex_reasoning_items: Any = None,
        codex_message_items: Any = None, platform_message_id: str = None, observed: bool = False,
        effect_disposition: Optional[str] = None, _compressed_summary: bool = False,
        timestamp: Any = None, api_content: Optional[str] = None,
        display_kind: Optional[str] = None, display_metadata: Optional[Dict[str, Any]] = None,
        compression_lock_holder: Optional[str] = None, turn_lease_holder: Optional[str] = None,
        turn_lease_ttl_seconds: float = 300.0,
    ) -> int:
        """Append one message; returns the row id."""

    @abstractmethod
    def append_messages_batch(
        self, session_id: str, messages: List[Dict[str, Any]],
        compression_lock_holder: Optional[str] = None, turn_lease_holder: Optional[str] = None,
        chunk_rows: Optional[int] = None, turn_lease_ttl_seconds: float = 300.0,
    ) -> int:
        """Append messages in one write transaction; returns the inserted count."""

    @abstractmethod
    def replace_messages(self, session_id: str, messages: List[Dict[str, Any]],
                         active_only: bool = False, archive_dropped: bool = False,
                         reject_active_turn_lease: bool = False) -> None:
        """Atomically replace a session's messages (/retry, /undo, /compress)."""

    @abstractmethod
    def get_messages(self, session_id: str, include_inactive: bool = False,
                     include_compacted: bool = False, limit: Optional[int] = None,
                     offset: int = 0, latest: bool = False,
                     after_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load messages in insertion order (id, never timestamp)."""

    @abstractmethod
    def get_messages_as_conversation(self, session_id: str, include_ancestors: bool = False,
                                     include_inactive: bool = False,
                                     repair_alternation: bool = False,
                                     include_row_ids: bool = False,
                                     include_compacted: bool = False) -> List[Dict[str, Any]]:
        """Load messages in OpenAI conversation format."""

    # ── Search / counts ──

    @abstractmethod
    def search_messages(
        self, query: str, source_filter: List[str] = None, exclude_sources: List[str] = None,
        role_filter: List[str] = None, limit: int = 20, offset: int = 0, sort: str = None,
        include_inactive: bool = False, fields: Optional[Collection[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text (or degraded LIKE) search over message content."""

    @abstractmethod
    def search_sessions(
        self, source: str = None, limit: int = 20, offset: int = 0, workspace_key: str = None,
    ) -> List[Dict[str, Any]]:
        """Sessions MRU-first with a computed ``last_active``."""

    @abstractmethod
    def list_sessions_rich(
        self, source: str = None, sources: List[str] = None, exclude_sources: List[str] = None,
        cwd_prefix: str = None, limit: int = 20, offset: int = 0, include_children: bool = False,
        min_message_count: int = 0, project_compression_tips: bool = True,
        order_by_last_active: bool = False, include_archived: bool = False,
        archived_only: bool = False, id_query: str = None, search_query: str = None,
        compact_rows: bool = False, include_pinned: bool = False, session_key: str = None,
        include_hidden: bool = False,
    ) -> List[Dict[str, Any]]:
        """Rich session listing for the dashboard/TUI."""

    @abstractmethod
    def session_count(
        self, source: str = None, sources: List[str] = None, cwd_prefix: str = None,
        min_message_count: int = 0, include_archived: bool = False, archived_only: bool = False,
        exclude_children: bool = False, exclude_sources: List[str] = None,
    ) -> int:
        """Count sessions matching the listing filters."""

    @abstractmethod
    def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for one session."""

    # ── Meta key/value ──

    @abstractmethod
    def get_meta(self, key: str) -> Optional[str]:
        """Read a state_meta value."""

    @abstractmethod
    def set_meta(self, key: str, value: str, *, cursor: Optional[sqlite3.Cursor] = None) -> None:
        """Upsert a state_meta value; ``cursor`` writes inline in the caller's transaction."""

    # ── Write-transaction hooks (testing seams; see tests/session/) ──
    # Concrete (MemoryProvider-style defaults): the SQLite engine consults these lists in
    # its write funnel; other providers honor the same contract around their own commits.
    # The pre_commit argument is the backend-native transaction handle (a sqlite3.Connection
    # for the SQLite engine), hence Any.

    def add_pre_commit_hook(self, hook: Callable[[Any], None]) -> None:
        """Register ``hook(txn)`` to run INSIDE the write transaction after the mutation
        and before commit; raising vetoes the write (it is rolled back)."""
        self.__dict__.setdefault("_pre_commit_hooks", []).append(hook)

    def add_post_write_hook(self, hook: Callable[[], None]) -> None:
        """Register ``hook()`` to run after a successful commit; never on rollback."""
        self.__dict__.setdefault("_post_write_hooks", []).append(hook)

    def clear_write_hooks(self) -> None:
        """Drop all registered pre_commit/post_write hooks."""
        self.__dict__["_pre_commit_hooks"] = []
        self.__dict__["_post_write_hooks"] = []


class SQLiteProviderMixin:
    """SessionDBProvider conformance for the bundled SQLite engine.

    Kept in this sibling (not the hermes_state facade); must precede SessionDBProvider in
    SessionDB's bases so these concrete members win MRO over the abstract declarations.
    """

    _db_corrupt: bool  # declared by SessionDB.__init__

    if TYPE_CHECKING:  # provided by SessionDB; declaration keeps the mixin self-describing
        def close(self) -> None: ...

    @property
    def name(self) -> str:
        return "sqlite"

    def is_available(self) -> bool:
        """SQLite is always present; False once this handle is corruption-quarantined."""
        return not self._db_corrupt

    def initialize(self, **kwargs: Any) -> None:
        """SessionDB connects eagerly in __init__; the lifecycle contract is a no-op."""

    def shutdown(self) -> None:
        """Lifecycle alias for close()."""
        self.close()


def get_session_db_provider(db_path: Optional[Path] = None, read_only: bool = False,
                            config: Optional[Dict[str, Any]] = None) -> SessionDBProvider:
    """Construct the configured session-storage provider (``sessiondb.provider`` in
    config.yaml; default ``sqlite``). Unknown providers fail loudly — never silently
    fall back and split a user's history across two stores."""
    if config is None:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    provider_name = (config.get("sessiondb") or {}).get("provider") or "sqlite"
    if provider_name != "sqlite":
        raise ValueError(
            f"sessiondb.provider {provider_name!r} is not available in this build; "
            "Phase 1 ships only 'sqlite' (RFC #23717)."
        )
    from hermes_state import SessionDB  # late import: the facade imports this module
    return SessionDB(db_path=db_path, read_only=read_only)


def __getattr__(name: str) -> Any:
    if name == "SQLiteSessionProvider":
        from hermes_state import SessionDB  # late import: the facade imports this module
        return SessionDB
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
