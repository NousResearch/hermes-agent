"""Provider-neutral durable conversation-store contract.

Hermes keeps hermes_state.SessionDB as its public state facade. This contract
is only for canonical, user-visible conversation/history authority; SQLite
remains responsible for Hermes operational state unless explicitly delegated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConversationStoreError(RuntimeError):
    """Base error for conversation-store failures."""


class ConversationStoreUnavailableError(ConversationStoreError):
    """The configured store cannot be loaded or is not ready."""


class ConversationConflictError(ConversationStoreError):
    """A mutation was computed from a stale canonical conversation revision."""


@dataclass(frozen=True)
class ConversationRevision:
    """Opaque provider revision used for compare-and-swap mutations."""

    value: Any


class ConversationStore(ABC):
    """Exclusive canonical conversation-history provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable config/entry-point name for this store."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether required configuration/dependencies are ready."""

    def unavailable_reason(self) -> str:
        return ""

    def initialize(self, *, hermes_home: Path) -> None:
        """Bind resources for one profile-scoped SessionDB generation."""

    def close(self) -> None:
        """Release provider resources. Must be idempotent."""

    # Phase 2: canonical read surface.
    def get_conversation(self, conversation_id: str):
        raise NotImplementedError

    def resolve_conversation_id(self, conversation_id_or_prefix: str):
        raise NotImplementedError

    def list_conversations(self, **filters):
        raise NotImplementedError

    def search_conversations(self, **filters):
        raise NotImplementedError

    def count_conversations(self, **filters) -> int:
        raise NotImplementedError

    def find_conversation_by_title(self, title: str):
        raise NotImplementedError

    def resolve_conversation_by_title(self, title: str):
        raise NotImplementedError

    def list_messages(
        self, conversation_id: str, *, include_inactive: bool = False,
        include_compacted: bool = False, limit=None, offset: int = 0,
        latest: bool = False, after_id=None,
    ):
        raise NotImplementedError

    def messages_around(self, conversation_id: str, message_id: int, *, window: int = 5):
        raise NotImplementedError

    def conversation_history(
        self, conversation_id: str, *, include_ancestors: bool = False,
        include_inactive: bool = False, include_row_ids: bool = False,
        include_compacted: bool = False,
    ):
        raise NotImplementedError

    def resume_histories(self, conversation_id: str):
        raise NotImplementedError

    def resolve_resume_conversation_id(self, conversation_id: str) -> str:
        raise NotImplementedError

    def resume_message_count(self, conversation_id: str, *, tip_only: bool = False) -> int:
        raise NotImplementedError

    def count_messages(self, conversation_id=None) -> int:
        raise NotImplementedError

    def search_messages(self, query: str, **filters):
        raise NotImplementedError

    def anchored_view(
        self, conversation_id: str, message_id: int, *, window: int = 5,
        bookend: int = 3, keep_roles=("user", "assistant"),
    ):
        raise NotImplementedError

    def recent_user_messages(
        self, conversation_id: str, *, limit: int = 20, include_inactive: bool = False,
    ):
        raise NotImplementedError

    def message_storage_state(self, message_id: int):
        raise NotImplementedError

    def latest_message_preview(self, conversation_id: str) -> str:
        raise NotImplementedError
