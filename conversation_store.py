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


@dataclass(frozen=True)
class ConversationMutationResult:
    """Successful canonical mutation and the revision it produced.

    For ``append_messages``, ``message_ids`` contains one stable Hermes integer
    id per submitted message (including repaired/adopted rows),
    ``canonical_messages`` may return the provider-authoritative content for
    those inputs, and the two count fields update Hermes' local operational
    shadow only after the canonical commit succeeds.
    """

    revision: ConversationRevision
    affected_count: int = 0
    message_ids: tuple[int, ...] = ()
    canonical_messages: tuple[dict[str, Any], ...] = ()
    tool_call_count_delta: int = 0
    details: Any = None


@dataclass(frozen=True)
class ConversationSnapshot:
    """Provider-atomic state used to compute a fenced mutation."""

    revision: ConversationRevision
    conversation: dict[str, Any]
    messages: tuple[dict[str, Any], ...] = ()


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

    def active_message_ids(self, conversation_id: str) -> list[int]:
        """Stable ids of the current live transcript in canonical order."""
        raise NotImplementedError

    # Phase 3: optimistic revision/CAS mutation contract.
    def get_revision(self, conversation_id: str) -> ConversationRevision:
        raise NotImplementedError

    def snapshot(self, conversation_id: str, *, include_messages: bool = False) -> ConversationSnapshot:
        raise NotImplementedError

    def ensure_conversation(self, conversation) -> ConversationRevision:
        """Create if absent, otherwise return the existing canonical revision unchanged."""
        raise NotImplementedError

    def append_messages(
        self, conversation_id: str, messages, *, expected_revision: ConversationRevision,
        idempotency_key: str | None = None,
    ):
        """Append atomically and return canonical row identities/content for the input batch."""
        raise NotImplementedError

    def update_conversation(
        self, conversation_id: str, changes, *, expected_revision: ConversationRevision,
        include_lineage: bool = False,
    ):
        raise NotImplementedError

    def set_conversation_title(
        self, conversation_id: str, title, *, source: str,
        expected_revision: ConversationRevision,
    ):
        """Apply Hermes title sanitization/precedence/uniqueness semantics atomically."""
        raise NotImplementedError

    def set_conversation_title_source(
        self, conversation_id: str, source: str, *, expected_revision: ConversationRevision,
    ):
        raise NotImplementedError

    def set_latest_matching_message_display(
        self, conversation_id: str, *, role: str, content: str, display_kind: str,
        display_metadata, expected_revision: ConversationRevision,
    ):
        raise NotImplementedError

    def set_message_reaction(
        self, conversation_id: str, message_id: int, emoji, *, author: str,
        expected_revision: ConversationRevision,
    ):
        raise NotImplementedError

    def get_message_reactions(self, conversation_id: str, message_id: int):
        raise NotImplementedError

    def replace_messages(
        self, conversation_id: str, messages, *, expected_revision: ConversationRevision,
        expected_active_ids, active_only: bool = False, archive_dropped: bool = False,
    ):
        """Replace only if both the revision and observed live message ids still match."""
        raise NotImplementedError

    def rewind_to_message(
        self, conversation_id: str, message_id: int, *, expected_revision: ConversationRevision,
        expected_active_ids, expected_target_content=None,
        preserve_compaction_handoff: bool = False,
    ):
        """Rewind only the transcript snapshot represented by ``expected_active_ids``."""
        raise NotImplementedError

    def publish_compaction(
        self, conversation_id: str, messages, *, expected_revision: ConversationRevision,
        expected_active_ids=None, model_config_patch=None, tail_count: int = 0,
        mode: str = "in_place", child_conversation=None, pending_parent_messages=(),
    ):
        """Publish a fenced compaction generation atomically.

        ``expected_revision`` and ``expected_active_ids`` are the provider-atomic
        compaction-start fence.  A provider must reject any canonical change after
        that fence; it must not merge a later append into this stale summary.
        For ``mode='rotated'`` the pending parent rows are part of this same
        operation: they are appended to parent evidence, the child is created and
        populated, and the parent is closed atomically.  Implementations return
        canonical ids/content for both message groups in ``details``.
        """
        raise NotImplementedError
