"""Provider-neutral durable conversation-store contract.

Hermes keeps hermes_state.SessionDB as its public state facade. This contract
is only for the canonical, user-visible conversation/history authority;
SQLite remains responsible for Hermes operational state unless a later
contract explicitly says otherwise.
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
    """Exclusive canonical conversation-history provider.

    Phase 1 owns only selection and lifecycle. Read/write methods are added as
    Hermes delegates each behaviour in later phases; provider-specific storage
    primitives must not leak through this boundary.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable config/entry-point name for this store."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether required local configuration/dependencies are ready."""

    def unavailable_reason(self) -> str:
        """User-facing explanation when is_available() is false."""
        return ""

    def initialize(self, *, hermes_home: Path) -> None:
        """Bind resources for one profile-scoped SessionDB generation."""

    def close(self) -> None:
        """Release provider resources. Must be idempotent."""
