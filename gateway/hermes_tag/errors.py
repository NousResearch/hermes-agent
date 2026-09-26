"""Typed failures for the Hermes Tag governance kernel."""

from __future__ import annotations


class HermesTagError(RuntimeError):
    """Base class for deterministic Hermes Tag failures."""


class ConfigurationError(HermesTagError):
    """Configuration is invalid or cannot be resolved safely."""


class StorageError(HermesTagError):
    """Durable state could not be initialized or mutated safely."""


class IdentityConflict(HermesTagError):
    """An external identity is already bound to another principal."""


class UnknownIdentity(HermesTagError):
    """No principal binding exists and guest admission is unavailable."""


class IncompleteScope(HermesTagError):
    """A consequential action omitted a required scope discriminator."""


class PolicyDenied(HermesTagError):
    """Policy denied an action."""


class ApprovalRequired(PolicyDenied):
    """Policy requires an exact approval before authority may be issued."""


class StaleWriteError(HermesTagError):
    """An optimistic write attempted to overwrite a newer version."""


class BudgetExceeded(PolicyDenied):
    """An atomic budget reservation could not be satisfied."""


class LeaseError(PolicyDenied):
    """A capability lease is missing or invalid."""


class LeaseExpired(LeaseError):
    """A capability lease is no longer current."""


class LeaseTampered(LeaseError):
    """A capability lease signature or bound digest is invalid."""


class LeaseReplay(LeaseError):
    """A one-shot capability lease was reused or completed out of order."""


class ReplayDetected(HermesTagError):
    """A duplicate or cyclic continuity envelope was rejected."""


class ReceiptChainError(HermesTagError):
    """The append-only receipt chain failed verification."""


class ObligationError(PolicyDenied):
    """Required precondition or completion evidence is absent or invalid."""
