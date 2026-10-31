"""Typed error hierarchy for the enterprise control plane.

Every error here is fail-closed: callers must treat any EnterpriseError as a
denial of the requested operation. There is no "retry against another
implementation" path anywhere in the control plane.
"""

from __future__ import annotations


class EnterpriseError(Exception):
    """Base class for all control-plane errors."""


class ValidationError(EnterpriseError):
    """A resource or request failed structural validation."""


class ScopeError(EnterpriseError):
    """A reference crossed an Installation or Namespace boundary."""


class NotFoundError(EnterpriseError):
    """The exact requested resource does not exist in the requested scope."""


class ConflictError(EnterpriseError):
    """Optimistic-concurrency conflict or duplicate resource name."""


class AuthorizationError(EnterpriseError):
    """The authoritative IAMAdapter denied (or could not verify) the action.

    An unavailable adapter raises this too: unverifiable == denied.
    """


class AdmissionError(EnterpriseError):
    """Identity evidence could not be verified or admitted (OAG boundary)."""


class RestrictionError(AuthorizationError):
    """A platform Restriction narrowed an otherwise-allowed operation."""


class DriverError(EnterpriseError):
    """A selected Driver failed, is unavailable, or cannot verify enforcement."""


class DeploymentError(EnterpriseError):
    """Deployment choreography failed; the previous revision remains active."""


class RollbackError(DeploymentError):
    """Post-activation failure whose rollback could not be verified.

    The Agent must remain inactive and serve no traffic.
    """


class SecretAccessError(EnterpriseError):
    """A brokered secret operation was denied or could not be mediated."""
