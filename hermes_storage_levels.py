"""The ownership levels a tenancy-aware store can hold, and the refusal for a pin it cannot serve.

Shared by the storage ports outside state.db (memory, skills index and their peers), so none of
them imports another port for these two names. They are tenancy-ownership concepts read from the
bound ``hermes_scope.Scope``, not part of the state.db persistence interface (``persistence/``).
"""

from __future__ import annotations

# The ownership levels a store can hold (``hermes_scope.Scope`` docs): participant-on-agent, agent,
# owner-wide, tenant-wide. A backend declares which it stores and refuses writes at any other.
LEVELS = frozenset({"participant", "agent", "owner", "tenant"})


class StoragePinRefused(PermissionError):
    """The bound scope is not one this store serves: another tenant, another agent, or (for a
    backend that requires one) no scope at all. Says nothing about whether the target exists."""
