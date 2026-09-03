"""Durable Hermes-owned authority for sealed Phase 2 graph nodes.

Stable public facade.  The implementation lives in bounded modules:

- ``agent.phase2_sqlite``   — descriptor-pinned SQLite opening/hardening
- ``agent.phase2_errors``   — typed authority exceptions
- ``agent.phase2_envelope`` — envelope v2 validation, canonical values,
  context-scoped binding
- ``agent.phase2_budget``   — budget reservation/reconciliation mixin
- ``agent.phase2_ledger``   — append-only plan/event ledger and lease/fence
  lifecycle (concrete ``Phase2AuthorityStore``)

Import from this module; downstream code must not depend on the split.
"""

from __future__ import annotations

from agent.phase2_envelope import (
    bind_sealed_envelope,
    current_authoritative_fence,
    current_sealed_envelope,
    validate_sealed_envelope,
)
from agent.phase2_errors import (
    AuthorityError,
    AuthorityMigrationRequired,
    MalformedEnvelopeFence,
    ResultRejection,
)
from agent.phase2_ledger import Phase2AuthorityStore, default_db_path
from agent.phase2_sqlite import (  # noqa: F401  (shared-store private surface)
    _check_phase2_db_files,
    _open_phase2_sqlite,
)

__all__ = [
    "AuthorityError",
    "AuthorityMigrationRequired",
    "MalformedEnvelopeFence",
    "Phase2AuthorityStore",
    "ResultRejection",
    "bind_sealed_envelope",
    "current_authoritative_fence",
    "current_sealed_envelope",
    "default_db_path",
    "validate_sealed_envelope",
]
