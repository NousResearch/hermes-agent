"""Durable Hermes-owned authority for sealed Phase 2 graph nodes.

Stable public facade.  The implementation lives in bounded modules:

- ``agent.phase2_sqlite``   — descriptor-pinned SQLite opening/hardening
- ``agent.phase2_errors``   — typed authority exceptions
- ``agent.phase2_envelope`` — envelope v2 validation, canonical values,
  context-scoped binding
- ``agent.phase2_budget``   — budget reservation/reconciliation mixin
- ``agent.phase2_ledger``   — append-only plan/event ledger and lease/fence
  lifecycle (concrete ``Phase2AuthorityStore``)

Import authority ledger and envelope APIs from this module. Mutation claims
have a separate stable API in ``agent.phase2_idempotency``. Neither raw context
binding nor an idempotency claim authorizes an effect. These are primitives;
runtime wiring requires a composed effect-time live-authority and claim gate,
including revocation-between-bind-and-effect coverage.

Design provenance
-----------------
This storage-only subset follows Axl Ibiza (andrexibiza)'s Hermes Authority
Execution Layer architecture [1] and S2 implementation plan [2], published
August 25, 2026. The design correspondence is:

- Append-only history, immutable terminal acceptance and qualified operation
  identity follow [1]; hash-chained ledger positions are explicit in [2].
- Monotonic fencing and non-reused attempts follow [1]'s generation/lifecycle
  contract; this local fence is not its complete generation vector.
- Same-intent replay versus different-intent refusal follows [1]. The concrete
  qualified-collision correction and hostile regressions were requested in
  Axl's September 3 review [3], which also drove exact integer token ceilings
  and the bounded-module split.
- Current authority must be checked at the effect boundary [1]. Context
  binding and this separate claim store do not implement that runtime gate
  or the complete carrier/settlement system.

Axl's Authority Policy ABI/compiler work is published in #95101 [4]. This
slice does not yet consume the complete canonical policy/runtime contract;
policy-identity reconciliation remains required before runtime adoption.
Jack Field (jdot-dev) authored the implementation and tests in this PR;
the design and review attribution above is separate from Git authorship.

[1] https://github.com/NousResearch/hermes-agent/issues/95028#issuecomment-5416260554
[2] https://github.com/NousResearch/hermes-agent/issues/95028#issuecomment-5416261977
[3] https://github.com/NousResearch/hermes-agent/pull/102085#pullrequestreview-5100984119
[4] https://github.com/NousResearch/hermes-agent/pull/95101
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
