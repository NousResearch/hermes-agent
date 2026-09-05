# Hermes Tag additive kernel

This package is the behavior-neutral HT-01 substrate for Hermes Tag: a
profile-local governance, identity, continuity, provenance, approval, budget,
lease, and receipt authority.

It deliberately does **not** import or mutate `gateway.run`, a platform adapter,
the tool executor, provider request construction, or any external-effect path.
Those seams are separate source-pinned integration PRs. Until an effect boundary
calls `verify_effect()` immediately before the effect, existing Hermes behavior
remains authoritative and no runtime-governance claim is made.

## Components

- `model.py` — immutable canonical identities, scopes, intents, decisions,
  leases, facts, continuity envelopes, and admissions.
- `ledger.py` — profile-local atomic SQLite state, one-time provider-event
  reservation, replay fences, budgets, approvals, and hash-chained receipts.
- `identity.py` — tenant-qualified external aliases and explicit rebinding.
- `continuity.py` — isolated/principal/workspace/project/explicit continuity,
  optimistic checkpoints, and loop/replay rejection.
- `omniscience.py` — provenance-required facts, scope and sensitivity filtering,
  supersession, and equal-authority conflict preservation.
- `capability.py`, `policy.py`, `enforcement.py`, `obligations.py` — registry
  metadata, deny-overrides policy, exact one-time approvals, atomic reservations,
  short-lived HMAC leases, and pre/post-effect obligations.
- `middleware.py`, `runtime.py`, `bridge.py`, `service.py`, `kernel.py` —
  transport-authorized admission, task-local authority, current gateway identity
  conversion, profile-local construction, and the composed public facade.

## Invariants

1. Unknown capabilities and incomplete scopes fail closed.
2. Profile, platform, tenant/workspace, chat, actor, and continuity identity are
   never collapsed into one ambiguous key.
3. A caller may raise risk/effect metadata but cannot downgrade registry truth.
4. Deny overrides allow; high and critical actions require an exact one-time
   approval by default.
5. Budget capacity is reserved atomically before authority is issued.
6. Leases bind the exact principal, continuity, capability, intent digest, scope
   digest, decision, approval, budget reservation, obligations, issuance, expiry,
   and nonce. Each lease is reserved once immediately before its effect and may be
   completed only once.
7. Fact retrieval filters scope and sensitivity before prompt rendering and
   keeps equal-authority disagreements visible.
8. Provider event replay, continuity replay, stale writes, and receipt-chain
   tampering are explicit failures.
9. Signing material is resolved by reference and is never stored in SQLite or
   receipts.
10. Shadow admission may fail open only as observation; effect enforcement never
    does.
11. Task-local authority is one atomic admission/decision/lease tuple. Rebinding
    an admission or decision clears authority derived from the prior tuple, and
    cross-principal, cross-continuity, cross-scope, or mismatched leases are
    rejected before binding.
12. Approval creation is itself a governed `approval.grant` effect. The public
    kernel requires an authenticated durable principal, explicit policy allow,
    an exact argument-bound lease, pre-effect evidence, one-shot completion, and
    rollback if the governing effect cannot complete. Raw approval storage is
    not exported through the package or kernel facade.

Campaign authority: NousResearch/hermes-agent#79772. Slack flagship:
NousResearch/hermes-agent#80338. Executable campaign ledger:
NousResearch/hermes-agent#91036.

## Provenance boundary

This is a current-main reconstruction of the source-pinned HT-01 contract, not a
claim that inaccessible historical archive bytes were recovered unchanged. The
August 14 packet's public metadata, file inventory, architecture, integration
plan, and verification receipts define provenance and expected behavior; this
implementation was independently rebuilt, hardened, and tested against the
current repository contract. Runtime insertion remains HT-02 and later.
