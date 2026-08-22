# Cross-Layer Certified-Lot Discovery Pattern

When a read-only audit mission must understand how a certified component
(Lot N at commit X) integrates with infrastructure at a different layer
or revision (e.g. LAHB submission code on `main`, or a different repo),
use this pattern.

## When to use

The mission certifies a specific artifact (e.g. Lot D `ApprovalPreparationPacket`)
that is NOT yet wired into the broader runtime (e.g. LAHB submission system).
You need to determine:
- Does the existing infrastructure already handle this packet schema?
- If not, what is the gap? Adapter, repair, or both?
- Can an existing transport/client/queue be reused?

## Step 1 — Identify both revision targets

Two classes of code to read:

| Source | Where | What |
|--------|-------|------|
| **Certified artifact** | Mission-specific worktree at pinned commit | The new component being certified |
| **Broader infrastructure** | Canonical checkout (often `main` or latest) | The pre-existing system it must interact with |

**Don't assume they share the same revision.** The certified lot may be
at a different commit, branch, or even worktree than the infrastructure
code that evolved independently.

## Step 2 — Read the certified artifact first

```bash
cd /home/deploy/lah-stack-worktrees/<lot-worktree> && \
  git diff <parent> HEAD --name-only
```

Read every file in the diff. This is the bounded surface of the new
component. Understand:
- Its schema constants and version strings
- Its identity model (what IDs does it use? How are they derived?)
- Its state model (what statuses does it define?)
- Its immutability guarantees
- Its domain separation boundaries

## Step 3 — Read the broader infrastructure

Find all relevant files in the canonical checkout:

```bash
cd /home/deploy/lah-stack-repos/<repo> && \
  rg -l -i '<keyword>' --type js --type mjs | sort
```

For each relevant file, read its full content. Understand:
- Its schema constants and expected input format
- Its transport contract (endpoint, method, headers, auth)
- Its identity model (how does it identify requests?)
- Its state/status model
- Its idempotency and retry model
- Its failure model and UNKNOWN_OUTCOME handling
- Its reconciliation interfaces

## Step 4 — Produce the layer-gap analysis

Compare the two schemas on these axes:

| Axis | Certified artifact | Infrastructure | Compatible? |
|------|-------------------|---------------|:-----------:|
| Schema version string | — | — | NO / PARTIAL / YES |
| Identity model | — | — | NO / PARTIAL / YES |
| Primary descriptor | — | — | NO / PARTIAL / YES |
| Target model | — | — | NO / PARTIAL / YES |
| State model | — | — | NO / PARTIAL / YES |
| Idempotency | — | — | NO / PARTIAL / YES |
| Transport endpoint | (may not exist) | — | N/A → REUSE or NEW |
| Auth | (may not exist) | — | N/A → REUSE or NEW |
| Timeout | (may not exist) | — | N/A → REUSE or NEW |
| UNKNOWN_OUTCOME | (may not exist) | — | N/A → REUSE or NEW |
| Reconciliation | (may not exist) | — | N/A → REUSE or NEW |

## Step 5 — Reuse vs Adapter decision

Apply these rules:

| Finding | Decision |
|---------|----------|
| Same schema version + compatible identity | `DIRECT_REUSE_SAFE` |
| Different schema but compatible transport | `ADAPTER_REQUIRED` |
| Different identity model + different schema | `ADAPTER_REQUIRED` |
| Same transport but incompatible status model | `ADAPTER_REQUIRED` with status mapping |
| Incompatible idempotency model | New idempotency key derivation required |
| Infrastructure has automatic retry but new component must not | Use raw client (skip retry queue) |

The most common result for x402 lot-level missions is `ADAPTER_REQUIRED`
— the transport, auth, timeout, and reconciliation infrastructure is
reusable, but the packet schema, identity model, and status tracking
require a new adapter.

## Step 6 — Classify existing infrastructure for reuse

For every infrastructure surface found in Step 3, classify:

```
CANONICAL_ACTIVE   → Can be imported/called directly
ACTIVE_SUPPORTING  → Can be used but may need option overrides
TEST_ONLY          → Only exists in test fixtures
LEGACY             → Still present but being phased out
DEPRECATED         → Marked for removal
UNRESOLVED         → Uncertain role or ownership
```

## Step 7 — Document the adapter requirements

The adapter analysis should answer:

1. Which exact functions from existing infrastructure can be reused?
2. Which options/overrides does the adapter need (e.g. `_submitRaw` to skip queue)?
3. What new code must be written (payload builder, identity derivation, etc.)?
4. What interface does the adapter expose for the certified artifact?
5. What existing tests can be adapted vs new tests needed?
6. What configuration or feature flags must be added?

## Pitfalls

| Trap | Symptom | Fix |
|------|---------|-----|
| **Assuming same revision** | Agent reads Lot D code from worktree but reads LAHB code from wrong branch | Always verify HEAD SHA of both locations. Document the divergence. |
| **Schema constants mismatch** | Agent assumes `cloe_governed_action_packet_v1` is compatible with `cloe_x402_approval_packet_v1` because both have "packet" in the name | Compare exact schema version strings. A single byte difference means incompatible. |
| **Identity model conflation** | Agent maps `packetId → action_id` because both are "identifiers" | Map the IDENTITY CHAIN, not individual fields. x402 uses `purchaseIntentId → bindingId → packetId`. Governed-action uses `action_id → correlation_id`. |
| **Status model collapse** | Agent treats `SUBMISSION_ACCEPTED` as `APPROVED` | Distinguish transport acceptance (LAHB said "we got it") from human approval (someone said "yes"). |
| **Assuming retry queue is optional** | Agent builds adapter that uses `submitApproval` (with queue) instead of `submitApprovalRaw` (without queue) | Always check which submission function the infrastructure exports. Prefer the RAW variant for components that must not auto-retry. |
