# x402 Reconciliation Pattern (Lot E4)

Standard flow for building an offline-safe reconciliation layer for UNKNOWN_OUTCOME submission attempts. Reusable template for Lot E5 and any future LAHB reconciliation task.

## Flow

```
E2/E3 UNKNOWN_OUTCOME / protocol error + reconciliationRequired
        ↓
1. Eligibility check   ← accept only legitimate unresolved outcomes
        ↓
2. Identifier selection ← remoteEntryId > remoteRequestId > idempotencyKey
        ↓
3. Deterministic request ← build immutable request + canonical identity
        ↓
4. Injected transport    ← explicitly provided, no default
        ↓
5. Single lookup         ← at most one network call
        ↓
6. Response normalization ← normalize remote record (dual naming: snake_case + camelCase)
        ↓
7. Status classification  ← PENDING/SUBMITTED/REJECTED/APPROVED(untrusted)
        ↓
8. Immutable result       ← deep frozen, hard-coded safety constraints
```

## Key Architecture Decisions

| Decision | Rule | Rationale |
|----------|------|-----------|
| Eligibility gate | `UNKNOWN_OUTCOME` or `PROTOCOL_ERROR+unknownOutcome+reconciliationRequired` | Prevents wasting lookups on already-settled outcomes |
| Identifier chain | remoteEntryId → remoteRequestId → idempotencyKey | Most specific identifier first; entry ID has lowest false-match risk |
| Single lookup max | ≤ 1 | No fallback chains; errors preserve unresolved state |
| Transport contract | Explicitly injected async function | No default network, no credential auto-discovery |
| Read-only constraints | Hard-coded `false` for submission/retry/mutation/approval | Cannot be overridden by caller |
| Remote APPROVED | → `FOUND_SUBMITTED` + `approvalValidationRequired=true` | Never becomes validated approval authority; Lot E5 must validate |
| Not-found | → `reconciliationRequired=true` (conservative) | Eventual consistency means absence ≠ definitive absence |

## LAHB API Field Naming

LAHB endpoints use **snake_case**; the x402 code model uses **camelCase**. Every boundary adapter must normalize both:

| LAHB (snake_case) | x402 model (camelCase) |
|--------------------|------------------------|
| `entry_id` | `remoteEntryId` |
| `approval_request_id` | `remoteRequestId` |
| `status` | `remoteStatus` |
| `protocol_version` | `protocolVersion` |
| `created_at` | `createdAt` |
| `updated_at` | `updatedAt` |

## Pitfalls

- **Contradiction check before classification**: A 2xx response with `accepted: false` or `rejected: true` is self-contradictory. Detect this BEFORE calling status classification and override the result to PROTOCOL_ERROR.
- **not-found ≠ definitely-not-submitted**: A 404 means the record is absent *right now*. Due to eventual consistency, this does not mean it was never submitted. Never set `resubmissionAllowed = true` from a single not-found.
- **Remote APPROVED ≠ validated approval**: The remote status field is a claim, not a proof. Lot E5 must independently verify bindingId, packetId, purchaseIntentId, approver identity, role, second consent, expiry, and receipt authenticity.

## Reference Implementation

Files in `lah-openclaw-mvp/src/services/x402/reconciliation/`:
- `reconciliation-context.js` — Schema + lookup strategies
- `reconciliation-request.js` — Eligibility + identifier selection + builder + identity
- `reconciliation-result.js` — Classifications + status mapping + normalizeRemoteRecord
- `injectable-lahb-reconciliation-adapter.js` — Orchestration entry point

Created during `CLOE_X402_LOT_E4_UNKNOWN_OUTCOME_AND_APPROVAL_RECONCILIATION_V1`.
