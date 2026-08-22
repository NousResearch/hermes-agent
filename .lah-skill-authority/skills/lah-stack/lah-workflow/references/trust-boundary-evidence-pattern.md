# Trust-Boundary Evidence Pattern (Separate Provenance)

Introduced during `CLOE_X402_LOT_E5_AUTHENTICITY_APPROVER_AND_SECOND_CONSENT_BINDING_CLOSURE_V1`.

## Problem

A single untrusted input (e.g. a remote approval receipt) may contain fields that claim authenticity, approval, consent, or authority. Validating the claim by reading its own fields is circular: the receipt's `receiptAuthenticityEvidence` field cannot prove the receipt is authentic if the receipt is the only input.

## Solution

Require **separate evidence objects** for each trust boundary:

- **Trusted read provenance** — proves the receipt was obtained through a trusted read channel, not fabricated by the caller. Supplied separately from the receipt.
- **Trusted approver authority** — proves the approver's identity, role, and authority are backed by a known authorized source. Supplied separately.
- **Second-consent evidence** — proves second consent was obtained with full operation identity binding. Supplied separately.

## Rules

1. **Receipt fields alone cannot establish trust.** Any field named `*Authenticity*`, `*Evidence*`, `*Proof*`, or similar on the receipt is ignored for trust decisions. Only separate evidence objects count.

2. **Evidence objects must be cross-validated against the trusted context** (e.g. E4 reconciliation result). Provenance `reconciliationRequestId` must match the context's. Approver authority `targetAuthorityId` must match. Consent `submissionRequestId` must match.

3. **Gap order dependency.** Validation gates run in a specific order. If an earlier gate (e.g. provenance) fails, later gates (e.g. approver) are never reached. Tests must account for this: a test that removes approver authority while keeping provenance valid will fail at PROVENANCE, not at APPROVER, if the provenance check also depends on the same context fields.

4. **Provenance binds to its source, not to the caller's context.** The provenance's `targetAuthorityId` is validated against the E4 reconciliation result's `targetAuthorityId`, not against the caller's `validationContext.targetAuthorityId`. The caller may set a different target authority for policy purposes; provenance must match the reconciliation record that produced it.

5. **Identifier binding is strict when the trusted context provides them.** If the trusted context provides `remoteRequestId`, the receipt and provenance must too. Omission is a binding failure, not an allowed absence.

## When to Use

Any mission where:
- A remote claim (approval, payment, authorization) is received as a single opaque payload
- The payload contains self-referential authenticity fields
- The claim must be validated before a local authority is created
- Multiple independent trust dimensions exist (provenance, authority, consent, identity)

## Pitfalls

| Pitfall | Fix |
|---------|-----|
| Receipt `receiptAuthenticityEvidence` treated as proof | Ignore it for trust decisions; provenance must come separately |
| Approver role checked only against a hardcoded allowlist | Also validate against a separately supplied authority that grants the role |
| Second consent validated as a flat boolean | Require a separate consent evidence object with full operation identity binding (submissionRequestId, packetId, bindingId, purchaseIntentId, targetAuthorityId) |
| Remote IDs on receipt treated as optional when source provides them | Make them mandatory: source present + receipt absent/contradictory → binding failure |
| Test helpers freeze fixture data | Test input data must NOT be frozen — only production function results should be |
| `Object.prototype.__proto__` rejected by accessor check | Always exclude `Object.prototype` from prototype accessor detection |
