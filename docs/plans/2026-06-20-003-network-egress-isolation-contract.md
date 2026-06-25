---
title: "docs: Add network egress isolation contract"
status: active
date: 2026-06-20
type: docs
target_repo: hermes-agent
origin: batch-003 network egress isolation audit
---

# docs: Add network egress isolation contract

## Summary

This batch (Batch 003) documents and regression-tests the current network egress
isolation guidance without changing runtime behavior. It is intentionally
narrower than a backend refactor: the goal is to make the existing
Docker network segmentation contract explicit.

The batch follows the pattern from
`docs/plans/2026-06-20-002-sandbox-approval-policy-contract.md`.

## Multi-Agent Audit Findings

Consensus:
1. Do **not** change runtime terminal backend or network behavior.
2. Do **not** add speculative enforcement.
3. Add static contract tests for the network-egress-isolation.md doc.
4. Add cross-links from SECURITY.md.
5. Make no runtime behavior changes.

## Requirements

- R1. Verify existence and key phrases in network-egress-isolation.md.
- R2. Verify cross-link from sandbox-approval-policy.md (already present).
- R3. Add cross-link from SECURITY.md to network-egress-isolation.md.
- R4. Add regression tests that fail if the contract or links disappear.
- R5. Make no runtime behavior changes.

## Implementation Units

### U1. Contract tests

**File:** `tests/test_network_egress_isolation_contract.py`

Covers:
- existence and required phrases in the network doc,
- cross-links from sandbox doc and SECURITY.md,
- existence of this plan doc.

### U2. Cross-links

**Files:**
- `SECURITY.md`
- (sandbox already links)

Add minimal link to the network doc.

### U3. Plan document

**File:** `docs/plans/2026-06-20-003-network-egress-isolation-contract.md`

## Verification Commands

```bash
uv run --extra dev python -m pytest tests/test_network_egress_isolation_contract.py -q --tb=short -o 'addopts='
uv run --extra dev ruff check tests/test_network_egress_isolation_contract.py
git diff --check
```

## Deferred Work

- Network egress changes only after E2E coverage and concrete plan.
- Combine with sandbox backends in separate work.
