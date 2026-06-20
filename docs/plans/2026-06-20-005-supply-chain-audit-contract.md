---
title: "test: Guard supply-chain audit and Dependabot contracts"
status: active
date: 2026-06-20
type: test
target_repo: hermes-agent
origin: batch-005 supply-chain-audit/dependabot static audit
---

# Batch 005: supply-chain-audit/dependabot contract

## Summary

This batch adds static regression coverage for the existing
`supply-chain-audit.yml` and `.github/dependabot.yml` posture. It makes no
workflow behavior changes and no runtime behavior changes.

It extends the earlier supply-chain work documented in
`docs/plans/2026-06-20-001-verified-supply-chain-observability-plan.md`.

## Multi-Agent Audit Findings

Consensus:

1. Credential scoping is already covered by existing tests/docs.
2. Observability expansion is lower signal for this batch.
3. File safety bugfixes would be runtime work and should be deferred.
4. The uncovered high-signal gap is static coverage for:
   - `.github/workflows/supply-chain-audit.yml`
   - `.github/dependabot.yml`
5. This batch should be tests + plan only.

## Requirements

- R1. `supply-chain-audit.yml` keeps pull-request coverage for
  `opened`, `synchronize`, and `reopened`, without adding push/schedule/manual
  triggers.
- R2. `supply-chain-audit.yml` permissions stay narrow:
  `contents: read` and `pull-requests: write`.
- R3. `actions/checkout` usage in `supply-chain-audit.yml` is pinned to a
  full 40-character commit SHA.
- R4. The workflow keeps the required jobs: `changes`, `scan`, and
  `dep-bounds`.
- R5. The scan job remains gated on the `changes` job's `scan` output being
  true, without requiring exact shell/YAML formatting.
- R6. Critical scan patterns include `.pth`, base64 decode plus exec/eval,
  and encoded/obfuscated subprocess calls.
- R7. Dependency-bounds checks continue to inspect unbounded `>=` PyPI
  specifiers in `pyproject.toml`.
- R8. `.github/dependabot.yml` remains scoped to the `github-actions`
  ecosystem only, at directory `/`, on a weekly schedule, with constrained
  open PR count and `dependencies` / `github-actions` labels.
- R9. No pip/npm Dependabot ecosystems are enabled in this batch.
- R10. Make no workflow behavior changes.

## Implementation Units

### U1. Static workflow/dependabot tests

**File:** `tests/test_supply_chain_workflows.py`

Adds static tests for:

- `supply-chain-audit.yml` semantic/static trigger, permission, SHA pinning,
  job, gate, critical-pattern, and dep-bounds contracts without requiring
  exact shell text.
- `.github/dependabot.yml` ecosystem/schedule/label/limit contract.
- this Batch 005 plan document.

### U2. Plan document

**File:** `docs/plans/2026-06-20-005-supply-chain-audit-contract.md`

Records the batch scope and explicit no-behavior-change boundary.

## Verification Commands

```bash
uv run --extra dev python -m pytest tests/test_supply_chain_workflows.py -q --tb=short -o 'addopts='
uv run --extra dev ruff check tests/test_supply_chain_workflows.py
git diff --check
uv lock --check
```

## Deferred Work

- Runtime supply-chain enforcement changes.
- New advisory-only scanners.
- Dependabot ecosystems beyond `github-actions`.
