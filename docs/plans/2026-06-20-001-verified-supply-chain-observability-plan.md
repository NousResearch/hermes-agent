---
title: "chore: Add verified supply-chain posture checks and prepare observability baseline"
status: active
date: 2026-06-20
type: chore
target_repo: hermes-agent
origin: verified multi-agent research and local audit
---

# chore: Add verified supply-chain posture checks and prepare observability baseline

## Summary

This plan implements the first safe slice of the verified Hermes improvement roadmap. The immediate change is intentionally edge-only: add scheduled/manual supply-chain posture checks and static tests that guard their configuration. It does not change the agent loop, tool schema, prompt construction, or runtime behaviour.

The next implementation slices should use the evidence gathered in `E:/d/codebase_analysis/00_verified_improvement_plan.md`, but should remain conservative: observability before async refactors, policy before new sandbox backends, and tests before production changes.

---

## Verified Baseline

- Python files: 2,278.
- Python lines: 1,088,514.
- Workflows: 18 before this plan.
- Existing supply-chain controls: Dependabot for GitHub Actions, OSV Scanner, uv lockfile check, narrow high-signal supply-chain diff scanner, SHA-pinned actions.
- Existing design constraints: keep core narrow, preserve prompt caching, avoid speculative hooks, prefer edge/plugin/CI changes before core agent changes.

---

## Requirements

- R1. Add advisory GitHub Actions static analysis for workflow security.
- R2. Add scheduled repository security posture scoring.
- R3. Keep new checks scheduled/manual only until false positives are tuned.
- R4. Keep external GitHub Actions pinned by full commit SHA.
- R5. Add static tests so workflow files cannot silently disappear or broaden permissions.

---

## Implementation Units

### U1. Add Zizmor workflow

**Files:**
- `.github/workflows/zizmor.yml`
- `tests/test_supply_chain_workflows.py`

**Approach:**
- Run `uvx zizmor .github/workflows`.
- Trigger only on `schedule` and `workflow_dispatch`.
- Grant only `contents: read`.
- Pin checkout/setup actions by commit SHA.

### U2. Add OpenSSF Scorecard workflow

**Files:**
- `.github/workflows/scorecard.yml`
- `tests/test_supply_chain_workflows.py`

**Approach:**
- Use the official `ossf/scorecard-action` pinned to a full commit SHA.
- Upload SARIF via `github/codeql-action/upload-sarif`, also pinned by full commit SHA.
- Trigger only on `schedule` and `workflow_dispatch`.
- Grant only `contents: read` and `security-events: write`.

### U3. Verification

**Commands:**

```bash
uv run --extra dev python -m pytest tests/test_supply_chain_workflows.py -q --tb=short -o 'addopts='
uv run --extra dev python -m pytest tests/test_lint_config.py tests/test_supply_chain_workflows.py -q --tb=short -o 'addopts='
```

---

## Deferred Slices

### Observability baseline

The current safe slice keeps observability contract-only and avoids speculative
runtime hooks. It adds static regression tests for the existing observer schema
and documentation contract:

- `tests/test_observability_contract.py`
- `hermes_cli.middleware.OBSERVER_SCHEMA_VERSION`
- `docs/observability/README.md`

Runtime tracing expansion remains deferred until there is a concrete consumer
and E2E test path.

### Sandbox policy

This slice does not change terminal backend defaults. During verification it
found and fixed an existing Windows-specific approval guard bug: absolute native
Windows Hermes home paths such as `C:\Users\...\AppData\Local\hermes\config.yaml`
were normalized after shell backslash stripping, collapsing them into
`C:Users...` before detection. The fix rewrites the resolved Hermes home before
backslash escape stripping and supports both `\` and `/` path variants.

Covered by:

- `tests/tools/test_approval.py::TestHermesConfigWriteProtection`

The next safe step is documenting and testing existing approval/sandbox
invariants around `tools/approval.py`, `agent/tool_dispatch_helpers.py`, and
`tools/terminal_tool.py` before changing backend behaviour.

### Async refactor

Do not start until tracing and targeted regression tests exist. Begin with `tool_executor` compatibility, then model API calls, then streaming/cancellation.

---

## Success Criteria

- New workflow YAML parses.
- Static tests pass.
- New workflows are advisory/manual-scheduled, not required on every PR.
- All external actions in the new workflows are pinned to full commit SHAs.
