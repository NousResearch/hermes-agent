---
title: "docs: Add sandbox and approval policy contract"
status: active
date: 2026-06-20
type: docs
target_repo: hermes-agent
origin: batch-002 multi-agent sandbox/approval audit
---

# docs: Add sandbox and approval policy contract

## Summary

This batch documents and regression-tests the current sandbox and approval
policy without changing runtime behavior. It is intentionally narrower than a
backend refactor: the goal is to make the existing contract explicit before any
future sandbox, terminal, or async execution changes.

The batch follows the deferred note from
`docs/plans/2026-06-20-001-verified-supply-chain-observability-plan.md`: test
and document approval/sandbox invariants around `tools/approval.py`,
`tools/path_security.py`, and terminal backend configuration before changing
backend behavior.

## Multi-Agent Audit Findings

Eight parallel review agents inspected:

- `tools/approval.py`
- `agent/tool_dispatch_helpers.py`
- `tools/terminal_tool.py`
- `tools/path_security.py`
- `SECURITY.md`
- `docs/security/`
- `cli-config.yaml.example`
- `website/docs/user-guide/configuration.md`

Consensus:

1. Do **not** change runtime terminal backend behavior in this slice.
2. Do **not** add speculative enforcement hooks.
3. Add a policy document that clearly says the approval gate is heuristic.
4. Add static contract tests for current defaults, dangerous/hardline patterns,
   path validation, and documentation links.
5. Defer `agent/tool_dispatch_helpers.py` heuristic changes to a separate TDD
   bugfix if needed.

## Requirements

- R1. Document OS-level isolation as the real boundary.
- R2. Document terminal-backend isolation versus whole-process wrapping.
- R3. Document `config.yaml` approval defaults and `command_allowlist`.
- R4. Document Docker mount and `docker_extra_args` review requirements.
- R5. Link the new policy from `SECURITY.md` and the configuration guide.
- R6. Add regression tests that fail if the contract or links disappear.
- R7. Make no runtime behavior changes.

## Implementation Units

### U1. Contract tests

**File:** `tests/test_sandbox_policy_contract.py`

Covers:

- default approval config values,
- dangerous/hardline pattern floor,
- `tools.path_security` traversal and escape behavior,
- existence and required phrases in the policy doc,
- cross-links from `SECURITY.md` and configuration docs.

### U2. Policy document

**File:** `docs/security/sandbox-approval-policy.md`

Sections:

- threat model,
- approval gate policy,
- terminal-backend isolation,
- whole-process wrapping,
- path/file safety helpers,
- operator review requirements,
- limitations and out-of-scope,
- related docs.

### U3. Cross-links

**Files:**

- `SECURITY.md`
- `website/docs/user-guide/configuration.md`

Add links to the new policy without broad rewrites.

## Verification Commands

```bash
uv run --extra dev python -m pytest tests/test_sandbox_policy_contract.py -q --tb=short -o 'addopts='
uv run --extra dev python -m pytest tests/test_supply_chain_workflows.py tests/test_observability_contract.py tests/test_sandbox_policy_contract.py tests/tools/test_approval.py::TestHermesConfigWriteProtection -q --tb=short -o 'addopts='
uv run --extra dev ruff check tests/test_sandbox_policy_contract.py
uv lock --check
git diff --check
```

## Deferred Work

- Improve `_is_destructive_command()` in `agent/tool_dispatch_helpers.py` with a
  separate RED/GREEN bugfix if batching heuristics need broader command
  detection.
- Add direct terminal backend runtime changes only after E2E coverage and a
  concrete sandbox policy migration plan exist.
- Consider dedicated unit tests for more Windows, UNC, symlink, and profile
  path-security cases in a separate patch.
