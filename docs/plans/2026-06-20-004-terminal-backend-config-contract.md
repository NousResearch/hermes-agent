---
title: "docs: Add terminal backend config contract"
status: active
date: 2026-06-20
type: docs
target_repo: hermes-agent
origin: batch-004 terminal backend config audit
---

# Batch 004: Terminal Backend Config Contract

## Summary

This batch documents and regression-tests the current terminal backend
configuration contract without changing runtime behavior. It is intentionally
narrower than a backend refactor: the goal is to make the existing contract
explicit before any future terminal backend changes. Batch 004 follows TDD
RED/GREEN with static tests/docs only.

The batch covers `terminal.backend` default (local), the warning for local,
docker/modal/daytona as non-local sandbox backends or documented backend
choices, and the exact documented config keys `terminal.docker_image`,
`terminal.docker_mount_cwd_to_workspace`, `terminal.modal_image`, and
`terminal.daytona_image`. It does not invent alias keys; the contract also
records that the local backend runs on host/no OS isolation.

## Multi-Agent Audit Findings

Review of `cli-config.yaml.example`, `website/docs/user-guide/configuration.md`,
and `docs/security/sandbox-approval-policy.md` confirmed:

1. Do **not** change runtime terminal backend behavior in this slice.
2. Do **not** add speculative enforcement.
3. Add static contract test that first fails on missing plan.
4. Document exact config keys and exact documentation claims.
5. Add cross-links between the plan, configuration.md, and sandbox policy.
6. Make no runtime behavior changes.

## Requirements

- R1. Document `terminal.backend` default/local warning.
- R2. Document that docker/modal/daytona are non-local sandbox backends or documented backend choices.
- R3. Use exact config keys from cli-config.yaml.example: `backend: "local"`, `docker_image`, `docker_mount_cwd_to_workspace`, `modal_image`, `daytona_image`.
- R4. Use exact documentation claims from configuration.md / sandbox policy: "The default. Commands run directly on your machine with no isolation.", "local` has no isolation. Commands run with the operator user's host access.", "docker`, `singularity`, `modal`, and `daytona` run commands in a configured sandbox target."
- R5. Link the new plan from configuration.md (cross-link) and sandbox policy.
- R6. Add regression tests that fail if the contract or links disappear.
- R7. Make no runtime behavior changes.

## Implementation Units

### U1. Contract tests

**File:** `tests/test_terminal_backend_config_contract.py`

Covers:
- plan existence and required contract phrases
- exact phrases in configuration.md
- exact keys in cli-config.yaml.example
- exact claims in sandbox-approval-policy.md

### U2. Plan document

**File:** `docs/plans/2026-06-20-004-terminal-backend-config-contract.md`

Sections:
- summary of batch 004 scope (static only)
- requirements using exact keys and claims
- implementation units
- verification commands

### U3. Cross-links

**Files:**
- `website/docs/user-guide/configuration.md` (add explicit cross-link to plan)
- `docs/security/sandbox-approval-policy.md` (existing link to configuration.md covers terminal backend config)

## Verification Commands

```bash
uv run --extra dev python -m pytest tests/test_terminal_backend_config_contract.py -q --tb=short -o 'addopts='
uv run --extra dev python -m ruff check tests/test_terminal_backend_config_contract.py
```

## Contract Phrases (high-signal, exact)

The following are the required terminal backend config contract concepts:

- terminal.backend default/local warning
- docker/modal/daytona are non-local sandbox backends or documented backend choices
- terminal.docker_image, terminal.docker_mount_cwd_to_workspace, terminal.modal_image, terminal.daytona_image
- no undocumented terminal backend alias keys are part of this contract
- local backend runs on host/no OS isolation
- '  backend: "local"'
- '  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"'
- '  docker_mount_cwd_to_workspace: false  # SECURITY: off by default.'
- '  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"'
- '  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"'
- "The default. Commands run directly on your machine with no isolation."
- "local` has no isolation. Commands run with the operator user's host access."
- "docker`, `singularity`, `modal`, and `daytona` run commands in a configured sandbox target."
- no runtime behavior changes
- static contract tests + plan/docs/cross-links for terminal backend configuration
