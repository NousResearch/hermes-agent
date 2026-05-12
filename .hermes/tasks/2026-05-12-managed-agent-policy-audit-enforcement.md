# Managed-Agent Policy Audit Enforcement Tasks

Status: in_progress
Owner: Hermes main / Multica Codex worker
Branch: feat/managed-agent-policy-audit-enforcement
Started: 2026-05-12 09:10 +08
Multica project: Hermes Agent / 966bde1c-667a-42e9-8088-56e3cbe97b79
Multica parent: JEF-225 / b1e1a621-c210-4996-9ad7-a36573bcf05d
Multica implementation: JEF-226 / 82e76fe0-bbfe-423d-91e7-f0f80a2c2c8a
Multica review gate: JEF-227 / 7d68a1c1-acfd-4101-a85b-9649f66d629d

## Tasks

### phase3-pr24075-check

- Status: done
- Owner: Hermes main
- Evidence: Checked PR #24075 after Dragon's instruction. PR is open, mergeable, has no review comments/check failures reported at start of Phase 3.

### phase3-plan-ledger

- Status: review
- Owner: Hermes main
- Acceptance:
  - Phase 3 plan and task ledger written under `.hermes/`.
  - Ledger committed and pushed to GitHub-visible feature branch before implementation assignment.
  - Important ledger files copied to NAS backup.

### phase3-multica-trace

- Status: done
- Owner: Hermes main
- Acceptance:
  - Parent issue created for Phase 3.
  - Implementation issue assigned to Codex/Claude execution lane.
  - Review gate issue assigned to architect/review lane.
  - Issue IDs recorded in this ledger and plan.
- Evidence: JEF-225 parent, JEF-226 implementation, JEF-227 review gate created on 2026-05-12.

### phase3-implementation

- Status: pending
- Owner: Multica Codex worker
- Acceptance:
  - Add bounded policy audit comparison for no-edit policies.
  - Persist audit outcome in task run metadata and task events.
  - Ensure no OS/container sandbox claims are introduced.
  - Add or update focused tests.

### phase3-review

- Status: pending
- Owner: Multica review lane
- Acceptance:
  - Reviewer inspects diff for safety, honesty of claims, and test coverage.
  - Reviewer reruns focused tests and records evidence.

### phase3-closeout

- Status: pending
- Owner: Hermes main
- Acceptance:
  - Controller reruns verification.
  - Ledger status updated with evidence.
  - Branch pushed; NAS backup refreshed.
  - PR created or updated according to Dragon's next instruction.

## Verification checklist

- [ ] `python -m pytest -q tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_cli.py -o 'addopts='`
- [ ] `python -m pytest -q tests/test_hermes_memory_provider.py tests/agent/test_auxiliary_temperature_retry.py tests/agent/test_prompt_builder.py -o 'addopts='`
- [ ] `git diff --check`

## Notes

- No Docker/VM/container/OS sandbox implementation in this phase.
- No automatic rollback/destructive cleanup.
- No user config mutation or deployment.
- Do not store secrets or raw private data in `.hermes/`.
