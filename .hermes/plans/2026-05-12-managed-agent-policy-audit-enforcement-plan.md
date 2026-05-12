# Managed-Agent Policy Audit Enforcement

Status: in_progress
Owner: Hermes main / Multica Codex worker
Started: 2026-05-12 09:10 +08
Branch: feat/managed-agent-policy-audit-enforcement
Depends on: PR #24075 / feat/managed-agent-policy-guardrails
Multica project: Hermes Agent / 966bde1c-667a-42e9-8088-56e3cbe97b79
Multica parent: JEF-225 / b1e1a621-c210-4996-9ad7-a36573bcf05d
Multica implementation: JEF-226 / 82e76fe0-bbfe-423d-91e7-f0f80a2c2c8a
Multica review gate: JEF-227 / 7d68a1c1-acfd-4101-a85b-9649f66d629d

## Scope

Continue Managed-Agent/Kanban worker work after Phase 2 policy-contract guardrails. Phase 3 turns the explicit policy contract into auditable runtime evidence without claiming OS/container sandboxing.

Implement bounded post-run policy audit for local Kanban workers:

- Preserve Phase 2 honest boundary: local workers remain contract/path scoped; `os_sandbox=false` unless a future backend proves otherwise.
- For policies that forbid edits (`read_only`, `test_only`), compare pre-run workspace evidence against post-run evidence when a worker completes or blocks.
- Store policy audit results in run metadata and task events so supervisors/operators can inspect violations.
- Surface concise audit status in CLI/task JSON if the existing shape has a natural place for it.
- Block or clearly flag successful completion when a no-edit policy changed workspace state, without running destructive rollback commands automatically.

## Non-goals

- No Docker/VM/container/OS sandbox implementation.
- No automatic rollback/destructive cleanup.
- No user config mutation.
- No production deploy/restart.
- No secrets, `.env`, tokens, DB dumps, raw media/model weights in `.hermes`.
- No merge to upstream/main unless Dragon explicitly authorizes it.

## Candidate affected files

- `.hermes/plans/2026-05-12-managed-agent-policy-audit-enforcement-plan.md`
- `.hermes/tasks/2026-05-12-managed-agent-policy-audit-enforcement.md`
- `hermes_cli/kanban_db.py`
- `hermes_cli/kanban.py` if CLI visibility needs a small addition
- `tests/hermes_cli/test_kanban_db.py`
- `tests/hermes_cli/test_kanban_cli.py` if CLI visibility changes

## Acceptance Criteria

- PR #24075 is checked before Phase 3 starts; any active comments/check failures are handled or explicitly recorded as absent.
- Phase 3 branch and `.hermes` ledger are committed/pushed before assigning implementation work.
- Policy audit helper compares bounded evidence for git and manifest workspaces.
- `read_only` and `test_only` completions with workspace mutations cannot silently pass as clean completions.
- Audit outcome is persisted in run metadata and task events with enough detail to diagnose changed paths/status.
- Existing policy contract fields remain honest (`policy_enforced_by=contract`, no OS sandbox claim).
- Focused Kanban DB/CLI tests pass; local patch protection tests pass; `git diff --check` is clean.

## Verification plan

```bash
source venv/bin/activate
python -m pytest -q tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_cli.py -o 'addopts='
python -m pytest -q tests/test_hermes_memory_provider.py tests/agent/test_auxiliary_temperature_retry.py tests/agent/test_prompt_builder.py -o 'addopts='
git diff --check
```

## Discussion / decisions

- 2026-05-12: Dragon instructed to handle PR #24075 and then go directly to Phase 3.
- 2026-05-12: PR #24075 checked before Phase 3. It is open, mergeable, and has no review comments/check failures to address at start of Phase 3.
- 2026-05-12: Phase 3 scope chosen as policy audit enforcement rather than real OS sandboxing, because Phase 2 intentionally stopped at honest contract/path-scoped local workers.
- 2026-05-12: Multica trace created: parent JEF-225, implementation JEF-226, review gate JEF-227.

## Status log

- 2026-05-12 09:10 +08 — in_progress: branch created and Phase 3 plan drafted after checking PR #24075.
- 2026-05-12 09:10 +08 — in_progress: Multica trace created and linked in ledger; implementation not assigned until ledger commit is pushed for agent visibility.
