# Managed-Agent Policy Audit Enforcement

Status: review
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
- 2026-05-12 09:10 +08 — in_progress: Multica trace created and linked in ledger; ledger commit `2f622fb33` pushed to `fork/feat/managed-agent-policy-audit-enforcement`; JEF-226 moved to in_progress.
- 2026-05-12 09:12 +08 — in_progress: supervisor `f9d30fdf5eee` started; Multica auto-execution shows JEF-226 and JEF-227 in_progress.
- 2026-05-12 09:13 +08 — blocked: JEF-226 run `8f7d1d41-1cf6-4cc3-a758-a71710b77ec0` reported the Multica workspace only exposed `court-booking-management`, not Hermes Agent. The worker marked JEF-226 blocked and produced no code commit.
- 2026-05-12 09:15 +08 — blocked: JEF-227 run `555e2513-9073-41c2-8289-9c02616c00b0` completed with blocking review. It confirmed the correct Hermes Agent branch only contains Phase 3 `.hermes` plan/ledger, no implementation or focused tests. Reviewer evidence: Kanban DB/CLI `140 passed`, related regressions `135 passed, 1 skipped`, `git diff --check` clean.
- 2026-05-12 09:22 +08 — blocked: Supervisor tick confirmed parent JEF-225 is also blocked after comment-triggered run `a40dd6ad-800a-4f4c-962c-6e5c15f29b1f`. The run located the correct local Hermes Agent repo/branch but found no implementation diff. Next action is to re-route/reassign implementation to the correct repo context; no merge/deploy/config mutation performed.
- 2026-05-12 09:30 +08 — review: Controller found and verified the intended implementation in the canonical Hermes Agent repo after the Multica workspace misroute. Implementation commit `c1fa47b37` adds bounded no-edit policy audit metadata/events and blocks dirty `read_only`/`test_only` completions without claiming OS/container sandboxing. Verification: Kanban DB/CLI `142 passed`; related regressions `135 passed, 1 skipped`; `git diff --check` clean. Formal JEF-227 review is being rerun against the committed implementation.
