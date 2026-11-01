# orchestrator_stats

> This file is the Orchestrator session brain. It persists progress across interactions and allows work to resume if the session drops or context runs out.
>
> **Rule**: The Orchestrator must read this file at startup and write to it at the end of every phase.
>
> **Autonomy principle**: this file should contain enough context for the Orchestrator to decide the next action without asking the user for information already captured here.

---

## Session

- **started_at**: `YYYY-MM-DD HH:MM:SS`
- **current_phase**: `Phase 3` | `Phase 4` | `Phase 5`
- **repository**: `<repo-name>`
- **branch**: `<active-branch>`
- **last_updated**: `YYYY-MM-DD HH:MM:SS`

---

## Project Context (auto-discovered)

Fill once at the beginning. Do not re-ask if already present.

- **stack**: `.NET` | `Python` | `Angular/TypeScript` | `Node` | `Java` | `Go` | `Other`
- **test_command**: `dotnet test` | `pytest -v` | `npm test -- --watch=false` | `...`
- **build_command**: `dotnet build` | `npm run build` | `python -m build` | `...`
- **lint_command**: `dotnet format --verify-no-changes` | `ruff check .` | `npx eslint .` | `...`
- **coverage_target**: `80` (percent)
- **package_manager**: `npm` | `yarn` | `pnpm` | `bun` | `pip` | `poetry` | `nuget`

---

## Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| `auto_t1` | `true` | Auto-execute Tier 1 (Fast Path) tasks without human prompt |
| `auto_t2` | `true` | Auto-execute Tier 2 (Batch) tasks and report at batch end |
| `ask_t3` | `true` | Always ask before Tier 3 (Strategic) tasks |
| `parallel_limit` | `2` | Maximum parallel worktrees/subagents |
| `worktree_threshold_minutes` | `10` | Single task exceeding this uses a dedicated worktree |
| `checkpoint_interval` | `3` | Run sanity checkpoint every N completed tasks |
| `halt_on_test_failure` | `true` | Stop DAG on any test failure |

---

## Identified Gaps (Phase 3)

| # | ID | Dimension | Severity | Description | Risk Tier | Status |
|---|----|-----------|----------|-------------|-----------|--------|
| 1 | `GAP-001` | Security | P1 | ... | T3 Blocking | 🔴 open |
| 2 | `GAP-002` | Architecture | P2 | ... | T2 Batchable | 🟡 queued |
| 3 | `GAP-003` | Lint | P4 | ... | T1 Auto | 🟢 done |

---

## Tasks (Phase 4 — DAG Queue)

### ID Convention

- `TASK-<NNN>` — unique identifier.
- `depends_on: ["TASK-<MMM>", ...]` — list of predecessor IDs.
- `issue_ref` — GitHub Issue number (#123).
- `spec_ref` — linked `.specs/SPEC-{YYYYMMDD}-{slug}.md`.
- `tier` — `T1` | `T2` | `T3`.
- `skill` — slash command or skill to invoke.
- `isolation` — `inline` | `worktree`.

### Pending Tasks

```yaml
- id: TASK-001
  desc: "Extract email validation into validators/email.ts"
  tier: T2
  skill: /execute-specs
  gap_ref: GAP-002
  issue_ref: "#101"
  spec_ref: ".specs/SPEC-20260908-email-validation.md"
  depends_on: []
  isolation: inline
  status: ready

- id: TASK-002
  desc: "Create tests for validators/email.ts"
  tier: T2
  skill: /qa-analyst
  gap_ref: GAP-002
  issue_ref: "#102"
  spec_ref: ".specs/SPEC-20260908-email-validation.md"
  depends_on: [TASK-001]
  isolation: inline
  status: blocked

- id: TASK-003
  desc: "Update auth architecture decision"
  tier: T3
  skill: /write-specs
  gap_ref: GAP-002
  issue_ref: "#103"
  spec_ref: ".specs/SPEC-20260908-auth-architecture.md"
  depends_on: []
  isolation: worktree
  status: ready
```

### Completed Tasks

```yaml
- id: TASK-000
  desc: "Example of completed task"
  tier: T1
  skill: /create-agent-harness
  gap_ref: GAP-000
  issue_ref: "#100"
  spec_ref: ".specs/SPEC-20260908-harness.md"
  depends_on: []
  isolation: inline
  status: done
  completed_at: "YYYY-MM-DD HH:MM:SS"
  validation: "PASS"
```

---

## Autonomous Decisions Log

Record every decision the Orchestrator made without human input. Include the reason and outcome so the user can audit later.

| # | Timestamp | Task | Decision | Reason | Outcome |
|---|-----------|------|----------|--------|---------|
| 1 | `YYYY-MM-DD HH:MM:SS` | TASK-001 | Auto-execute T1 lint fix | File count = 1, no logic change, `npm test` green | PASS |
| 2 | `YYYY-MM-DD HH:MM:SS` | TASK-002 | Escalate to T3 | New public API introduced | Awaiting approval |

---

## Backlog

Unapproved ideas, discovered scope or follow-ups that are **out of the current approved scope**.

| # | Description | Source | Proposed Tier | Status |
|---|-------------|--------|---------------|--------|
| 1 | Add e2e coverage for checkout flow | TASK-002 review | T2 | pending_approval |

---

## Sanity Checkpoints (Phase 4.5)

| # | After Task | Date | Reassessment Needed? | Action Taken |
|---|------------|------|----------------------|--------------|
| 1 | TASK-003 | `...` | No | Continue queue |
| 2 | TASK-006 | `...` | Yes — context changed | Re-run partial Phase 3 |

---

## Self-Applicable Actions Log (Tier T1)

| # | Date | Gap | Action | Result |
|---|------|-----|--------|--------|
| 1 | `...` | GAP-003 | Fix typo in README.md | Clean `git diff` |
| 2 | `...` | GAP-004 | Apply prettier to `src/` | No lint errors |

---

## Execution Batch (Tier T2)

> Batchable tasks awaiting batch approval or in progress.

| # | Gap | Task | Skill | Status |
|---|-----|------|-------|--------|
| 1 | GAP-002 | Create unit tests for `auth.ts` | /execute-specs | 🟡 pending_approval |

---

## Security Audit Log

Record security-relevant events: credential access, plugin/user/option mutations, `sudo`/`wp eval`/arbitrary code execution, external downloads, and prompt-injection escalations.

| # | Timestamp | Action | Approval Reference | Target | Outcome |
|---|-----------|--------|--------------------|--------|---------|
| 1 | `YYYY-MM-DD HH:MM:SS` | `plugin_activate` | approved in session X | `$WP_PATH` | `PASS` |
| 2 | `YYYY-MM-DD HH:MM:SS` | `wp_eval` | approved in session X | `mwai_options` | `PASS` |

## Metrics

Track time and cost to improve future estimates.

- **tasks_started**: `0`
- **tasks_completed**: `0`
- **tasks_blocked**: `0`
- **human_interventions**: `0`
- **validation_failures**: `0`
- **estimated_remaining_minutes**: `0`
