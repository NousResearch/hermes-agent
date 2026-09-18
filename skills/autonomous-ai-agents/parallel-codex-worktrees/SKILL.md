---
name: parallel-codex-worktrees
description: "Use when complex coding work can be split into low-conflict Codex worktree lanes that Hermes will verify and integrate."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [codex, worktrees, parallelism, orchestration, git]
    related_skills: [codex, kanban-codex-lane, kanban-worker, kanban-orchestrator]
---

# Parallel Codex Worktrees

Use this when a coding task is large enough to benefit from parallel implementation lanes and the work can be split into low-conflict chunks.

## Use when

All of these are true:

- The task is complex enough that orchestration overhead is worth it.
- The work can be decomposed into 2-4 mostly independent slices.
- Each slice has a clear file boundary, concern boundary, or acceptance test.
- You can verify each slice independently before integration.
- The repository is in git and supports isolated worktrees.

Do **not** use this workflow when:

- The task is tiny.
- Everything lands in the same hot file or the same fragile call path.
- Merge risk is higher than the expected speedup.
- The repo is already in a messy or unclear state that must be stabilized first.

Default to a single Codex lane when the task is not clearly parallelizable.

See `references/lane-selection-and-integration.md` for the quick lane-selection and verification checklist.

## Single-lane default

Unless the complexity clearly justifies a split, start with **one isolated Codex worktree lane**.

That means:

- create one dedicated worktree even for a single implementation lane
- keep Codex out of the shared checkout
- verify and integrate that one lane first
- expand to 2-4 lanes only when the task cleanly decomposes

Parallelism is the escalation path, not the default flex.

Bias even harder toward a single lane when the task is **phase/checklist-driven but cross-cutting** — for example one MVP phase that touches routes, coordinator flow, CLI surface, and focused tests along one backend path. If the work shares one acceptance boundary and one verification bundle, splitting it usually creates merge theater, not speed.

## Core rules

1. **Always use worktrees** for Codex implementation lanes. Never run parallel Codex edits in the same checkout.
2. **Use `gpt-5.4` by default** for Codex runs unless the user explicitly requests another model.
3. **Hermes is the orchestrator and integrator.** Codex workers may edit, test, and commit in their own worktrees, but Hermes verifies and decides what lands on the current branch.
4. **A printed diff is not success.** A lane only counts if it leaves a real commit on its branch and a clean worktree state.
5. **Verify before integrating.** Re-check branch state, diff shape, and targeted tests yourself.
6. **Merge into the current branch only after review.** Prefer cherry-pick or controlled reconciliation over blind merges.

## Standard flow

### 1. Read canon first

If the repo has plan docs, execution checklists, or MVP canon files, read them before decomposition. Feed those paths to every Codex lane.

### 2. Decide whether to parallelize

Ask:

- Can I define 2-4 scoped tasks with low merge conflict risk?
- Can each lane be validated with narrow tests?
- Will integration be simpler than doing the whole thing in one lane?

If not, use one worktree and one Codex lane.

### 3. Define slices by concern, not by arbitrary file count

Good slice patterns:

- page A cleanup vs page B cleanup
- API helper refactor vs UI test hardening
- backend endpoint vs frontend consumer when contracts are already known

Bad slice patterns:

- splitting one tightly coupled refactor across multiple workers
- giving multiple lanes the same hot component
- parallelizing before the acceptance criteria are clear

### 4. Create isolated worktrees

From the repo root:

```bash
git worktree add -b <branch-a> <worktree-a> <base>
git worktree add -b <branch-b> <worktree-b> <base>
```

Use a stable worktree base such as `~/.worktrees/<repo>/<task-name>/...`.

### 5. Launch Codex with hard success criteria

Use `codex exec -m gpt-5.4 --dangerously-bypass-approvals-and-sandbox ...` with `pty=true`.

Every lane prompt should include:

- exact repo/worktree/branch
- exact allowed files
- exact docs to read first
- what must stay untouched
- narrow acceptance criteria
- required targeted tests
- required commit message
- required proof before exit:
  - `git rev-parse --short HEAD`
  - `git status --short`
  - `git log --oneline -1`
- explicit statement that success requires:
  - `HEAD` changed from base
  - status is clean
  - one real commit exists

For long runs, use background terminal with notify-on-complete.

### 6. Verify each lane independently

For every completed lane:

1. check branch name and HEAD
2. confirm the expected commit exists
3. confirm `git status --short` is clean
4. inspect `git diff --stat <base>..HEAD`
5. inspect the changed files directly
6. re-run the narrowest useful tests yourself
7. reject any lane that only printed a patch or left dirty state

### 7. Integrate into the current branch

Hermes owns integration.

Preferred order:

1. integrate the cleanest, most independent commit first
2. reconcile overlapping lanes manually if needed
3. cherry-pick accepted commits or manually apply the accepted diff
4. verify the integrated branch after each step when risk is non-trivial

Do not tell the user a lane succeeded until you have verified it from git state, not just process output.

### 8. Final repo-level verification

After accepted lanes are integrated into the current branch:

- run focused tests for touched areas
- run broader build/test if warranted
- inspect `git status --short`
- summarize what landed, what was rejected, and any leftover conflicts or caveats

## Prompt template requirements

Each Codex lane prompt should explicitly say:

- this is an isolated git worktree
- scope is narrow and adjacent redesign is forbidden
- printing a diff is not success
- exactly one commit is required
- exit 0 is not enough if no commit exists

## Good defaults

- model: `gpt-5.4`
- worktrees: yes
- lane count: 2-4 when complexity justifies it
- tests: narrow per lane, then final integrated verification
- integration owner: Hermes

## Pitfalls

- launching parallel lanes for small work
- overlapping file ownership across lanes
- trusting stdout over branch state
- letting Codex self-certify completion
- merging lane branches before checking for dirty state
- forgetting that the current branch may already contain partial integration

## Output contract for the orchestrator

When reporting back, include:

- lanes launched
- worktree paths
- branch names
- verification result for each lane: accepted / rejected / partial
- commits integrated into the current branch
- final tests/build run
- any leftovers intentionally excluded
