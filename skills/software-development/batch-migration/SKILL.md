---
name: batch-migration
description: Orchestrate large-scale parallel code migrations using isolated git worktrees and background sub-agents. Breaks down massive refactors into 5-30 independent work units.
category: software-development
---

# Batch Migration Workflow

Use this skill when the user asks to "batch migrate", "bulk refactor", or "migrate X to Y across the codebase" for a large number of files.

This workflow decomposes a massive code migration into multiple independent logical units (e.g. 5-30 units), spawns a separate Hermes background process for each unit in an **isolated git worktree**, and tracks their progress.

## Workflow Phases

### Phase 1: Research, Plan & Chunk
1. **Scope understanding**: Research the codebase to understand what needs to be migrated (patterns, call sites, conventions).
2. **Decomposition**: Break the work into completely independent logical units. Each unit MUST be:
   - Independently implementable (no shared state with other units).
   - Mergeable on its own without depending on another unit's PR.
   - Roughly uniform in size (e.g. per-directory or per-module).
3. **E2E test recipe**: Determine how workers should verify their changes (e.g., unit tests, build commands).
4. **Plan creation**: Use the `todo` tool to create a status table. Add one item per work unit.
5. **Approval**: Present the plan to the user and request approval via the `clarify` tool before proceeding.

### Phase 2: Spawn Workers (Execute)
Once approved, start spawning the workers in the background. Do this for each unit:

1. **Create Worktree**: Use the `terminal` tool to create a git worktree.
   ```bash
   git worktree add .hermes/worktrees/unit-01 -b batch/unit-01 main
   ```
2. **Spawn Worker**: Use the `terminal` tool with `background=true` to spawn a hermes agent in the worktree.
   Provide a **fully self-contained prompt** (overall goal, this unit's task, codebase conventions, e2e test recipe, and standard instructions).
   ```bash
   hermes -q "You are a batch migration worker.
   GOAL: <overall goal>
   YOUR TASK: Migrate <specific files/module>.
   CONVENTIONS: <conventions>
   TEST RECIPE: <test command>

   INSTRUCTIONS:
   1. Simplify — Invoke the simplify skill to review and clean up your changes.
   2. Run tests — Run the test suite. Fix if failed.
   3. Commit and push — Commit all changes, push the branch.
   4. Create PR — gh pr create --title 'Batch: Unit 01' --body '...'
   5. Report — End with a single line: 'PR: <url>'"
   ```
   *Make sure to pass `workdir='.hermes/worktrees/unit-01'` and `background=true`.*

### Phase 3: Track Progress
1. Use the `process` tool (`action="poll"` or `"log"`) to check on the background hermes agents.
2. Update the `todo` tool status as workers complete (`running` -> `completed` / `failed`).
3. Parse the PR URL from the end of the worker's output and note it.
4. **Cleanup**: When a worker finishes, remove its worktree to save disk space.
   ```bash
   git worktree remove -f .hermes/worktrees/unit-01
   ```
5. Give the user a final summary with the list of merged/created PRs.