---
name: simplicio-tasks
description: Autonomous work-item orchestrator — drains a queue of bugs/issues/tasks with auto-looping, evidence-gated completion, and capacity-based parallel execution. Use when clearing a backlog, processing multiple bug reports, or running CI fix waves.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [orchestration, automation, task-queue, loop]
    related_skills: [bug-pipeline, orchestration-patterns, task-router]
---

# Simplicio Tasks — Autonomous Queue Orchestrator

> Drains a queue of work-items (bugs, issues, CI failures, kanban cards) autonomously. Loops until the queue is empty, then exits with evidence.

## Trigger

Use when asked to clear a queue of work — "process these bugs", "drain the CI failures", "clear the kanban board", "implement milestone X tickets", "fix all failing tests". Also triggers on `/simplicio-tasks` pattern.

Do NOT use for:
- A single well-defined bug fix (use `bug-pipeline` instead)
- A single feature implement (use `engineer` profile directly)

## How it works

Simplicio runs as a **loop**. It:
1. Discovers open work-items from the source (GitHub issues, kanban board, CI failures)
2. Prioritizes and deduplicates them
3. Processes them in parallel up to capacity limits
4. Verifies each item is truly done with evidence
5. Loops until queue is empty or cap is hit

---

## Step 1: Intake — discover the work

Identify the work source. For each source, run the appropriate discovery:

**GitHub issues:**
```bash
gh-axi issue list --repo <owner/repo> --state open --limit 50 --json number,title,labels
```

**Kanban board:**
```bash
kanban-axi tasks --column <column>
```

**CI failures:**
```bash
gh-axi run list --repo <owner/repo> --workflow <name> --branch main --limit 10
```

Prioritize by: severity (P1 > P2 > P3) → age (oldest first) → dependencies.

Emit: `Intake: <N> items discovered, <M> actionable after dedup`

---

## Step 2: Route & Scale

Decide how many items to process in parallel based on:
- Available capacity (CPU/memory/API rate limits)
- Item complexity (P1 needs full pipeline, P3 may be fast-path)
- Dependencies between items

For independent items, fan them out with `delegate_task(tasks=[...])` using appropriate profiles:
- Bug fixes → `engineer` with `context="use bug-pipeline skill"`
- Research → `search` or `web_researcher`
- Code review → `think` or `debugger`

For dependent items, run them sequentially.

---

## Step 3: Auto-Loop

Simplicio IS a loop. After completing a batch:

1. Verify each completed item:
   - Bug fix → evidence it's fixed (test output, PR link)
   - Issue → evidence it's closed (API response, merged PR)
2. Re-check the source for remaining work
3. If items remain → process next batch
4. If queue is empty → emit `SIMPLICIO_DONE` with evidence

**Loop termination** (any of these):
- Queue is fully drained AND all items verified — emit `SIMPLICIO_DONE` with evidence
- `/stop` interrupt received
- Cap reached (max iterations or budget)

---

## Step 4: Verification Gate

Every claimed completion MUST have evidence:

| Claim | Evidence |
|-------|----------|
| "Bug is fixed" | PR link + test output showing pass |
| "Issue is closed" | GitHub API response or PR merged |
| "Tests pass" | `pytest` output showing 0 failures |
| "CI is green" | Workflow run URL showing success |

Forbidden: "should be fixed", "looks good", "appears to work".

---

## Companion Skills

| Skill | Purpose | When to load |
|-------|---------|--------------|
| `bug-pipeline` | 6-stage quality gate for individual bug fixes | Per bug item in queue |
| `simplicio-review` | Deep adversarial review of completed diffs | Before merging |
| `simplicio-orient` | Terminal-first execution & token economy | Heavy shell work |
| `simplicio-compress` | Compression for long-running sessions | When context fills |

These are NOT required — simplicio-tasks works standalone. Load them when extra rigor is needed.

---

## Pitfalls

| Problem | How to avoid |
|---------|-------------|
| Over-parallelization hitting API rate limits | Cap parallel tasks at 3 for external APIs |
| Context window overflow from long loops | Compress intermediate results; use lore-axi KB to store state |
| Committing untested changes | Always run tests after engineer returns |
| Creating competing PRs | Run duplicate check before implementing |
| Claiming done on partial evidence | Gate every item on verifiable proof |
