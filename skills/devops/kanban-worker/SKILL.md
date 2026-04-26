---
name: kanban-worker
description: How a Hermes profile should work a task from the shared Kanban board. Load this skill in any profile that participates in the board (researcher, backend-eng, reviewer, etc.). Triggers on HERMES_KANBAN_TASK env var or a "work kanban task <id>" prompt.
version: 1.0.0
metadata:
  hermes:
    tags: [kanban, multi-agent, collaboration, workflow]
    related_skills: [kanban-orchestrator]
---

# Kanban Worker

Use this skill when you were spawned to work a task from the shared Hermes Kanban board. Symptoms:

- Your initial prompt says "work kanban task <id>" — e.g. `work kanban task t_9f2a`.
- Env vars set: `HERMES_KANBAN_TASK`, `HERMES_KANBAN_WORKSPACE`, optionally `HERMES_TENANT`.
- You were started by `hermes kanban dispatch` (cron) or a human ran `hermes -p <profile> chat -q "work kanban task <id>"`.

## Your job

You are **one run of one specialist profile working one task.** Read the task, do the work inside the workspace, record a result, and exit. Everything else is somebody else's job.

## Step 1 — Read the full context

```bash
hermes kanban context $HERMES_KANBAN_TASK
```

That command prints:
1. Task title + body.
2. Every comment on the task, in order, with author names.
3. Completion results of every `done` parent task (upstream context).

**Read all of it.** The comment thread is the inter-agent protocol — past peers, human clarifications, and blocker resolutions all live there. If a reviewer left feedback or the user answered a blocker, it's in the comments.

## Step 2 — Work inside the workspace

`cd $HERMES_KANBAN_WORKSPACE` and do the work there. The workspace kind determines what that means:

| `workspace_kind` | What it is | Your behavior |
|---|---|---|
| `scratch` | Fresh temp dir, yours alone | Read/write freely; it gets GC'd when the task is archived. |
| `dir:<path>` | Shared persistent directory | Treat as a long-lived workspace; other runs will read what you write. |
| `worktree` | Git worktree at the resolved path | You may need to `git worktree add <path> <branch>` if it doesn't exist yet. Commit work here. |

For `worktree` mode: check if `.git` exists in the workspace path. If not, run:
```bash
git worktree add $HERMES_KANBAN_WORKSPACE
```
from the main repo's root. Then cd and work normally.

## Step 3 — If tenancy matters, respect it

If `$HERMES_TENANT` is set, the task belongs to that tenant namespace. When reading or writing persistent memory, prefix memory entries with the tenant name so context doesn't leak across tenants:

> Good: memory entry `business-a: Acme is our biggest customer`
> Bad: unprefixed `Acme is our biggest customer` (leaks across tenants)

## Step 4 — If you hit an ambiguity you can't resolve, BLOCK. Don't guess.

Any of these should trigger a block:
- User-specific decision you can't infer (IP vs. user-id keys; which tone to use).
- Missing credential or access.
- Source that needs human input (paywalled article, 2FA-gated login).
- Peer profile needs to deliver something first and you can't reach around that.

```bash
hermes kanban block $HERMES_KANBAN_TASK "need decision: IP vs user_id for rate limit key?"
```

`block` also appends your reason as a visible comment. When the user or a peer unblocks and the dispatcher re-spawns you, you'll see the full comment thread including their answer in step 1's context read.

## Step 5 — Complete with a crisp, machine-readable result

```bash
hermes kanban complete $HERMES_KANBAN_TASK --result "rate_limiter.py implemented; keys on user_id with IP fallback; tests passing"
```

Rules for the `--result` string:
- One to three sentences. It's not a report, it's a handoff note.
- Name concrete artifacts you produced (file paths, URLs, commit SHAs).
- State any caveats a downstream profile needs to know.
- **Do not** include secrets, tokens, or raw PII — results are durable in the board DB forever.

Downstream tasks (children linked from this task) will see your `--result` verbatim as part of their parent-result context.

## Step 6 — If follow-up work is obvious, create it. Don't do it.

You are one task. If you notice something else needs doing, create a linked child task for the right profile instead of scope-creeping:

```bash
hermes kanban create "add concurrent-request test" \
    --assignee backend-eng \
    --parent $HERMES_KANBAN_TASK
```

## Leave comments to talk to peers

If you want to flag something for a reviewer, a future run, or the user — append a comment:

```bash
hermes kanban comment $HERMES_KANBAN_TASK "note: skipped the sqlite driver path; needs separate task"
```

Comments are the inter-agent protocol. Direct IPC does not exist; the board is the only channel.

## Do NOT

- Do not call `delegate_task` as a substitute for creating kanban tasks — `delegate_task` is for short synchronous reasoning subtasks inside your own run, not for cross-agent handoffs.
- Do not modify files outside `$HERMES_KANBAN_WORKSPACE` unless the task body explicitly asks for it.
- Do not assign tasks to yourself during your run (you're already running one; create new tasks for follow-ups only).
- Do not complete a task you didn't actually finish. Block it instead.

## Pitfalls

**The task might already be blocked or reassigned when you start.** Between when the dispatcher claimed and when you actually booted up, circumstances can change. Always read the current state at step 1. If `hermes kanban show` reports the task is blocked or reassigned, stop — don't keep running.

**The workspace may already have artifacts from a previous run.** Especially for `dir:` and `worktree` workspaces, a previous worker may have written files that are incomplete or stale. Read the comment thread — it usually explains why you're running again.

**Your memory persists but the task result does not carry over automatically.** If you learn something that matters for future runs of this profile in other tasks, write it to your profile memory via the normal mechanism. Comments on the task are for humans and peers; memory is for your future self.
