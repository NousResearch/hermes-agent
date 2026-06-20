---
name: kanban-worker
description: Pitfalls, examples, and edge cases for Hermes Kanban workers. The lifecycle itself is auto-injected into every worker's system prompt as KANBAN_GUIDANCE (from agent/prompt_builder.py); this skill is what you load when you want deeper detail on specific scenarios.
version: 2.0.0
platforms: [linux, macos, windows]
environments: [kanban]
metadata:
  hermes:
    tags: [kanban, multi-agent, collaboration, workflow, pitfalls]
    related_skills: [kanban-orchestrator]
---

# Kanban Worker — Pitfalls and Examples

> You're seeing this skill because the Hermes Kanban dispatcher spawned you as a worker with `--skills kanban-worker` — it's loaded automatically for every dispatched worker. The **lifecycle** (6 steps: orient → work → heartbeat → block/complete) also lives in the `KANBAN_GUIDANCE` block that's auto-injected into your system prompt. This skill is the deeper detail: good handoff shapes, retry diagnostics, edge cases.

## Workspace handling

Your workspace kind determines how you should behave inside `$HERMES_KANBAN_WORKSPACE`:

| Kind | What it is | How to work |
|---|---|---|
| `scratch` | Fresh tmp dir, yours alone | Read/write freely; it gets GC'd when the task is archived. |
| `dir:<path>` | Shared persistent directory | Other runs will read what you write. Treat it like long-lived state. Path is guaranteed absolute (the kernel rejects relative paths). |
| `worktree` | Git worktree at the resolved path | If `.git` doesn't exist, run `git worktree add <path> ${HERMES_KANBAN_BRANCH:-wt/$HERMES_KANBAN_TASK}` from the main repo first, then cd and work normally. Commit work here. |

## Tenant isolation

If `$HERMES_TENANT` is set, the task belongs to a tenant namespace. When reading or writing persistent memory, prefix memory entries with the tenant so context doesn't leak across tenants:

- Good: `business-a: Acme is our biggest customer`
- Bad (leaks): `Acme is our biggest customer`

## Repository code-change workflow

For repository code or documentation changes, the implementation worker owns the
branch and PR, not the merge:

1. Work in `$HERMES_KANBAN_WORKSPACE` on the assigned branch.
2. Commit, push, and open a PR against the correct base branch.
3. Do **not** merge, squash, rebase-merge, delete the branch, or enable
   auto-merge.
4. After the PR exists, create a separate reviewer Kanban task assigned to the
   reviewer profile. Include the PR URL/number, implementation task id, changed
   files, and verification performed. If this task is the implementation parent,
   pass `parents=[os.environ["HERMES_KANBAN_TASK"]]` so review cannot start
   before the implementation handoff exists.
5. Complete the implementation task with structured PR handoff metadata
   containing `pr_url`, `pr_number`, `changed_files`, and `verification`. Do
   **not** block it with `review-required` after creating a parent-gated reviewer
   task; a blocked parent keeps the reviewer task in `todo`.
6. The reviewer completes with a verdict or blocks with required changes.
   Required changes become a new implementation task linked from the reviewer
   task.
7. The final merge decision belongs to the user. A reviewer may approve or
   request changes, but neither reviewer nor implementer should merge without an
   explicit user merge instruction.

## Good summary + metadata shapes

The `kanban_complete(summary=..., metadata=...)` handoff is how downstream workers read what you did. Patterns that work:

**Coding task:**
```python
kanban_complete(
    summary="shipped rate limiter — token bucket, keys on user_id with IP fallback, 14 tests pass",
    metadata={
        "changed_files": ["rate_limiter.py", "tests/test_rate_limiter.py"],
        "tests_run": 14,
        "tests_passed": 14,
        "decisions": ["user_id primary, IP fallback for unauthenticated requests"],
    },
)
```

**Repository code-change task with PR review gate:**

When you open a PR and create a reviewer task with `parents=[your-task-id]`, complete the implementation task instead of blocking it. The completion row is the handoff that promotes the parent-gated reviewer task. Blocking the implementation task with `review-required` leaves the reviewer in `todo` forever because Kanban dependencies only promote when parents are `done`.

```python
review = kanban_create(
    title="review PR #123",
    assignee="reviewer-profile",
    body="Review PR #123: https://github.com/org/repo/pull/123. Check changed files, tests, and merge safety. Do not merge.",
    parents=[os.environ["HERMES_KANBAN_TASK"]],
)
kanban_complete(
    summary="PR #123 opened and reviewer task created; no merge/auto-merge performed.",
    metadata={
        "pr_number": 123,
        "pr_url": "https://github.com/org/repo/pull/123",
        "review_task_id": review["task_id"],
        "changed_files": ["rate_limiter.py", "tests/test_rate_limiter.py"],
        "verification": ["pytest tests/test_rate_limiter.py -q"],
        "merge_performed": False,
        "auto_merge_enabled": False,
    },
    created_cards=[review["task_id"]],
)
```

Use `kanban_block` only for genuine ambiguity or external input you need before you can produce the handoff. For reviewer-discovered required changes, the reviewer blocks with findings and a new implementation task is linked from that reviewer task.

**Research task:**
```python
kanban_complete(
    summary="3 competing libraries reviewed; vLLM wins on throughput, SGLang on latency, Tensorrt-LLM on memory efficiency",
    metadata={
        "sources_read": 12,
        "recommendation": "vLLM",
        "benchmarks": {"vllm": 1.0, "sglang": 0.87, "trtllm": 0.72},
    },
)
```

**Review task:**
```python
kanban_complete(
    summary="reviewed PR #123; 2 blocking issues found (SQL injection in /search, missing CSRF on /settings)",
    metadata={
        "pr_number": 123,
        "findings": [
            {"severity": "critical", "file": "api/search.py", "line": 42, "issue": "raw SQL concat"},
            {"severity": "high", "file": "api/settings.py", "issue": "missing CSRF middleware"},
        ],
        "approved": False,
    },
)
```

Shape `metadata` so downstream parsers (reviewers, aggregators, schedulers) can use it without re-reading your prose.

## Claiming cards you actually created

If your run produced new kanban tasks (via `kanban_create`), pass the ids in `created_cards` on `kanban_complete`. The kernel verifies each id exists and was created by your profile; any phantom id blocks the completion with an error listing what went wrong, and the rejected attempt is permanently recorded on the task's event log. **Only list ids you captured from a successful `kanban_create` return value — never invent ids from prose, never paste ids from earlier runs, never claim cards another worker created.**

```python
# GOOD — capture return values, then claim them.
c1 = kanban_create(title="remediate SQL injection", assignee="security-worker")
c2 = kanban_create(title="fix CSRF middleware", assignee="web-worker")

kanban_complete(
    summary="Review done; spawned remediations for both findings.",
    metadata={"pr_number": 123, "approved": False},
    created_cards=[c1["task_id"], c2["task_id"]],
)
```

```python
# BAD — claiming ids you don't have captured return values for.
kanban_complete(
    summary="Created remediation cards t_a1b2c3d4, t_deadbeef",  # hallucinated
    created_cards=["t_a1b2c3d4", "t_deadbeef"],                   # → gate rejects
)
```

If a `kanban_create` call fails (exception, tool_error), the card was NOT created — do not include a phantom id for it. Retry the create, or omit the id and mention the failure in your summary. The prose-scan pass also catches `t_<hex>` references in your free-form summary that don't resolve; these don't block the completion but show up as advisory warnings on the task in the dashboard.

## Block reasons that get answered fast

Bad: `"stuck"` — the human has no context.

Good: one sentence naming the specific decision you need. Leave longer context as a comment instead.

```python
kanban_comment(
    task_id=os.environ["HERMES_KANBAN_TASK"],
    body="Full context: I have user IPs from Cloudflare headers but some users are behind NATs with thousands of peers. Keying on IP alone causes false positives.",
)
kanban_block(reason="Rate limit key choice: IP (simple, NAT-unsafe) or user_id (requires auth, skips anonymous endpoints)?")
```

The block message is what appears in the dashboard / gateway notifier. The comment is the deeper context a human reads when they open the task.

## Heartbeats worth sending

Good heartbeats name progress: `"epoch 12/50, loss 0.31"`, `"scanned 1.2M/2.4M rows"`, `"uploaded 47/120 videos"`.

Bad heartbeats: `"still working"`, empty notes, sub-second intervals. Every few minutes max; skip entirely for tasks under ~2 minutes.

## Retry scenarios

If you open the task and `kanban_show` returns `runs: [...]` with one or more closed runs, you're a retry. The prior runs' `outcome` / `summary` / `error` tell you what didn't work. Don't repeat that path. Typical retry diagnostics:

- `outcome: "timed_out"` — the previous attempt hit `max_runtime_seconds`. You may need to chunk the work or shorten it.
- `outcome: "crashed"` — OOM or segfault. Reduce memory footprint.
- `outcome: "spawn_failed"` + `error: "..."` — usually a profile config issue (missing credential, bad PATH). Ask the human via `kanban_block` instead of retrying blindly.
- `outcome: "reclaimed"` + `summary: "task archived..."` — operator archived the task out from under the previous run; you probably shouldn't be running at all, check status carefully.
- `outcome: "blocked"` — a previous attempt blocked; the unblock comment should be in the thread by now.

## Notification routing

You can configure the gateway to receive cross-profile Kanban task notifications by adding `notification_sources` to `~/.hermes/config.yaml`.
- `notification_sources: ['*']` accepts subscriptions from all profiles.
- `notification_sources: ['default', 'zilor-ppt']` or `"default,zilor-ppt"` restricts subscriptions to specified profiles.
- Omitting the key keeps the default behavior (profile isolation).

## Do NOT

- Call `delegate_task` as a substitute for `kanban_create`. `delegate_task` is for short reasoning subtasks inside YOUR run; `kanban_create` is for cross-agent handoffs that outlive one API loop.
- Call `clarify` to ask the human a question. You are running headless — there is no live user to answer. The call will time out (default ~120s) and the task will sit silently in `running` with no signal that it needs input. Use `kanban_comment` (context) + `kanban_block(reason=...)` (decision needed) instead — the task surfaces on the board as blocked, the operator sees it, unblocks with their answer in a comment, and you respawn with the thread.
- Modify files outside `$HERMES_KANBAN_WORKSPACE` unless the task body says to.
- Create follow-up tasks assigned to yourself — assign to the right specialist.
- Complete a task you didn't actually finish. Block it instead.

## Pitfalls

**Task state can change between dispatch and your startup.** Between when the dispatcher claimed and when your process actually booted, the task may have been blocked, reassigned, or archived. Always `kanban_show` first. If it reports `blocked` or `archived`, stop — you shouldn't be running.

**Workspace may have stale artifacts.** Especially `dir:` and `worktree` workspaces can have files from previous runs. Read the comment thread — it usually explains why you're running again and what state the workspace is in.

**Don't rely on the CLI when the guidance is available.** The `kanban_*` tools work across all terminal backends (Docker, Modal, SSH). `hermes kanban <verb>` from your terminal tool will fail in containerized backends because the CLI isn't installed there. When in doubt, use the tool.

## CLI fallback (for scripting)

Every tool has a CLI equivalent for human operators and scripts:
- `kanban_show` ↔ `hermes kanban show <id> --json`
- `kanban_complete` ↔ `hermes kanban complete <id> --summary "..." --metadata '{...}'`
- `kanban_block` ↔ `hermes kanban block <id> "reason"`
- `kanban_create` ↔ `hermes kanban create "title" --assignee <profile> [--parent <id>]`
- etc.

Use the tools from inside an agent; the CLI exists for the human at the terminal.
