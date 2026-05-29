---
name: kanban-worker
description: Pitfalls, examples, and edge cases for Hermes Kanban workers. The lifecycle itself is auto-injected into every worker's system prompt as KANBAN_GUIDANCE (from agent/prompt_builder.py); this skill is what you load when you want deeper detail on specific scenarios.
version: 2.3.0
platforms: [linux, macos, windows]
environments: [kanban]
metadata:
  hermes:
    tags: [kanban, multi-agent, collaboration, workflow, pitfalls]
    related_skills: [kanban-orchestrator]
---

# Kanban Worker — Pitfalls and Examples

> You're seeing this skill because the Hermes Kanban dispatcher spawned you as a worker with `--skills kanban-worker` — it's loaded automatically for every dispatched worker. The **lifecycle** (6 steps: orient → work → heartbeat → block/complete) also lives in the `KANBAN_GUIDANCE` block that's auto-injected into your system prompt. This skill is the deeper detail: good handoff shapes, retry diagnostics, edge cases.

## References

| File | Description |
|---|---|
| `references/retry-diagnostics.md` | Outcome-by-outcome retry guide (timed_out, crashed, iteration-budget-exhausted, spawn_failed, reclaimed). |
| `references/worker-pitfalls.md` | Detailed pitfalls: task state changes, stale workspace artifacts, CLI unavailability in containers. |
| `references/shared-venv-poisoning-recovery.md` | Diagnosis and recovery for `pip install -e .` in a worktree poisoning the shared venv. |
| `references/git-commit-discipline.md` | Why and how to commit all git-tracked writes before completing a task. |
| `references/gh-cli-auth.md` | Handling `gh pr create` 401 errors when the keychain isn't visible to the worker subprocess. |
| `references/force-push-approval.md` | How to handle the interactive force-push approval gate from a dispatched worker. |

## Workspace discipline for git work

When your task touches `hermes-agent` source (or any other git repo this profile owns), the dispatcher provisions a per-task **git worktree** at `~/.hermes/worktrees/<task-id>` on a task-scoped branch `kanban/<task-id>`. Work entirely inside that path (`$HERMES_KANBAN_WORKSPACE` resolves there). The shared "live checkout" at `~/.hermes/hermes-agent` is read-only for you — do not `cd` into it, do not `git checkout` / `git reset` / `git rebase` there, do not push from there.

If you crash and respawn, you'll get the same worktree path with your committed work intact. Uncommitted state is lost — `git commit` early and often inside your worktree.

If you need a base other than `myfork/main`, set `task.branch_name` via `kanban_create`'s `branch_name` field when fanning out a child card. Don't change branches inside your worktree to chase someone else's work — let the parent task carry the right base.

Full reference (provisioning, respawn semantics, cleanup, the orchestrator escape hatch): `~/.hermes/skills/devops/kanban-orchestrator/references/worktree-per-task.md`.

## CRITICAL: Never git-branch or git-checkout outside your workspace

The dispatcher pins `TERMINAL_CWD` to your workspace (e.g. `~/.hermes/worktrees/t_abc123/`)
before spawning you. All `terminal()` calls default to that directory. **Do NOT manually
`cd ~/.hermes` or `cd ~/` and then run `git checkout -b` or `git commit` there.**

Why this matters: `~/.hermes` is simultaneously a git repo (hermes-config) **and** the
live runtime config root. If you `cd ~/.hermes && git checkout -b kanban/<task>`, you
branch the live config tree, not your task's worktree. This has caused the real symptoms:

- "Live tree drifts from main" — the checkout stays on worker branches after they finish.
- "Auto-pull never fires" — auto-pull requires clean+on-main; worker branches block it.
- "Skills disappear from disk when I reset" — they were committed on a worker branch, not main.

The structural fix: your `TERMINAL_CWD` is already `$HERMES_KANBAN_WORKSPACE`. Every
`terminal("git ...")` call without an explicit `cd` runs in your worktree. Use that.

```bash
# CORRECT — defaults to $HERMES_KANBAN_WORKSPACE, no cd needed
terminal("git add -A && git commit -m 'feat: implement rate limiter'")
terminal("git push origin HEAD:kanban/t_abc123")

# WRONG — escaping your workspace, polluting the live config tree
terminal("cd ~/.hermes && git checkout -b kanban/t_abc123")
terminal("cd ~ && git commit -am 'feat: ...'")
```

If you genuinely need to commit work to hermes-config (`~/.hermes/skills/`, `~/.hermes/profiles/`),
see `kanban-orchestrator/references/live-checkout-branch-hygiene.md` for the correct procedure
(branch first on the live checkout, commit only your files, PR, return to main). But most
workers do NOT need to commit there — skill edits via `skill_manage` write to the profile path
the agent reads from, and the skill files live under your profile's `HERMES_HOME`, not under
the git-tracked `~/.hermes/skills/` in the live checkout.

## Live-shared repo pickup — return to `main` before touching anything

When your worktree's `.git` is shared with a live checkout (`~/.hermes` itself, `~/.hermes/hermes-agent`), your previous PR may have merged on GitHub while the live checkout still points at your now-merged feature branch. The next worker (you) inherits this stale state and shares the same `.git/refs`. Before any git work in your worktree:

1. Read the live checkout's current state once with `git -C <live-checkout> branch --show-current`. If it's anything other than `main` AND the named branch is merged (`gh pr list --state merged --head <branch>` returns a PR), don't try to clean it up from inside the worktree — leave a `kanban_comment` on your card flagging the stale live checkout for the orchestrator and proceed with your own work in the worktree (your worktree's branch is independent).
2. Never run `git checkout main` or `git pull` in the live checkout from a worker — you don't own that working tree and can stomp on another worker's mid-rebase state. If the live checkout needs cleanup, that's the orchestrator's call via the recovery recipe in `~/.hermes/skills/devops/kanban-orchestrator/references/live-checkout-branch-hygiene.md`.
3. Your own merged feature branch can be deleted locally (`git branch -D <pr-branch>`) AFTER you confirm `gh pr view <pr-branch> --json state` returns `MERGED` and your current work is on a different branch. Stale local branches are a real source of drift.

## If your task writes files into a git-tracked path, commit them before completing

At the moment of terminal action (`kanban_review` / `kanban_complete` / `kanban_block`), `git status --porcelain` should be clean inside any git-tracked path you wrote into. The failure mode is sneaky because runtime checks pass even when git is dirty; the auto-reviewer may green-light a card that leaves untracked files on disk. Always include `changed_files` in your completion metadata.

See `references/git-commit-discipline.md` for the full rule and three-option decision tree.

## Workspace handling

Your workspace kind determines how you should behave inside `$HERMES_KANBAN_WORKSPACE`:

| Kind | What it is | How to work |
|---|---|---|
| `scratch` | Fresh tmp dir, yours alone | Read/write freely; it gets GC'd when the task is archived. |
| `dir:<path>` | Shared persistent directory | Other runs will read what you write. Treat it like long-lived state. Path is guaranteed absolute (the kernel rejects relative paths). |
| `worktree` | Git worktree at the resolved path (typically `~/.hermes/worktrees/<task-id>`) | The dispatcher provisions the worktree and branch (`kanban/<task-id>`) for you; just `cd "$HERMES_KANBAN_WORKSPACE"` and start working. Commit early and often — uncommitted state is lost on crash, committed state survives respawn. Push your branch to `myfork` and open the PR from there; **never** push or rebase from the shared live checkout. |

## `pip install -e .` inside a worktree — do NOT do this

Running `pip install -e .` from a worktree workspace poisons the shared venv by hardcoding the worktree path into the editable-install finder. When the worktree is cleaned up, every dispatcher spawn crashes with `ModuleNotFoundError`. If you must run an editable install, use a venv local to the worktree or install from the canonical checkout.

If you suspect this has happened, see `references/shared-venv-poisoning-recovery.md` for the full diagnostic recipe and one-line `sed` fix.

## Tenant isolation

If `$HERMES_TENANT` is set, the task belongs to a tenant namespace. When reading or writing persistent memory, prefix memory entries with the tenant so context doesn't leak across tenants:

- Good: `business-a: Acme is our biggest customer`
- Bad (leaks): `Acme is our biggest customer`

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

**Coding / skill / investigation / docs task → use `kanban_review` (the universal default):**

`kanban_review` is now the **default worker terminator** for almost every card. The `sdlc-review` auto-reviewer picks the right check matrix based on the card shape (PR / skill-doc / investigation / code-no-pr / decomposition — see the `sdlc-review` skill's `references/work-type-detection.md`). On a clean pass the card lands in `human_review` for Sahil to approve; on a fail the card bounces back to `ready` for another pass.

`kanban_complete` is reserved for these explicit exceptions:

- **Orchestrator decomposition / routing.** A planner card whose deliverable is the spawned child tasks. The orchestrator has no diff and no writeup to review; the children are the deliverable. Use `kanban_complete(summary=..., created_cards=[c1, c2, ...])`.
- **Cron heartbeat / status-report tasks.** Scheduled jobs that emit a status line and end. There's no review surface — `kanban_complete` carries the heartbeat content.
- **Body marker `skip-review: <reason>`.** A one-line literal `skip-review: <reason>` in the card body opts the card out of review. Use sparingly — Sahil's explicit "no need to review" cases (e.g. "land this typo fix straight away").

Everything else — code PRs, skill / config / docs changes, investigations, in-place fixes — terminates via `kanban_review`:

```python
import json

kanban_comment(
    body="PR <url>\nSummary: <2-3 sentences>\nTests: 14/14 pass\n" + json.dumps({
        "changed_files": ["rate_limiter.py", "tests/test_rate_limiter.py"],
        "tests_run": 14,
        "tests_passed": 14,
        "decisions": ["user_id primary, IP fallback for unauthenticated requests"],
    }, indent=2),
)
kanban_review(reason="PR <url>, AC: rate limiter with user_id+IP fallback shipped, 14/14 tests pass")
```

`kanban_block(reason="review-required: ...")` is **forbidden** as a handoff for code-review work — it's the old pre-two-stage-flow pattern and now causes blocked notifications to fire on cards that aren't actually blocked. `kanban_review` is the only correct verb for "my work is ready, please review."

Use `kanban_complete` only for the three exceptions listed above (orchestrator decomposition, cron heartbeat, body marker `skip-review:`). When in doubt, `kanban_review` — the auto-reviewer handles the dispatch.

**Research / investigation task (the deliverable IS the writeup) → still `kanban_review`:**
```python
kanban_comment(body="""Conclusion: vLLM is the best fit on throughput.
Evidence:
- vllm throughput 1.0 baseline (benchmark: <url>)
- sglang 0.87, trtllm 0.72 (same benchmark)
Alternatives ruled out: SGLang loses on cold-start; TRT-LLM ops cost is 2x.
Next step: queue migration card on platform-eng.""")
kanban_review(reason="research done; vLLM recommended, 3 alternatives evaluated, evidence in comment")
```

The `investigation` mode in `sdlc-review` checks evidence-quoted, alternatives-ruled-out, actionable-next-step — exactly the shape the comment above takes.

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

## Card body is the contract — honor scope negations

The card body is what the orchestrator (or human) wrote when queueing the work. Treat it as the authoritative scope, not a suggestion. In particular, when a card body says **"do not ship,"** **"surface for review first,"** **"block before applying,"** or any similar negation, that negation **overrides your instinct to finish the job.** Do not interpret a successful evaluation step as license to proceed to the ship step.

The pattern that fails: card asks for `diagnose → propose → evaluate`; worker diagnoses, proposes, evaluates, then (because evaluation passed and the next logical step is to ship) calls the ship tool too. Even when the card body literally said *"Do NOT ship — surface it for Sahil's review first."*

The correct shape when the card carries a negation: do exactly the phases the card body lists, then `kanban_block(reason="needs-approval-for-next-phase: ...")` and put the would-be-next-step plan in a `kanban_comment` so the human can approve it explicitly. (Note: the `review-required:` prefix used by the old pre-two-stage-flow pattern is **reserved/forbidden** — see the `kanban_review` section above. Use a semantic prefix that names what the block actually is.) If the next phase is intrinsically tied to the current work (e.g. an `apply_candidate` call that uses an in-memory candidate ID), say so in the comment — "applying requires re-running propose+evaluate after approval" — and let the human decide. The structural fix is the orchestrator's job: split multi-phase work into separate cards with `parents=[...]` gates. But until that lands, the card body's negation is your contract.

When the card body is ambiguous about scope (no explicit "do not X"), default to the **smallest reversible deliverable**: produce the artifact, write the comment, block for review. Shipping changes to live systems that don't have a `kanban_block` gate in the workflow is the failure mode.

## Block reasons are clipped at 160 chars in Telegram notifications

The gateway's kanban notifier truncates the `reason` field at **160 characters** before publishing it as a Telegram (or other adapter) message. Long reasons get clipped mid-sentence. The full reason is still in the database (`hermes kanban show` reads it intact), but the notification surface is lossy.

**Discipline: short reason + long comment.** Treat `kanban_block(reason=...)` (and `kanban_review(reason=...)`) as a tweet-length headline; put the substance in a `kanban_comment` first. Aim for ≤140 chars on the reason.

```python
# Bad (clipped at 160, the actionable part is gone):
kanban_block(reason="needs-approval-for-next-phase: proposed `braintrust-agent-optimization` rewrite + new `references/ship-to-repo.md`. Diff and full files in workspace; comment has approach, verification, and on-approve steps...")

# Good (full context in the comment, short headline in the reason):
kanban_comment(body="""handoff:
- Proposed rewrite of `braintrust-agent-optimization` SKILL.md (326 lines, +59%) + new `references/ship-to-repo.md`.
- Full diff at `~/.hermes/kanban/workspaces/<task_id>/SKILL.md.diff`.
- On approve: patch SKILL.md + write reference + commit/push.""")
kanban_block(reason="needs-approval-for-next-phase: braintrust-agent-optimization rewrite. See comment for diff + on-approve plan.")
```

The same applies to `completed` events (summary truncated at 200 chars) and `gave_up` events (error truncated at 200 chars). Keep headlines short; rely on the comment thread for substance.

## `gh` CLI auth from worker subprocesses

`gh pr create` may fail with HTTP 401 in a worker even when it works for the user at the terminal — keychain-backed token stores aren't visible to worker subprocess trees. Do all the work (clone, commit, push), then `kanban_block` with the branch compare URL and ready-to-paste PR body so the human can open the PR manually. See `references/gh-cli-auth.md` for the full pattern.

## Force-push interactive approval gate

Hermes intercepts every `git push --force` / `--force-with-lease` for live human approval. Pre-approval in the card body does **not** bypass the gate. Do all mechanical work, capture the lease SHA, then `kanban_block` with the ready-to-paste `--force-with-lease=<branch>:<sha>` command. See `references/force-push-approval.md` for the full recipe.

## Heartbeats worth sending

Good heartbeats name progress: `"epoch 12/50, loss 0.31"`, `"scanned 1.2M/2.4M rows"`, `"uploaded 47/120 videos"`.

Bad heartbeats: `"still working"`, empty notes, sub-second intervals. Every few minutes max; skip entirely for tasks under ~2 minutes.

## Retry scenarios

If prior runs exist (check `kanban_show` for `runs: [...]`), you are a retry — load `references/retry-diagnostics.md` for outcome-by-outcome guidance.

## Notification routing

You can configure the gateway to receive cross-profile Kanban task notifications by adding `notification_sources` to `~/.hermes/config.yaml`.
- `notification_sources: ['*']` accepts subscriptions from all profiles.
- `notification_sources: ['default', 'zilor-ppt']` or `"default,zilor-ppt"` restricts subscriptions to specified profiles.
- Omitting the key keeps the default behavior (profile isolation).

## Do NOT

- Call `delegate_task` as a substitute for `kanban_create`. `delegate_task` is for short reasoning subtasks inside YOUR run; `kanban_create` is for cross-agent handoffs that outlive one API loop.
- Modify files outside `$HERMES_KANBAN_WORKSPACE` unless the task body says to.
- Create follow-up tasks assigned to yourself — assign to the right specialist.
- Complete a task you didn't actually finish. Block it instead.
- **Run `git checkout`, `git branch`, or `git commit` in `~/.hermes` or `~/` from a worker.** Your `TERMINAL_CWD` is already your workspace. See "CRITICAL: Never git-branch or git-checkout outside your workspace" above.
- **On BrainTrust repos**, run forbidden commands. The full allow/deny tiers are in `braintrust-eng-process` → "Worker command policy". TL;DR: tests, lints, type-checks, edits, `git`, and `gh pr create` are fine; `gh pr merge`, force-pushes, prod deploys, and prod DDB writes are forbidden under all conditions; running the app live or any raw `aws` CLI call requires `kanban_block` first.

## Pitfalls

- **Task state can change** between dispatch and startup — always `kanban_show` first; stop if `blocked` or `archived`.
- **Workspace may have stale artifacts** — especially `dir:` and `worktree`; read the comment thread for context.
- **Don't rely on the CLI in containers** — `hermes kanban <verb>` fails in Docker/Modal/SSH; use the `kanban_*` tools.

See `references/worker-pitfalls.md` for full details on each pitfall.

## CLI fallback (for scripting)

Every tool has a CLI equivalent for human operators and scripts:
- `kanban_show` ↔ `hermes kanban show <id> --json`
- `kanban_complete` ↔ `hermes kanban complete <id> --summary "..." --metadata '{...}'`
- `kanban_block` ↔ `hermes kanban block <id> "reason"`
- `kanban_create` ↔ `hermes kanban create "title" --assignee <profile> [--parent <id>]`
- etc.

Use the tools from inside an agent; the CLI exists for the human at the terminal.
