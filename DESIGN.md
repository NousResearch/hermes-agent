# Design: Post-Approve Merger Agent for Kanban

> **Update (2026-05-26 — post-design-approval):** Sahil confirmed: **no new `merging`
> status, no new dashboard column.** The task stays in `human_review` while the
> merger agent works, then transitions directly to `done` (or `blocked` on failure).
> Sections Q6, the state machine, and the file-change list below have been updated
> to reflect this. The `merging` column from the original sequence diagram has been
> replaced with the `running` approach described in §3.



**Task:** t_5a521a19  
**Phase:** Design (no code — approval gate before implementing)  
**Author:** braintrusteng (hermes worker)

---

## 1. Questions, Recommended Answers, and Reasoning

### Q1 — New `post-approve-merger` profile vs. extending `sdlc-review`?

**Recommendation: new dedicated `post-approve-merger` profile.**

Reasoning:
- `sdlc-review` already has a defined contract: it runs on `review` status tasks, checks PR AC, and transitions to `human_review` or rejects. Adding merge-on-approve logic there conflates two distinct responsibilities: "does this PR satisfy AC?" vs. "merge the approved PR".
- The merger's toolset is minimal: `kanban_*` tools + `terminal` (for `gh pr view / merge`). `sdlc-review` may have a broader toolset that merger doesn't need and that could create unexpected interactions.
- Single-responsibility makes the profile easy to reason about, debug, and replace. If the merging logic needs to change (e.g. migrate from squash to merge commits), only this profile changes.
- The dispatch pattern is already proven for the `review` column (dispatcher claims → spawns `sdlc-review`). The `merging` column can use the identical pattern with a different profile name.

**Profile name:** `post-approve-merger`  
**Skills force-loaded by dispatcher:** a new `post-approve-merger` skill (analogous to how `sdlc-review` is force-loaded for review tasks).

---

### Q2 — Auth identity for the merge?

**Recommendation: `sahilm-ai` OAuth token via `GH_TOKEN_SAHILM_AI` env var.**

Per the established pattern documented in session memory and the `github-pr-workflow` skill, autonomous workers push/merge using:
```
git -c http.https://github.com/.extraheader="Authorization: Basic $(echo -n x-access-token:$TOKEN | base64)" \
    -c credential.helper= push ...
```
where `TOKEN` comes from `GH_TOKEN_SAHILM_AI`.

For `gh pr merge` specifically: the merger skill will set `GH_TOKEN` from `GH_TOKEN_SAHILM_AI` before invoking `gh`:
```bash
GH_TOKEN="$GH_TOKEN_SAHILM_AI" gh pr merge <url> --squash --delete-branch
```

This keeps identity consistent with other workers. `sahilm-ti` is Sahil's personal interactive token and must never be used in autonomous workers (it could trigger notifications/permissions that Sahil hasn't approved for automation).

---

### Q3 — CI wait timeout default?

**Recommendation: 10 minutes, with a `ci_wait_timeout_minutes` comment override.**

10 minutes covers most repos in this ecosystem (typical PR checks run in 2–5 minutes). The merger skill reads a `ci_wait_timeout_minutes=N` annotation from the task body or a task comment to allow per-task override. If the annotation is absent, default is 10.

The skill polls every 30 seconds (20 polls in 10 minutes), which is non-spammy against the GitHub API and prompt enough to not waste wall-clock time.

---

### Q4 — No-PR `human_review` tasks?

**Recommendation: `approve_task()` checks for a PR URL. If none found → `human_review → done` (current behavior). If found → `human_review → merging`.**

Tasks without a PR are legitimate: research deliverables, design doc approvals, investigation summaries, skill edits. They should continue to behave exactly as they do today.

PR detection strategy (in priority order):
1. Scan the task's `events` for the most recent `review_requested` event whose `reason` field matches `_RESPAWN_GUARD_PR_URL_RE` (already defined at line 4751 of `kanban_db.py`).
2. Fall back to scanning all task `comments` for the same pattern.

If no URL is found after both passes → no PR → go directly to `done`.

---

### Q5 — Idempotency on merger restart?

**Recommendation: use `gh pr view --json state,merged` as the source of truth, not local state.**

If the merger worker crashes mid-execution and is re-dispatched, it re-runs the PR state check from scratch. Specifically:
- If `state == "MERGED"` → the merge already happened → transition to `done` with a note. No double-merge possible.
- If `state == "OPEN"` → pick up where we left off (check CI, mergeable, etc.).
- If `state == "CLOSED"` (not merged) → block back.

The merger never stores intermediate state in the task itself. It relies entirely on `gh pr view` output. This makes re-runs safe and crash recovery trivial.

---

### Q6 — Dashboard new `merging` column vs. `human_review`-with-spinner?

**Decision (Sahil, 2026-05-26): no new status, no new dashboard column.**

The implementation uses a direct-spawn approach instead of a dispatcher-loop approach:

- `approve_task()` finds a PR URL → calls `claim_merger_task()` which moves the task
  `human_review → running`, appends a `merge_requested` event, and returns the claimed
  `Task` object.
- The tool handler `_handle_approve` (in `kanban_tools.py`) receives the claimed task
  and immediately calls `_default_spawn()` to fire up the `post-approve-merger` worker
  — no waiting for the next dispatcher tick.
- The merger worker runs normally: reads the task, merges (or blocks), calls
  `kanban_complete` / `kanban_block` which transition `running → done/blocked`.
- The task is in `running` state while the merger works — that's indistinguishable
  from any other worker run on the dashboard. No new column needed.

Trade-off vs. the original `merging` approach: crash recovery is slightly slower
(the TTL reap loop reclaims the task and re-queues it as `ready` if the merger crashes,
but the assignee will be whatever the task's `assignee` field holds — the merger must
be invoked with the correct profile, which the direct-spawn path guarantees; a
re-dispatch of the `ready` task would use the original assignee, not `post-approve-merger`).
The idempotency rule (§Q5) mitigates this: re-spawn checks `gh pr view` first.

---

### Q7 — Migration: what about existing `human_review` tasks at deploy time?

**Recommendation: zero migration needed — behavioral backwards-compat via PR detection.**

At deploy time, existing `human_review` tasks have no `merging`-related history. When `kanban_approve` is called on them:
- If the task has a PR URL in events/comments → transitions to `merging` (new path). This is correct — the PR still needs to be merged.
- If the task has no PR URL → transitions directly to `done` (old path). Research/doc tasks are unaffected.

No SQL migration is needed. `VALID_STATUSES` is a Python set in memory; adding `"merging"` takes effect on process restart. The `tasks` table stores status as a `TEXT` column with no DB-level CHECK constraint (verified: `kanban_db.py` enforces validity in Python, not SQL).

---

## 2. Full State Machine (updated)

```
triage → todo → ready → running → review → human_review → done
                                ↓        ↓               ↓
                             blocked   blocked         blocked (from running, via merger)
                                ↑        ↑
                             (rejected)  (rejected)
```

With the merger flow, `human_review → done` is now split into two paths:

1. **No PR found** — `approve_task()` transitions `human_review → done` directly
   (current behavior, unchanged).
2. **PR found** — `approve_task()` calls `claim_merger_task()` which transitions
   `human_review → running`. The merger worker runs, then either:
   - `kanban_complete` → `running → done`
   - `kanban_block` → `running → blocked`

No new status values added to `VALID_STATUSES`. No new dispatcher column.

---

## 3. Sequence Diagram

```mermaid
sequenceDiagram
    participant Sahil
    participant kanban_approve (tool/CLI)
    participant kanban_db.approve_task()
    participant _handle_approve (kanban_tools.py)
    participant post-approve-merger worker
    participant GitHub (gh CLI)

    Note over Sahil: Task is in human_review
    Sahil->>kanban_approve (tool/CLI): kanban_approve(task_id, reason)
    kanban_approve (tool/CLI)->>kanban_db.approve_task(): approve_task(conn, task_id, reason)

    kanban_db.approve_task()->>kanban_db.approve_task(): _extract_pr_url(conn, task_id)
    alt No PR URL found
        kanban_db.approve_task()-->>kanban_approve (tool/CLI): (True, "done", None)
        kanban_approve (tool/CLI)-->>Sahil: {status: done}
    else PR URL found
        kanban_db.approve_task()->>kanban_db.approve_task(): claim_merger_task() → human_review → running
        kanban_db.approve_task()->>kanban_db.approve_task(): _append_event(merge_requested, {pr_url})
        kanban_db.approve_task()-->>kanban_approve (tool/CLI): (True, "merge_triggered", pr_url, task)
        kanban_approve (tool/CLI)-->>_handle_approve (kanban_tools.py): merge_triggered
        _handle_approve (kanban_tools.py)->>post-approve-merger worker: _default_spawn(task, workspace, board)
        _handle_approve (kanban_tools.py)-->>Sahil: {status: merge_triggered, pr_url: ...}

        post-approve-merger worker->>kanban_db.approve_task(): kanban_show() → get pr_url from merge_requested event
        post-approve-merger worker->>GitHub (gh CLI): gh pr view --json state,mergeable,statusCheckRollup,isDraft

        alt PR already merged
            post-approve-merger worker->>kanban_db.approve_task(): kanban_complete(summary="PR already merged")
        else PR closed (not merged)
            post-approve-merger worker->>kanban_db.approve_task(): kanban_block(reason="PR closed without merge")
        else PR is draft
            post-approve-merger worker->>kanban_db.approve_task(): kanban_block(reason="PR is draft")
        else CI pending
            loop poll every 30s up to CI_WAIT_TIMEOUT
                post-approve-merger worker->>GitHub (gh CLI): gh pr view --json statusCheckRollup
            end
            alt CI green within timeout
                post-approve-merger worker->>GitHub (gh CLI): GH_TOKEN=$GH_TOKEN_SAHILM_AI gh pr merge --squash --delete-branch
                post-approve-merger worker->>kanban_db.approve_task(): kanban_complete(summary="Merged PR #N")
            else timeout
                post-approve-merger worker->>kanban_db.approve_task(): kanban_block(reason="CI still pending after 10min")
            end
        else CI failed
            post-approve-merger worker->>kanban_db.approve_task(): kanban_block(reason="CI failing: <check_url>")
        else merge conflicts
            post-approve-merger worker->>GitHub (gh CLI): git fetch + git rebase + git push
            alt rebase clean
                post-approve-merger worker->>GitHub (gh CLI): gh pr merge --squash --delete-branch
                post-approve-merger worker->>kanban_db.approve_task(): kanban_complete(summary="Merged after rebase")
            else rebase fails
                post-approve-merger worker->>kanban_db.approve_task(): kanban_block(reason="merge conflict, manual rebase needed")
            end
        else open + mergeable + CI green
            post-approve-merger worker->>GitHub (gh CLI): GH_TOKEN=$GH_TOKEN_SAHILM_AI gh pr merge --squash --delete-branch
            post-approve-merger worker->>kanban_db.approve_task(): kanban_complete(summary="Merged PR #N")
        end
    end
```

---

## 4. `@design-guard` Text for Modified / New Constructs

### `kanban_db.py :: approve_task()`

```
@design-guard
# INVARIANT: approve_task() is the ONLY function that transitions human_review → merging
# or human_review → done. No code path may move a task OUT of human_review except
# approve_task() (merging/done) or reject_task() (ready).
#
# INVARIANT: The PR-detection logic (_extract_pr_url) is side-effect-free. It only reads
# events and comments; it never writes to the DB.
#
# INVARIANT: If _extract_pr_url returns a URL, approve_task() MUST transition to 'merging'
# and append a 'merge_requested' event carrying the pr_url. It must NOT go directly to done.
#
# INVARIANT: If _extract_pr_url returns None, approve_task() transitions directly to 'done'
# (backwards-compat path, identical behavior to pre-merger code).
```

### `kanban_db.py :: _extract_pr_url()` (new private function)

```
@design-guard
# INVARIANT: _extract_pr_url() scans (1) task events for review_requested.reason,
# then (2) task comments, in that order. It returns the FIRST GitHub PR URL found
# via _RESPAWN_GUARD_PR_URL_RE (already defined in this file). Returns None when no
# URL is found anywhere.
#
# INVARIANT: This function is pure/read-only — no DB writes, no network calls.
```

### `kanban_db.py :: dispatch_once()` — merging column section

```
@design-guard
# INVARIANT: The merging column dispatch section (SELECT … WHERE status='merging')
# is structurally identical to the review column dispatch section. It shares the
# same concurrency cap (max_spawn), crash-loop breaker, profile-existence guard,
# and workspace resolution logic.
#
# INVARIANT: The dispatcher force-loads the 'post-approve-merger' skill onto claimed
# merging tasks, analogous to force-loading 'sdlc-review' onto review tasks.
#
# INVARIANT: claim_merging_task() is the ONLY function that transitions merging →
# running for dispatcher-spawned merger workers.
```

### `post-approve-merger` skill (new)

```
@design-guard
# INVARIANT: The merger worker NEVER calls gh pr merge without first verifying
# gh pr view --json state,mergeable,statusCheckRollup,isDraft returns a state
# consistent with merging: state=OPEN, isDraft=false, mergeable=MERGEABLE,
# and all statusCheckRollup items are SUCCESS or SKIPPED (or the repo has no
# required checks).
#
# INVARIANT: The merger worker uses GH_TOKEN=$GH_TOKEN_SAHILM_AI for all gh
# invocations. It never uses sahilm-ti credentials.
#
# INVARIANT: The merger worker calls kanban_complete on success or kanban_block
# on irrecoverable failure. It never calls kanban_review or kanban_human_review.
#
# INVARIANT: The merger worker's first action is always gh pr view to check current
# PR state, even on re-spawn (idempotency from GitHub state, not local memory).
```

---

## 5. Files Changed (implementation plan)

| File | Change |
|---|---|
| `hermes_cli/kanban_db.py` | Add `_extract_pr_url()`; add `claim_merger_task()`; modify `approve_task()` to call `claim_merger_task()` when PR found and return a tuple `(success, outcome, pr_url, task)` |
| `tools/kanban_tools.py` | Update `_handle_approve()` to handle `merge_triggered` outcome: call `_default_spawn()` with the claimed task, return `{status: "merge_triggered", pr_url: ...}` |
| `hermes_cli/kanban.py` | Update `_cmd_approve()` to print "Sent to merger (PR: <url>)" on `merge_triggered` path |
| `gateway/run.py` | Add `"merge_requested"` to the notifier event set; add human-readable message |
| `~/.hermes/profiles/post-approve-merger/` | New profile directory with `config.yaml` |
| `skills/devops/post-approve-merger/SKILL.md` | New skill: PR merger procedure, all 6 PR state branches, auth pattern, idempotency |
| `tests/hermes_cli/test_kanban_merging.py` | New test file: `_extract_pr_url` (event path, comment path, no-URL path), `claim_merger_task`, `approve_task` routing |
| `tests/hermes_cli/test_kanban_human_review.py` | Add `test_approve_routes_to_merge_triggered_when_pr_url_present`, `test_approve_routes_to_done_when_no_pr_url` |

**NOT changed vs. original design:**
- `VALID_STATUSES` — no new status
- `dispatch_once()` — no new dispatch column
- `has_spawnable_review()` / `has_spawnable_ready()` — no new counterparts needed

---

## 6. Gateway Notifier Events (new event kinds)

Two new event kinds need notifier support in `gateway/run.py`:

| Event kind | Trigger | Notifier message |
|---|---|---|
| `merge_requested` | `approve_task()` detected a PR and moved to `merging` | `⚙️ Kanban {task_id} merging PR — {title}: {pr_url}` |
| `merged` | merger worker calls `kanban_complete` with `outcome=merged` | reuses `approved` / `completed` event — no new kind needed |

The `merged` outcome doesn't need a new event kind — `kanban_complete` already emits a `completed` event, and the existing `approved` notifier path fires on `kanban_approve → done` (the no-PR path). The distinction between "approved-no-PR" and "approved-then-merged-by-worker" is captured in the run history's `summary` field, which the notifier already includes in the message.

---

## 7. What This Does NOT Change

- `kanban_approve` on a `done` or `blocked` task: still returns an error, unchanged.
- `kanban_reject` on `human_review`: still bounces to `ready`, unchanged.
- The `review` column dispatch: unchanged (sdlc-review profile unchanged).
- The `sdlc-review` skill: unchanged.
- `kanban_review`, `kanban_complete`, `kanban_block` tools: unchanged in contract.
- Upstream `NousResearch/hermes-agent`: this PR targets `sahilm-ti/hermes-agent` only.

---

## 8. Open Decision for Sahil

**Dashboard column order:** should `merging` appear between `human_review` and `done`, or is the current two-column display (human_review | done) fine with `merging` inserted? The dashboard plugin is in `plugins/kanban/dashboard/` — I'll update it but want to confirm the desired visual placement.

Everything else above is ready to implement on approval.
