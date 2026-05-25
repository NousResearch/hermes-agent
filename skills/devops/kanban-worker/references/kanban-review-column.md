# Kanban Review column — three lanes, dispatch, tools

Use when the user asks how **Review** works, or when choosing between Review, Blocked, and a separate reviewer task.

## Three different “review” mechanisms

| Lane | Status / pattern | Who acts | Typical use |
|------|------------------|----------|-------------|
| **Review column** | `status = review` | Dispatcher spawns **sdlc-review** on the **same** card | Automated SDLC/PR verification after implementer `kanban_complete` |
| **Blocked + `review-required:`** | `status = blocked`, reason prefix | **Human** after automated review approves AC | Review agent calls `kanban_block(review-required: …)`; human merges/unblocks |
| **Ready + reviewer assignee** | `status = ready`, assignee = reviewer profile | Normal worker on that profile | Orchestrator pattern: **separate review task**, not the Review column |

Do not conflate these when explaining the board or planning work.

## End-to-end SDLC flow (worktree / dir)

When `require_review_workspace_kinds` includes `worktree` and/or `dir` (default for coding boards):

```
running (implementer)
  → kanban_complete(summary, metadata)
  → review          # kernel redirect; run outcome submitted_for_review
  → running         # dispatcher + sdlc-review (HERMES_KANBAN_REVIEW=1)
  → blocked         # kanban_block(review-required: …) → human merge/sign-off
     or ready       # kanban_request_changes(reason) → implementer fixes
```

**Entering `review`:** implementer calls `kanban_complete` with structured handoff (`metadata.pr`, `changed_files`, test counts). Kernel sets `status = review` and ends the run as `submitted_for_review`. No separate `kanban_submit_for_review` tool.

**Leaving `review` (review agent):**

| Outcome | Tool | Next status |
|---------|------|-------------|
| AC met, human should merge | `kanban_block(reason="review-required: …")` | `blocked` |
| Fixes needed | `kanban_comment` + `kanban_request_changes(reason=…)` | `ready` (same assignee) |

Review agents **cannot** `kanban_complete` to `done` — the tool rejects when `HERMES_KANBAN_REVIEW=1`.

## Dispatcher (review → running)

On each dispatch tick, after the Ready queue:

1. Select `status = 'review'`, unclaimed, by priority.
2. Require assignee that maps to a real Hermes profile.
3. `claim_review_task`: `review` → `running` (separate from `ready` → `running`).
4. New run id — review history separate from implementation run.
5. Parent dependencies **not** re-checked.
6. Force-load **`sdlc-review`** + `kanban-worker` lifecycle; set `HERMES_KANBAN_REVIEW=1`.
7. Counts toward the same **max concurrent workers** cap as Ready.

## Config knobs

| Key | Effect |
|-----|--------|
| `kanban.require_review_workspace_kinds` | Which workspace kinds redirect `kanban_complete` → `review` (default often `worktree`, `dir`) |
| `kanban.allow_complete_without_review` | When true, worktree/dir complete goes straight to `done` |
| `metadata.review_waived` | Per-task bypass of redirect |

`scratch` workspaces complete to `done` normally.

## Pitfalls for agents

- Saying implementer `kanban_complete` on a worktree lands in **Blocked** — that was the old redirect; current kernel lands in **Review**.
- Using `kanban_block(review-required:…)` as the **implementer** handoff — implementers use `kanban_complete`; only the **review agent** blocks for human sign-off.
- Creating a Ready card assigned to `reviewer` and calling it “the Review column” — that is a separate task, still `ready`.
- Calling `kanban_request_changes` without a prior `kanban_comment` fix list — post findings first; the tool message expects durable context in the thread.
- Assuming dashboard drag-drop into Review works everywhere — outbound dispatch is implemented; some PATCH paths may still lag. Prefer `kanban_complete` from the implementer.

## Board UI

`BOARD_COLUMNS` includes `scheduled` and `review`. Bundled dashboard JS may have stale column-order comments — labels can be thin until frontend catches up.

See also: `sdlc-review` skill and `references/sdlc-review-flow.md` under that skill directory.
