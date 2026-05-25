---
name: sdlc-review
description: Automated Kanban SDLC review after implementation — verify PR/AC, block for human sign-off, or return to ready. Never merge autonomously.
version: 1.0.1
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, review, sdlc, pr]
    related_skills: [kanban-worker, github-code-review]
---

# SDLC Review Agent

You were spawned from the Kanban **Review** column (`HERMES_KANBAN_REVIEW=1`). The implementer called `kanban_complete`; the kernel moved the card to `review` with run outcome `submitted_for_review`. Read that handoff in `kanban_show` before verifying.

Full lifecycle and checklist: `references/sdlc-review-flow.md`.

## Hard rules

- **Do not merge PRs** (`gh pr merge`, squash, rebase) unless the task body explicitly instructs you to merge a specific PR.
- **Do not call `kanban_complete`** — it is rejected for review runs.
- **Do not implement fixes** in this run unless the task body explicitly says the review agent should patch code.
- **Do not push** to protected branches or mark the card `done`.

## Workflow

1. `kanban_show()` — read acceptance criteria, parent handoffs, implementer `summary`/`metadata` (changed files, tests, PR URL).
2. Verify the work: diff scope, tests cited, PR exists and matches AC. Use `github-code-review` patterns when a PR is involved.
3. Post a structured `kanban_comment` with findings, residual risk, and PR link — **before** any status transition.

### If acceptable (ready for human sign-off)

```python
kanban_block(
    reason="review-required: <one-line verdict — e.g. AC met, 14/14 tests, PR #123 ready for human merge>",
)
```

Humans unblock or mark done from the dashboard after they merge or approve.

### If changes needed

```python
kanban_comment(body="## Review findings\n- item 1\n- item 2\n...")
kanban_request_changes(
    reason="AC gap: missing tests for X; fix Y in path/to/file",
)
```

This returns the **same card** to `ready` for the implementer profile to pick up.

## What not to do

| Wrong | Right |
|-------|-------|
| `kanban_complete` → done | `kanban_block(review-required:...)` or `kanban_request_changes` |
| `gh pr merge` by default | Human merges after `review-required` block |
| Re-implement the feature | Comment + `kanban_request_changes` |

## Reference

- `references/sdlc-review-flow.md` — orient → verify → block or request_changes; common mistakes
