# SDLC review agent — tool sequence and verification checklist

You are on a card that already has an implementer run with `outcome: submitted_for_review` in run history.

## Orient

1. `kanban_show()` — AC in body, implementer `summary`/`metadata`, parent handoffs, prior runs.
2. `cd $HERMES_KANBAN_WORKSPACE` — diff/PR live here for worktree tasks.
3. Confirm `HERMES_KANBAN_REVIEW=1` (review spawn). If unset, you are not the review agent.

## Verify (before any transition)

| Check | How |
|-------|-----|
| Scope matches AC | `git diff`, file list vs `metadata.changed_files` |
| Tests cited | Re-run or spot-check; do not trust counts without evidence |
| PR exists | `metadata.pr` or `gh pr view`; diff matches workspace |
| No drive-by changes | Files outside AC called out in comment |

Load `github-code-review` when a GitHub PR is involved.

## Hand off

**Pass (human merge):**

```python
kanban_comment(body="## Review\n- AC: …\n- Tests: …\n- PR: …\n- Risk: …")
kanban_block(reason="review-required: AC met, N/N tests, PR #X ready for human merge")
```

**Fail (implementer retry):**

```python
kanban_comment(body="## Review findings\n- …")
kanban_request_changes(reason="AC gap: …; fix … in path/to/file")
```

Post the comment **before** `kanban_request_changes` so the implementer has a fix list in the thread.

## Common mistakes

| Mistake | Why it fails |
|---------|----------------|
| `kanban_complete` | Tool error — review runs cannot complete to done |
| `gh pr merge` | Out of scope unless task body explicitly orders merge |
| Re-implementing the feature | Use `kanban_request_changes`; assignee picks up from `ready` |
| Empty `kanban_block` reason | Reason is required; use `review-required:` prefix for dashboard filtering |

## After you block

Card is `blocked` until a human merges the PR and marks done or unblocks. A comment alone does not move status — humans use dashboard Unblock or Done.
