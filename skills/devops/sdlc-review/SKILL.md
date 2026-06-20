---
name: sdlc-review
description: >-
  Review skill for Kanban review-phase workers. Loaded automatically when the
  dispatcher spawns a reviewer agent for a task in ``review`` status. Covers
  verification of implementer handoffs, acceptance criteria checks, and the
  approve / request-changes / escalate lifecycle.
tags:
  - kanban
  - review
  - quality
  - verification
environments:
  - kanban
---

# Kanban Review Skill

You have been spawned as a **reviewer** for a Kanban task that the implementer
submitted for review. Your job is to independently verify the work and reach a
verdict: **approve**, **request changes**, or **escalate**.

## How you got here

1. An implementer agent finished its work and called
   ``kanban_request_review(summary=..., metadata=...)`` instead of
   ``kanban_complete``.
2. The task transitioned ``running → review``.
3. The dispatcher claimed it and spawned you (with this skill loaded).

## Orientation

1. **Call ``kanban_show()`` first.** The response includes:
   - The task title and body (the original spec / acceptance criteria).
   - The implementer's handoff ``summary`` and ``metadata`` from the
     ``review_requested`` event — what they claim to have done.
   - The comment thread (may contain design decisions, constraints).
   - Prior runs (attempt history — useful if this is a re-review).

2. **Understand what was asked vs what was done.** Read the acceptance criteria
   in the task body. Read the implementer's summary. Note any gaps between the
   two before you start verifying.

## Verification

### For code changes

1. **Review the diff.** If the implementer provided a ``diff_path`` in metadata,
   read it. Otherwise, find the changed files (``metadata.changed_files``) and
   read them in the workspace. Check:
   - Does the code do what the acceptance criteria require?
   - Are there obvious bugs, edge cases, or error paths not handled?
   - Does the code follow existing conventions in the file/project?
   - Are there unused imports, dead code, or leftover debug statements?

2. **Run the tests / linter** if available:
   - ``flutter analyze``, ``pytest``, ``ruff``, ``eslint``, etc.
   - If tests were listed as passing in metadata, spot-check by running them.
   - If no tests exist, verify the change manually by reading the logic.

3. **Check for scope creep.** Did the implementer change files outside the
   task's scope? Flag unrelated changes in your review comment.

### For non-code work

1. Verify the deliverable matches what the task body asked for.
2. Check data quality, formatting, completeness.
3. Validate any URLs, references, or external links the work depends on.

## Verdict

Choose **one** of these three outcomes:

### ✅ Approve → task complete

The work meets all acceptance criteria. Call:

```
kanban_complete(
    summary="Reviewed and approved. <1-2 sentences on what was verified>",
    metadata={"review_outcome": "approved", "reviewer_checks": [...]}
)
```

This transitions ``review → done``. The task is complete.

### ❌ Request changes → back to implementer

The work needs fixes before it can be approved. Write a detailed comment
explaining exactly what needs to change, then block the task:

```
kanban_comment("Changes requested:\n1. <specific issue>\n2. <specific issue>")
kanban_block(reason="changes-requested: <one-line summary>")
```

This transitions ``review → blocked``. A human or orchestrator unblocks and
the dispatcher respawns the implementer to fix the issues. When the implementer
re-submits, you'll be spawned again to re-review.

**Be specific** in your change requests. Don't write "the code needs work" —
write "the `_AlnavBookmarkTile` widget doesn't handle null `message.subject`,
add a fallback like the NAVADMIN tile has at line 45."

### ⚠️ Escalate → human needed

The task has a fundamental problem that can't be fixed by the implementer
alone (wrong approach, missing requirements, ambiguous spec). Block with:

```
kanban_block(reason="escalation: <what needs a human decision>")
```

## Pitfalls

- **Don't rubber-stamp.** Actually read the code / deliverable. The whole
  point of a review phase is independent verification, not a second pair of
  eyes that glances and approves.

- **Don't fix the code yourself.** If you find bugs, request changes — don't
  edit the files. Your job is verification, not implementation. The
  implementer fixes; you verify the fix.

- **Don't complete without checking acceptance criteria.** Read the task body.
  If the spec says "3 things" and the summary says "done", verify all 3.

- **Don't block for style nits.** If the code is correct and follows
  conventions, approve. Save "request changes" for things that would break or
  mislead — wrong logic, missing error handling, incomplete acceptance criteria.

## What a good review looks like

**Bad:** "Looks good, approved."

**Good:** "Reviewed the diff — `_AlnavBookmarkTile` correctly mirrors the
NAVADMIN pattern with Dismissible, proper `removeBookmark(id, 'alnav')` call,
and navigation to `AlnavDetailScreen`. Empty state text updated. Ran
`flutter analyze` — 0 issues. All 6 acceptance criteria verified."
