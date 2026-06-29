# Sync Sprint Status

Shared sub-step for updating `sprint-status.yaml` during quick-dev. Called from any route (plan-code-review, one-shot, future routes) with a `{target_status}` parameter.

## Preconditions

Skip this entire file (return to caller) if ANY of:
- `{story_key}` is unset
- `{sprint_status}` does not exist on disk

## Instructions

1. Load the FULL `{sprint_status}` file.
2. Find the `development_status` entry matching `{story_key}`. If not found, warn the user once (`"{story_key} not found in sprint-status; skipping sprint sync"`) and return to caller.
3. **Idempotency check.** If `development_status[{story_key}]` is already at `{target_status}` or a later state (`review` is later than `in-progress`; `done` is later than both), return to caller — no write needed. Never regress a story's status.
4. Set `development_status[{story_key}]` to `{target_status}`.
5. **Epic lift (only when `{target_status}` = `in-progress`).** Derive the parent epic key as `epic-{N}` from the leading numeric segment of `{story_key}` (e.g., `3-2-digest-delivery` → `epic-3`). If that entry exists and is `backlog`, set it to `in-progress`. Leave it alone otherwise. Skip this sub-step entirely when `{target_status}` is not `in-progress`.
6. Refresh `last_updated` to the current date.
7. Save the file, preserving ALL comments and structure including STATUS DEFINITIONS and WORKFLOW NOTES.
