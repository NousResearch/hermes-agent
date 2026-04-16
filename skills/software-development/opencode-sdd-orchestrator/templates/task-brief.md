# OpenCode Task Brief

## Context
- Task ID: `<task-id>`
- Title: `<short-title>`
- Repository: `<owner/repo or local path>`
- Worktree: `<absolute-worktree-path>`
- Branch: `<type>/<task-slug>`
- Base branch: `<main|develop|...>`

> Replace **all** placeholders (`<...>`) before using this brief with the helper.

## Strict execution contract

1. Use OpenCode agent: `sdd-orchestrator`.
2. If the requested agent cannot be honored, stop and report failure.
3. Do not silently fall back to `build` or any default agent.
4. OpenCode must create the required git commit(s) in the isolated worktree before returning success.
5. The helper will refuse to push or create a PR if no commits exist relative to the base branch.

## Problem statement
`<what needs to be built/fixed and why>`

## In scope
- `<explicit item 1>`
- `<explicit item 2>`

## Out of scope
- `<explicitly excluded change 1>`
- `<explicitly excluded change 2>`

## Required implementation tasks
1. `<task step 1>`
2. `<task step 2>`
3. `<task step 3>`

## Acceptance criteria (must all pass)
- [ ] `<criterion 1>`
- [ ] `<criterion 2>`
- [ ] `<criterion 3>`

## Verification commands
Run exactly:

```bash
<command-1>
<command-2>
<command-3>
```

## File constraints
- Allowed paths:
  - `<path/prefix/or/file>`
  - `<path/prefix/or/file>`
- Forbidden paths:
  - `scripts/whatsapp-bridge/package-lock.json`
  - `<other-restricted-paths>`

## Required final output format

Return a structured report:

1. **Summary** — what changed and why
2. **Files changed** — list with one-line rationale per file
3. **Verification results** — command + exit status + key output
4. **Acceptance criteria mapping** — each criterion marked pass/fail with evidence
5. **Risks / follow-ups** — remaining concerns and next actions

The response must reflect committed work on the assigned branch, not only uncommitted workspace changes.
