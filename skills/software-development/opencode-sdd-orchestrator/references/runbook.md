# Runbook: Hermes-Orchestrated OpenCode SDD Delivery

This runbook is operational. Execute steps in order. If a hard gate fails, stop immediately.

## 0) Inputs required from user

- `task_id` or short task title
- implementation scope
- base branch (default: `main`)
- requested OpenCode agent (must be `sdd-orchestrator`)

## 1) Prerequisites (hard gates, before branch/worktree actions)

Run in the **main repository checkout root** only.
Do **not** invoke this helper from an already-linked git worktree root, because the helper creates and manages its own isolated worktree.

```bash
command -v opencode
opencode --version
git rev-parse --is-inside-work-tree
git status --porcelain
```

Git safety must be clean before continuing:
- `git status --porcelain` must be empty
- No in-progress operations in `.git/` such as `MERGE_HEAD`, `CHERRY_PICK_HEAD`, `REVERT_HEAD`, `REBASE_HEAD`, `rebase-apply/`, `rebase-merge/`
- Repository root must have a `.git/` directory (not a `.git` file from a linked worktree)

If PR is expected:

```bash
command -v gh
gh auth status
```

If any command fails, stop and report exactly what is missing.

When PR creation is requested, both `--pr-title` and `--pr-body-file` are required.
`--pr-body-file` must point to a populated file (not empty/whitespace-only, no `<placeholder>` tokens).

If repository is dirty or an in-progress git operation exists, stop and report the blocking state.

## 2) Agent availability and alias hard guard (fail closed)

Normalize and validate requested agent before running OpenCode:

- Allowed in this workflow: `sdd-orchestrator` only.
- Reject `sdd-orquesador` explicitly.

Use this guard logic:

```text
if requested_agent != "sdd-orchestrator":
  fail("Unsupported agent for this workflow. Use 'sdd-orchestrator'.")
```

Why: `sdd-orquesador` does not exist on this machine and may silently fall back to `build`.

## 3) Create isolated branch + worktree

```bash
git fetch origin "${BASE_BRANCH}"
git worktree add ".worktrees/${BRANCH_TYPE}-${TASK_SLUG}" -b "${BRANCH_TYPE}/${TASK_SLUG}" "origin/${BASE_BRANCH}"
```

Work only inside `<repo>/.worktrees/${BRANCH_TYPE}-${TASK_SLUG}` from this point.

## 4) Generate structured brief for OpenCode

Prepare a **filled** brief file using `templates/task-brief.md` as a starter.

Path resolution note: relative `--task-file` / `--pr-body-file` paths are resolved from the caller's current working directory.

Important behavior alignment:
- The helper validates the brief is populated.
- The helper does **not** auto-generate or auto-fill brief content.
- Any `<placeholder>` values must be replaced before running OpenCode.

Brief must include:
- problem statement and boundaries
- explicit in-scope items
- explicit out-of-scope items
- explicit acceptance criteria
- validation commands
- file constraints / forbidden changes
- required final report format

Do not start OpenCode until brief is complete.

## 5) Execute OpenCode with pinned SDD orchestrator agent

From worktree directory:

```bash
opencode run "Implement the attached brief exactly. Return changed files + validations and acceptance criteria evidence." \
  --agent sdd-orchestrator \
  --file "<filled-brief-file>.md" \
  --format json
```

OpenCode must create the git commit(s) in the isolated worktree before handing control back.
This helper will refuse to push or create a PR if the branch has zero commits relative to the base branch.

## 6) Detect fallback and fail closed

After execution, verify the run metadata/output indicates `sdd-orchestrator`.

Fail immediately if any evidence of:
- default `build` agent
- unknown agent warning
- missing agent metadata when metadata is expected

If fallback suspected, stop and report:
"Aborted: requested agent unavailable or OpenCode fallback detected."

## 7) Verify outcomes

- Confirm changed files match brief scope.
- Run **all** brief-specified verification commands (explicitly listed in the brief).
- Capture command outputs and exit codes as evidence.
- Confirm OpenCode report includes files changed, verification results with exit evidence, and acceptance criteria mapping/evidence.
- Do not mark workflow successful, push, or create PR unless verification passed.

If verification fails, loop back with a revised brief or corrective OpenCode prompt (same agent requirement).

## 8) Create PR

From worktree branch:

```bash
cp "skills/software-development/opencode-sdd-orchestrator/templates/pr-body.md" \
  ".tmp/pr-body-${TASK_SLUG}.md"
# edit .tmp/pr-body-${TASK_SLUG}.md until every placeholder is replaced

git push -u origin "${BRANCH_TYPE}/${TASK_SLUG}"
gh pr create --base "${BASE_BRANCH}" --head "${BRANCH_TYPE}/${TASK_SLUG}" \
  --title "${BRANCH_TYPE}: ${TASK_TITLE}" \
  --body-file ".tmp/pr-body-${TASK_SLUG}.md"
```

Do not point `--body-file` directly at the raw template.
Do not submit a PR using an empty or template-placeholder PR body file.
Do not rely on silent existing-PR reuse. If `gh pr create` reports a duplicate PR, stop and update or close that PR explicitly.

## 9) Return response to user

Must include:
- branch and worktree path used
- verification summary (what passed / failed)
- explicit verification commands executed (or dry-run plan)
- PR URL

Never return success without PR URL when PR creation was requested.

## Pitfalls checklist

- [ ] Used wrong agent alias (`sdd-orquesador`) instead of `sdd-orchestrator`
- [ ] Did not validate agent before run
- [ ] Allowed fallback to `build`
- [ ] Ran in non-isolated checkout
- [ ] Opened PR without acceptance-criteria verification
