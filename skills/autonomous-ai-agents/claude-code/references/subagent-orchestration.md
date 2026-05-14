# Claude Code Subagent Orchestration

Use this reference when Hermes launches Claude Code to coordinate subagents, agent teams, or a long autonomous implementation where Claude may spawn specialized agents internally.

## Parent/child responsibility split

- **Claude Code child**: implement the requested change, use Claude-native subagents when useful, run targeted validation, and report exactly what changed.
- **Hermes parent**: supervise lifecycle, inspect the resulting git tree, perform an independent review, commit coherent changes, open or update the PR, and notify the user.

Do not let the child agent's final message be the only proof. The parent must verify the filesystem and git state directly.

## Launch prompt requirements

When starting Claude Code for subagent-style implementation, put these requirements in the prompt file or inline prompt:

```markdown
You may use Claude Code subagents for implementation/review if useful.

Completion contract:
1. Keep changes limited to the requested scope.
2. Run the repo's documented validation commands, or explain why they cannot run.
3. Leave the worktree in a reviewable state.
4. Commit only if explicitly asked; otherwise leave changes uncommitted for the parent Hermes agent to review.
5. Final response must include:
   - files changed
   - tests/validation run with results
   - any skipped checks and why
   - any follow-up work
```

Prefer parent-owned commits and PRs unless the user explicitly asked Claude Code to commit. Parent-owned commits keep the review gate outside the child worker.

## Supervision loop

1. Start Claude Code in print mode for bounded jobs, ideally with JSON output:
   ```bash
   claude -p "$(< /tmp/prompt.md)" \
     --output-format json \
     --max-turns 20 \
     --max-budget-usd 5 \
     --allowedTools "Read,Edit,Write,Bash"
   ```
2. If running in the background, wait or poll until the process exits. Do not stop at "started" unless the user explicitly requested a background launch only.
3. Parse the result for `subtype`, `terminal_reason`, `session_id`, cost, and permission/tool errors.
4. Inspect git state directly:
   ```bash
   git status --short --branch
   git diff --stat
   git diff --check
   git log --oneline --decorate --max-count=5
   ```
5. If Claude reports success but git shows no relevant changes, treat it as incomplete and investigate before reporting success.

## Independent parent review

Before committing or opening a PR, the Hermes parent must review the result:

- Scope: changed files match the user's request and no unrelated cleanup snuck in.
- Safety: no secrets, token-bearing remotes, credentials, or private paths in committed files.
- Portability: no hardcoded `/Users/...`, venv interpreters, local temp paths, or machine-specific assumptions unless intentionally documented.
- Tests: run the repo's expected local verification, or at minimum the targeted tests plus syntax/compile checks for touched languages.
- Docs: if behavior changed, user-facing docs or skill docs were updated where appropriate.

Useful commands:

```bash
git diff --stat origin/main...HEAD 2>/dev/null || git diff --stat
git diff --check
# project-specific validation goes here, e.g. scripts/run_tests.sh or pytest
```

If the child made partial or broken changes, fix them in the parent or relaunch Claude Code with the exact gaps. Do not commit a branch solely because Claude said it was done.

## Commit protocol

After review and validation:

1. Ensure the branch is not `main` unless the user explicitly authorized direct mainline commits.
2. Stage only intended files:
   ```bash
   git add <intended files>
   git status --short
   ```
3. Commit with a conventional commit message:
   ```bash
   git commit -m "docs: add Claude Code subagent orchestration guidance"
   ```
4. Re-check cleanliness:
   ```bash
   git status --short --branch
   ```

If Claude already created commits, inspect them with `git show --stat --oneline HEAD` and amend/follow up only if needed. Avoid rewriting a pushed PR branch unless necessary; use `--force-with-lease` if you do.

## PR protocol

After committing, check whether a PR already exists for the current branch before creating one:

```bash
BRANCH=$(git branch --show-current)
gh pr view --json number,title,url,headRefName,baseRefName,state 2>/dev/null || \
  gh pr list --head "$BRANCH" --json number,title,url,headRefName,baseRefName,state
```

- If a PR exists: push the branch, update the PR body or add a comment with validation results when useful.
- If no PR exists and the branch is not `main`: push and create one using a temp body file:
  ```bash
  git push -u origin HEAD
  cat > /tmp/pr-body.md <<'EOF'
  ## Summary
  - ...

  ## Test Plan
  - [x] ...
  EOF
  gh pr create --title "..." --body-file /tmp/pr-body.md
  ```
- If the work landed directly on `main`, provide the commit URL instead of pretending there is a PR.

## User notification

Notify the user when the child work has actually reached a durable state:

- Local-only completion: final response with summary, changed files, validation, and any blockers.
- PR opened/updated: include PR URL, branch, commit SHA, and local/CI status separately.
- Background or out-of-band run: if a messaging tool is available and the user asked for notification, send a short message via the current/home channel after commit/PR creation. Keep the notification factual, not chatty.

Example notification:

```text
Claude Code subagent run finished. Reviewed and committed changes on docs/claude-code-subagent-orchestration, opened PR #123, local validation passed; CI pending: <url>
```

## Pitfalls

1. **Trusting child summaries.** Always inspect git state and files directly.
2. **Opening duplicate PRs.** `gh pr list --head` can miss some cases; if `gh pr create` says a PR already exists, treat that URL as authoritative and view it.
3. **Letting background runs disappear.** Capture stdout/stderr paths, session IDs, and process IDs before waiting or reporting.
4. **Committing local machine details.** Subagents often copy exact paths from prompts or logs; search/skim before committing.
5. **Reporting CI as green from local tests.** Say local validation passed and CI is pending/missing/failing separately.
