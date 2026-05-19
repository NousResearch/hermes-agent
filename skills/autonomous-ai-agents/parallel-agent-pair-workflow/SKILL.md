---
name: parallel-agent-pair-workflow
description: "Use when heavy work needs Claude+Codex pair gates."
version: 1.2.0
author: MATTHEW ISSEI LAABAN + Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [parallel, delegation, claude-code, codex, orchestration, review, workflow, git-safety]
    related_skills: [claude-code, codex, subagent-driven-development, writing-plans, kanban-orchestrator, github-pr-workflow]
---

# Parallel Agent Pair Workflow

Split heavy multi-file work across parallel agent pairs: each pair has a Claude Code implementer and a Codex reviewer running in an isolated git worktree, with Hermes as commander and integrator. Multiple pairs run concurrently (max 4) without stepping on each other. Six sequential gates enforce quality and safety before any merge reaches a human reviewer.

## When to Use

- **Large tasks** (> 8 files, > 3 h estimated, multiple subsystems): multiple parallel pairs
- **Medium tasks** (3-8 files, 1-3 h, one subsystem): one pair (one implementer + one reviewer)
- **Small tasks** (< 3 files, single concern): handle directly or with a single subagent -- no pair needed
- **Dangerous / irreversible tasks** (drops data, rewrites migrations, touches prod secrets, force-push): **stop immediately and ask the user -- do not proceed autonomously**

When in doubt, classify up. A medium classified as large wastes a little time; a large classified as medium creates integration debt.

## Prerequisites

**Required CLIs (verify before pre-flight):**
- `claude` -- Claude Code CLI; `claude --version`
- `codex` -- OpenAI Codex CLI; `codex --version`

**Optional CLI (only for the optional pre-PR review step):**
- `roborev` -- local AI review/history helper; `roborev version`

If Codex is unavailable, a `delegate_task` fallback exists but is **degraded mode only** and requires explicit user approval before use (see Procedure, Step 6). It is not equivalent to Codex. If roborev is unavailable, skip only the optional roborev pre-PR review step; do not skip required gates.

**Required credentials:**
- `ANTHROPIC_API_KEY` set, or Claude OAuth active
- `OPENAI_API_KEY` set, or Codex OAuth active

**Repository state:**
- Clean git working tree (`git status` shows no uncommitted changes)
- Working on a feature branch (not `main`/`master`/`$BASE_BRANCH`)
- No orphaned worktrees with conflicting names (`git worktree list`)

**Dependency policy:** This workflow must not add packages or libraries to the project. If a task requires new dependencies, pause and ask the user for confirmation before proceeding.

## How to Run

1. Set `BASE_BRANCH`, `BASE_REF`, and unique `RUN_ROOT`; run Pre-flight gate (Gate 1)
2. Create isolated worktrees and lock push in each
3. Launch implementers in background; capture PIDs or session handles
4. Poll until implementers complete; inspect result JSON
5. Run Pair spec gate (Gate 2) per pair
6. Run Codex quality/security gate (Gate 3) per pair; loop up to 3x
7. Create isolated integration worktree; merge pair branches
8. Run Integration conflict gate (Gate 4); stop and ask user if conflicts arise
9. Run Final tests + PR gate (Gate 5); (optional) run roborev pre-PR review for large/risky PRs or when extra confidence is needed; create PR -- stop, do not merge
10. If any gate is unresolvable: Escalation/abort gate (Gate 6)

## Quick Reference

### Gate Summary (run in order, never skip)

| # | Gate name | Trigger | Fail action |
|---|-----------|---------|-------------|
| 1 | Pre-flight gate | Before dispatching any pair | Fix blocker or stop |
| 2 | Pair spec gate | After each implementer finishes | Re-dispatch implementer with gap list |
| 3 | Codex quality/security gate | After spec gate passes | Fix -> re-review (max 3 loops) -> escalate |
| 4 | Integration conflict gate | After all pairs pass Gate 3 | Stop, report conflict details, ask user |
| 5 | Final tests + PR gate | After integration clean | Stop; repair only after explicit user approval |
| — | roborev pre-PR review (optional) | Before PR creation; large/risky PRs or on request | Report critical/important issues; repair only after explicit user approval; rerun roborev (max 3 loops); skip if unavailable or clean |
| 6 | Escalation/abort gate | Any unresolvable failure | Stop all pairs, report, wait for user |

### Task Classification

| Size | Criteria | Workflow |
|------|----------|----------|
| Small | < 3 files, < 1 h, single concern | Direct or single subagent -- no pair |
| Medium | 3-8 files, 1-3 h, one subsystem | One pair (one implementer + one reviewer) |
| Large | > 8 files, > 3 h, multiple subsystems | Multiple parallel pairs + integration gate |
| Dangerous | Deletes data, drops tables, prod secrets, force-push | **Stop. Ask user. Do not proceed autonomously.** |

### Architecture

```
Hermes (commander/integrator)
|
+-- Gate 1: Pre-flight ──────────────── block if unsafe
|
+-- Pair 1 (worktree: $RUN_ROOT/wt-phase1-<task-a>, branch: phase1-<run-id>-<task-a>)
|   +-- Claude Code implementer  -->  code written
|   +-- Codex reviewer           -->  spec + quality/security check
|
+-- Pair 2 (worktree: $RUN_ROOT/wt-phase1-<task-b>, branch: phase1-<run-id>-<task-b>)
|   +-- Claude Code implementer  -->  code written
|   +-- Codex reviewer           -->  spec + quality/security check
|
+-- ... (up to 4 pairs in parallel)
|
+-- Gate 4: Integration conflict ────── merge + check
|
+-- Gate 5: Final tests + PR ────────── pass, or stop and ask before repair
```

## Procedure

### Step 1 -- Pre-flight Gate (Gate 1)

Set `BASE_BRANCH`, `BASE_REF`, and a unique run directory before anything else:

```bash
# Fetch and detect repo default branch. If detection fails, stop and ask the user for the base branch.
git fetch origin
BASE_BRANCH=$(git remote show origin 2>/dev/null | awk '/HEAD branch/{print $NF}')
if [ -z "$BASE_BRANCH" ]; then
  echo "Could not detect BASE_BRANCH. Ask the user which branch to base this work on."
  exit 1
fi
BASE_REF="origin/$BASE_BRANCH"
if ! git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
  echo "Could not verify BASE_REF=$BASE_REF after fetch. Stop and inspect remotes."
  exit 1
fi

RUN_ROOT=$(mktemp -d -t hermes-agent-pairs.XXXXXX)
RUN_ID=$(basename "$RUN_ROOT")
RESULT_DIR="$RUN_ROOT/results"
AUTH_WT="$RUN_ROOT/wt-phase1-auth"
API_WT="$RUN_ROOT/wt-phase1-api"
INTEGRATION_WT="$RUN_ROOT/wt-integration"
AUTH_BRANCH="phase1-${RUN_ID}-feature-auth"
API_BRANCH="phase1-${RUN_ID}-feature-api"
INTEGRATION_BRANCH="integration/${RUN_ID}-phase1-staging"
mkdir -p "$RESULT_DIR"

echo "BASE_BRANCH=$BASE_BRANCH"
echo "BASE_REF=$BASE_REF"
echo "RUN_ROOT=$RUN_ROOT"
echo "AUTH_BRANCH=$AUTH_BRANCH"
echo "API_BRANCH=$API_BRANCH"
echo "INTEGRATION_BRANCH=$INTEGRATION_BRANCH"
```

Checklist -- fix all failures before proceeding:

- [ ] Task classified (not dangerous/irreversible)
- [ ] Written spec exists (from `writing-plans` skill or user requirements)
- [ ] Clean working tree: `git status` shows no uncommitted changes
- [ ] Current branch is NOT `$BASE_BRANCH`
- [ ] `BASE_REF=origin/$BASE_BRANCH` exists after `git fetch origin`
- [ ] `RUN_ROOT`, `RESULT_DIR`, worktree paths, and branch names are unique for this run
- [ ] No conflicting worktrees: `git worktree list`
- [ ] `claude --version` succeeds
- [ ] `codex --version` succeeds (if it fails: ask user for approval before using delegate_task fallback)
- [ ] Claude auth is active (`ANTHROPIC_API_KEY` or OAuth)
- [ ] Codex auth is active (`OPENAI_API_KEY` or OAuth)
- [ ] No secrets or PII in the task spec text
- [ ] No new project dependencies required (if yes: ask user before continuing)

### Step 2 -- Create Isolated Worktrees

Result JSON lives under the unique `$RESULT_DIR`, outside every worktree, so cleanup of a worktree never destroys output. Worktrees live under the unique `$RUN_ROOT` to avoid collisions with another run:

```bash
# Naming: include RUN_ID to avoid branch collisions across concurrent runs.
git worktree add -b "$AUTH_BRANCH" "$AUTH_WT" "$BASE_REF"
git worktree add -b "$API_BRANCH"  "$API_WT"  "$BASE_REF"

# Mandatory: block accidental push in every worktree before launch
git -C "$AUTH_WT" config push.default nothing
git -C "$API_WT"  config push.default nothing
```

Remove when done, after PR creation or user-approved abort:
```bash
git worktree remove "$AUTH_WT"
git worktree remove "$API_WT"
# Keep or archive "$RESULT_DIR" until the final report is delivered.
```

### Step 3 -- Launch Implementers in Background

`--disallowedTools` enforces command safety as a best-effort guard.

**Write prompt files before running this block** -- use Hermes `write_file` or your editor. Never embed arbitrary prompt text in a shell heredoc: if any line of the prompt exactly matches the heredoc sentinel (e.g. `EOF`), the shell terminates the heredoc early and silently drops the rest of the prompt or interprets it as shell commands.

```bash
# Prompt files must already exist. Create them with Hermes write_file or your editor:
#   $RESULT_DIR/prompt_auth.txt  -- implementer prompt for the auth pair
#   $RESULT_DIR/prompt_api.txt   -- implementer prompt for the api pair
if [ ! -f "$RESULT_DIR/prompt_auth.txt" ] || [ ! -f "$RESULT_DIR/prompt_api.txt" ]; then
  echo "ERROR: prompt files missing. Write them with Hermes write_file or your editor first."
  exit 1
fi

(cd "$AUTH_WT" && claude -p \
  --allowedTools "Read,Edit,Write,Bash" \
  --disallowedTools "Bash(git push*),Bash(git reset --hard*),Bash(rm -rf*),Bash(*DROP TABLE*),Bash(*deploy*)" \
  --max-turns 20 \
  --output-format json \
  < "$RESULT_DIR/prompt_auth.txt" > "$RESULT_DIR/auth.json") &
AUTH_PID=$!

(cd "$API_WT" && claude -p \
  --allowedTools "Read,Edit,Write,Bash" \
  --disallowedTools "Bash(git push*),Bash(git reset --hard*),Bash(rm -rf*),Bash(*DROP TABLE*),Bash(*deploy*)" \
  --max-turns 20 \
  --output-format json \
  < "$RESULT_DIR/prompt_api.txt" > "$RESULT_DIR/api.json") &
API_PID=$!

echo "Launched: auth PID=$AUTH_PID  api PID=$API_PID"
```

When launching via Hermes `terminal()` in background mode, capture the returned session handle instead of `$!` (see Step 4 polling).

### Step 4 -- Poll Until Implementers Complete

**Shell PID variant:**
```bash
wait $AUTH_PID; AUTH_EXIT=$?
wait $API_PID;  API_EXIT=$?
echo "auth exit=$AUTH_EXIT  api exit=$API_EXIT"

python3 -c "import json; r=json.load(open('$RESULT_DIR/auth.json')); print('auth subtype:', r.get('subtype'))"
python3 -c "import json; r=json.load(open('$RESULT_DIR/api.json'));  print('api subtype:',  r.get('subtype'))"
# Expect subtype: success for both
```

**Hermes tool-call variant:**
```python
# Start each implementer with terminal(background=True). The returned dict includes a session_id.
# Do not assume shell variables from a previous terminal() call persist here.
# Paste the concrete values printed during pre-flight, or define them explicitly.
# Write prompt files first — avoids shell quoting issues with multiline content:
AUTH_WT = "/tmp/hermes-agent-pairs.xxxxxx/wt-phase1-auth"
API_WT = "/tmp/hermes-agent-pairs.xxxxxx/wt-phase1-api"
RESULT_DIR = "/tmp/hermes-agent-pairs.xxxxxx/results"
write_file(path=f"{RESULT_DIR}/prompt_auth.txt", content="<PROMPT_A>")
write_file(path=f"{RESULT_DIR}/prompt_api.txt", content="<PROMPT_B>")
auth_proc = terminal(
    command=f"claude -p --allowedTools 'Read,Edit,Write,Bash' "
            f"--disallowedTools 'Bash(git push*),Bash(git reset --hard*),Bash(rm -rf*)' "
            f"--max-turns 20 --output-format json "
            f"< '{RESULT_DIR}/prompt_auth.txt' > '{RESULT_DIR}/auth.json'",
    workdir=AUTH_WT,
    background=True,
)
api_proc = terminal(
    command=f"claude -p --allowedTools 'Read,Edit,Write,Bash' "
            f"--disallowedTools 'Bash(git push*),Bash(git reset --hard*),Bash(rm -rf*)' "
            f"--max-turns 20 --output-format json "
            f"< '{RESULT_DIR}/prompt_api.txt' > '{RESULT_DIR}/api.json'",
    workdir=API_WT,
    background=True,
)

# Poll live output while work is running.
process(action="poll", session_id=auth_proc["session_id"])
process(action="poll", session_id=api_proc["session_id"])

# Wait for completion and inspect full logs if a process fails.
auth_done = process(action="wait", session_id=auth_proc["session_id"], timeout=600)
api_done = process(action="wait", session_id=api_proc["session_id"], timeout=600)
if auth_done.get("exit_code") != 0:
    process(action="log", session_id=auth_proc["session_id"])
if api_done.get("exit_code") != 0:
    process(action="log", session_id=api_proc["session_id"])
```

After both complete, inspect `$RESULT_DIR/*.json` and confirm `subtype == "success"` before proceeding.

### Step 5 -- Pair Spec Gate (Gate 2) per Pair

Run after each implementer finishes, before running Codex:

- [ ] All spec requirements implemented (no gaps, no scope creep)?
- [ ] File paths match spec?
- [ ] Function signatures and interfaces match spec?
- [ ] No extra files or features added beyond scope?
- [ ] No new packages or dependencies introduced without user approval?
- [ ] Pair branch is committed, clean, and ahead of `$BASE_REF` before review/merge?

Blocker check:
```bash
git -C "$AUTH_WT" status --short
git -C "$AUTH_WT" log -1 --oneline
test "$(git -C "$AUTH_WT" rev-list --count "$BASE_REF"..HEAD)" -gt 0
git -C "$API_WT" status --short
git -C "$API_WT" log -1 --oneline
test "$(git -C "$API_WT" rev-list --count "$BASE_REF"..HEAD)" -gt 0
# status output must be empty for each pair; rev-list count must be > 0.
```

**Fail action:** Dispatch implementer with specific gap list. Re-run spec gate after fix. Do not run Codex or integration until the pair branch is committed, clean, and ahead of `$BASE_REF`.

### Step 6 -- Codex Quality/Security Gate (Gate 3) per Pair

```bash
# Review changes against the detected base branch. Never use --dangerously-bypass-approvals-and-sandbox for reviewer runs.
codex review --base "$BASE_REF"
# Run with workdir=$AUTH_WT
```

Before Codex, re-verify the pair branch is committed, clean, and ahead of `$BASE_REF`:
```bash
test -z "$(git -C "$AUTH_WT" status --short)"
git -C "$AUTH_WT" log -1 --oneline
test "$(git -C "$AUTH_WT" rev-list --count "$BASE_REF"..HEAD)" -gt 0
```

Via Hermes terminal:
```python
BASE_REF = "origin/main"  # replace with the concrete value printed during pre-flight
AUTH_WT = "/tmp/hermes-agent-pairs.xxxxxx/wt-phase1-auth"
terminal(command=f'codex review --base "{BASE_REF}"',
         workdir=AUTH_WT, pty=True, timeout=120)
```

Review dimensions: SQL injection, XSS, command injection, hardcoded secrets/API keys, improper error handling, race conditions, missing tests, style violations.

**Fail action:** Fix critical/important issues; re-run Codex. Max 3 loops. If still failing -> escalate (Gate 6).

**Degraded mode -- only when Codex is unavailable AND user has explicitly approved:**
```python
delegate_task(
    goal="Review the implementation in the worktree against the spec",
    context="""
SPEC: <paste spec here>
FILES CHANGED: <list>
CHECK:
- [ ] All spec requirements implemented?
- [ ] No secrets, credentials, or PII in code?
- [ ] No destructive commands (rm -rf, DROP TABLE, force-push)?
- [ ] No obvious bugs or security holes?
- [ ] Tests pass?
OUTPUT: APPROVED or specific issues list.
    """,
    toolsets=['file', 'terminal']
)
```

This is not equivalent to Codex review. Inform the user of the substitution and log it in the integration summary.

### Step 7 -- Integration Conflict Gate (Gate 4)

Use an isolated integration worktree -- never check out or merge in the main workspace:

```bash
git worktree add -b "$INTEGRATION_BRANCH" "$INTEGRATION_WT" "$BASE_REF"
git -C "$INTEGRATION_WT" config push.default nothing

git -C "$INTEGRATION_WT" merge "$AUTH_BRANCH" --no-ff -m "integrate: auth"
git -C "$INTEGRATION_WT" merge "$API_BRANCH"  --no-ff -m "integrate: api"
git -C "$INTEGRATION_WT" diff --check
git -C "$INTEGRATION_WT" status
```

If conflicts arise, **stop immediately**. Report the conflicting files and each pair's intent to the user. Do not dispatch a repair subagent unless the user explicitly approves that follow-up action.

User-approved repair example (run **only after explicit user approval**):
```bash
claude -p "Resolve merge conflicts in <files>. Intent of each branch: auth=<summary>, api=<summary>. Preserve both intents." \
  --allowedTools "Read,Edit" \
  --disallowedTools "Bash(git push*),Bash(git reset --hard*)" \
  --max-turns 10
# workdir=$INTEGRATION_WT
```

**Fail action:** Stop. Report exact files and conflict context to the user. Do not attempt any autonomous repair, revert, cleanup, or merge continuation; wait for the user to decide what to do next.

### Step 8 -- Final Tests + PR Gate (Gate 5)

```bash
cd "$INTEGRATION_WT"

# Run whichever test command applies to the project
pytest -q
npm test
make test

# Diff and secret scan against BASE_REF
git diff "$BASE_REF"...HEAD --stat
# Prefer the project's configured secret scanner. If none exists, inspect
# the diff with Hermes search_files/read_file before creating the PR.
```

- [ ] All tests pass (no regressions)
- [ ] Secret scan or Hermes diff inspection finds no secrets
- [ ] Diff looks reasonable -- no unintended file changes

#### Optional roborev pre-PR review

Use for large/risky PRs or when extra confidence is needed before PR creation. Not required by default; skip for small/routine changes or when roborev is unavailable.

roborev is a pre-final/PR helper and review history layer. It does **not** replace the Codex Quality/Security Gate (Gate 3) or any final Codex judgment. Do **not** enable post-commit hooks or repo-wide daemon review.

```bash
# Preferred: review branch diff against base
roborev review --branch --base "$BASE_REF" --local --agent codex --reasoning thorough --wait

# Fallback: review uncommitted/staged changes
roborev review --dirty --local --agent codex --reasoning thorough --wait
```

If roborev reports CRITICAL or IMPORTANT issues, stop and report them to the user. Repair only after explicit user approval; if approved, fix and rerun Gate 5 tests/secret scan/diff inspection, then rerun roborev (max 3 loops). MINOR/cosmetic findings are advisory only -- no blocking required. If 3 approved repair loops do not resolve CRITICAL or IMPORTANT issues, escalate (Gate 6).

Only after all checks pass: create PR using `github-pr-workflow` skill.

**Stop here -- do not merge. Wait for human review.**

**Fail action:** Stop and report the failing tests. Ask the user before dispatching any post-integration repair subagent. If the user explicitly approves a repair loop, constrain it with the same file-ownership, dependency, secrets, committed-clean, and Codex review gates. If tests still fail after 3 user-approved repair loops -> escalate (Gate 6).

### Step 9 -- Escalation/Abort Gate (Gate 6)

| Situation | Required action |
|-----------|----------------|
| Task classified dangerous/irreversible | Stop immediately. Describe exactly what was found. Ask for explicit user authorization before any action. |
| Pre-flight has unfixable blocker | Stop. Report the blocker clearly. |
| Gate 3: critical issue after 3 repair loops | Stop all pairs. Report all open issues. Wait for user decision. |
| Gate 4: unresolvable merge conflict | Stop. Report exact files and conflict context. Do not revert or clean up autonomously; ask user what to do. |
| Gate 5: test failure after 3 user-approved repair loops | Stop. Report failing tests and last attempted fix. Do not merge. |
| Any pair attempts to write to `$BASE_BRANCH` directly | Stop immediately. Report the exact commands attempted and files affected. Ask the user what to do next before taking any further action. |

**Never merge to `$BASE_BRANCH` automatically.** Always stop at PR creation and confirm with the user.

---

## Output Format Templates

### Implementer Prompt Template

```
You are an expert software engineer implementing a specific task.

TASK: <one-sentence description>

SPEC:
<full spec text -- paste content directly, no external file references>

FILES YOU MAY MODIFY:
<explicit list of files this pair owns>

FILES YOU MUST NOT MODIFY (owned by other pairs):
<explicit list>

DEFINITION OF DONE:
- [ ] All spec requirements implemented
- [ ] Tests written and passing: <test command>
- [ ] No new lint errors: <lint command>
- [ ] Committed: git commit -m "<conventional commit message>"

CONSTRAINTS:
- Do not add features beyond the spec
- Do not touch files outside your assigned list
- Do not hardcode secrets or credentials
- Do not push to any remote branch
- Do not add new packages or dependencies without user approval
```

### Codex Reviewer Prompt Template

```
You are a code reviewer and security auditor. Review all changes in this worktree.

CONTEXT:
- Branch: <branch-name>
- Original spec: <one-paragraph summary>
- Implementer's intent: <summary>

REVIEW CHECKLIST:
Security:
- [ ] No hardcoded secrets, API keys, tokens, or passwords
- [ ] No SQL injection, XSS, or command injection vectors
- [ ] No unsafe deserialization or path traversal
- [ ] No dangerous shell commands (rm -rf, curl | sh, etc.)

Quality:
- [ ] All spec requirements implemented
- [ ] Tests cover happy path and at least one edge case
- [ ] No obvious logic bugs or off-by-one errors
- [ ] No unhandled exceptions that swallow errors silently

OUTPUT FORMAT:
CRITICAL: [blocking issues -- must fix before merge]
IMPORTANT: [should fix before merge]
MINOR: [cosmetic / optional]
VERDICT: APPROVED | REQUEST_CHANGES
```

### Integration Summary Template

```
## Integration Summary -- Phase <N>

**Pairs completed:** <N>
**Branches merged:** <list>
**Test result:** <pass/fail + counts>
**Codex used:** yes | no (degraded mode: delegate_task -- user approved)
**Open issues:** <none | list>

### Changes by subsystem
- <subsystem>: <one-line description of what changed>

### Conflicts resolved
- <file>: <resolution approach>

### What Hermes did NOT do
- Did not merge to $BASE_BRANCH (PR #<N> is open for review)
- Did not deploy to production
- Did not modify CI/CD configuration
- Did not add project dependencies
```

### PR Body Template

```markdown
## Summary
<!-- 2-3 bullet points of what this PR does -->

## Task Classification
<!-- small / medium / large -- reasoning -->

## Pairs used
| Pair | Branch | Implementer | Reviewer | Status |
|------|--------|-------------|----------|--------|
| 1 | phase1-<run-id>-<name> | Claude Code | Codex | APPROVED |

## Gates passed
- [x] Pre-flight gate
- [x] Pair spec gate (all pairs)
- [x] Codex quality/security gate (all pairs)
- [x] Integration conflict gate
- [x] Final tests + PR gate
- [ ] roborev pre-PR review (optional -- check if ran for this PR)

## Test results
<!-- Paste test output summary -->

## What reviewers should focus on
<!-- Areas of uncertainty or tradeoffs made -->

Generated with parallel-agent-pair-workflow skill
```

---

## Pitfalls

1. **Skipping the pre-flight gate** -- leads to wasted pair work when a blocker could have been caught early
2. **Letting pairs touch overlapping files** -- causes merge conflicts at Gate 4 that cannot be resolved automatically; assign non-overlapping file ownership before launch
3. **Running more than 4 pairs concurrently** -- exhausts VPS resources without proportional benefit
4. **Using dangerous Codex bypass flags for review** -- never use `--dangerously-bypass-approvals-and-sandbox` for Codex reviewer runs
5. **Treating "IMPORTANT" review findings as optional** -- important issues compound after merge; fix before merging
6. **Auto-merging after Gate 5** -- Hermes must always stop at PR creation and wait for human review; never merge programmatically
7. **Forgetting to remove worktrees** -- orphaned worktrees persist and confuse future runs; always clean up with `git worktree remove`
8. **Not providing explicit file ownership in implementer prompts** -- agents drift into shared files without explicit boundaries
9. **Using delegate_task without user approval** -- it is a degraded fallback, not equivalent to Codex; always ask the user first
10. **Storing result JSON inside a worktree** -- worktree removal deletes the file; always write to `$RESULT_DIR/`
11. **Not setting `push.default nothing`** -- without this, a misconfigured git remote can cause accidental pushes from worktrees
12. **Hardcoding `main` as the base branch** -- always detect `$BASE_BRANCH`, fetch origin, and use `$BASE_REF` during pre-flight; different repos use different defaults
13. **Skipping committed/clean checks** -- Codex and integration can silently miss uncommitted work; every pair branch must be committed, clean, and ahead of `$BASE_REF` before Gate 3/4
14. **Autonomous Gate 4 or Gate 5 repair** -- merge-conflict and post-integration test repair require explicit user approval first

## Verification

Before declaring the workflow complete:

- [ ] All pairs reached Gate 3 APPROVED verdict (Codex or user-approved degraded fallback)
- [ ] Integration worktree builds without errors
- [ ] Full test suite passes with no regressions
- [ ] Secret scan or Hermes diff inspection finds no API keys, passwords, or secrets
- [ ] Every pair worktree is committed, clean, and ahead of `$BASE_REF` before Codex/integration
- [ ] All worktrees removed: `git worktree list` shows only the main checkout
- [ ] `$RESULT_DIR/` cleaned up or archived after the final report
- [ ] PR created (not merged) with integration summary attached
- [ ] (Optional) roborev pre-PR review ran and found no CRITICAL/IMPORTANT issues -- required only for large/risky PRs or when explicitly requested
- [ ] User notified of PR URL and asked to review before any merge
