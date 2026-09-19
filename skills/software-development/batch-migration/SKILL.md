---
name: batch-migration
description: Parallel code migration across many files using isolated git worktrees. One agent per work unit, each creates its own PR. Use for "batch migrate", "migrate X to Y across the codebase", "bulk refactor". Manual invocation only.
version: 3.37.1-E2E
author: MorAlekss
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [migration, refactoring, parallel, git, worktree, batch, automation]
    category: software-development
    related_skills: [requesting-code-review]
  requires_toolsets: [terminal]
---

# batch-migration

Parallel code migration orchestration via isolated git worktrees.
**One agent per work unit. No shared state. Each unit produces its own PR.**

---

## COORDINATOR RULES - READ FIRST

**Before doing anything else, internalize these rules:**

- ☐ **NEVER edit source files yourself** - no `patch`, no `write_file` on `.py/.js/.ts` files. Ever. Not even if a worker gets stuck.
- ☐ **ALL file changes go to workers** - if you need to change one line, spawn a new worker.
- ☐ **DO NOT skip steps** - every step in this skill is required. "Simple" tasks still need worktrees and PRs.
- ☐ **DO NOT rewrite worker prompts** - copy the template exactly, fill in only the [bracketed] sections.
- ☐ **Manual invocation only** - never auto-trigger this skill. Only run when user explicitly asks.

**Test yourself:** "If I need to change one line in a file, do I edit it myself or spawn a worker?" → **SPAWN A WORKER.**

---

## When to Use

Activate **only** when user explicitly says:
- "batch migrate X to Y"
- "migrate all files from X to Y"
- "bulk refactor across the codebase"
- "parallel migration"

Skip for: single-file changes, small refactors under 3 files, or when user says "skip batch".

---

## Phase 1 - Research and Plan

### ☐ Step 1 - Explore the codebase

```bash
# Find all affected files
grep -r "PATTERN" --include="*.py" -l .
grep -r "PATTERN" --include="*.js" -l .
grep -r "PATTERN" --include="*.ts" -l .

# Count occurrences per file
grep -r "PATTERN" --include="*.py" -c . | sort -t: -k2 -rn

# Understand conventions
cat README.md 2>/dev/null | head -30
ls -la
```

→ Note codebase conventions - workers will need them.

### ☐ Step 1.1 - Discover project runtime and entry points

Find how to start the project and run verification. Skip sections that don't apply.

```bash
# Server start command (API projects)
grep -rn "uvicorn\|gunicorn\|flask\|django\|FastAPI\|app\.run" --include="*.py" --include="*.toml" --include="*.cfg" -l .
cat Makefile 2>/dev/null | grep -iE "serve|run|start|dev"
cat Procfile 2>/dev/null
cat docker-compose.yml 2>/dev/null | grep -A2 "command:"

# Dev server command (UI/frontend projects)
cat package.json 2>/dev/null | python3 -c "import sys,json; [print(f'{k}: {v}') for k,v in json.load(sys.stdin).get('scripts',{}).items()]" 2>/dev/null
cat vite.config.* next.config.* webpack.config.* 2>/dev/null | head -5

# CLI entry points
grep -A5 "entry_points\|console_scripts\|scripts" setup.py setup.cfg pyproject.toml 2>/dev/null
cat package.json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('bin',{}))" 2>/dev/null

# Test infrastructure
cat pytest.ini jest.config.* 2>/dev/null
ls tests/ __tests__/ test/ 2>/dev/null
```

→ Record: server start command, port, test command, CLI entry point.

### ☐ Step 2 - Run baseline tests

```bash
python3 -m pytest --tb=no -q 2>&1 | tail -5   # Python
npm test -- --passWithNoTests 2>&1 | tail -5   # Node
cargo test 2>&1 | tail -5                       # Rust
go test ./... 2>&1 | tail -5                    # Go
```

→ **Record the exact baseline result** — note both passing count AND names of any pre-existing failures:
```
Baseline: X passed, Y failed
Pre-existing failures (do NOT count as regression):
- test_full_purchase_flow (reason: ...)
```
→ In the final summary, compare NEW failures only against this list.

### ☐ Step 3 - Decompose into work units

Each unit must be:
- Independently implementable (no shared state with siblings)
- Mergeable on its own without depending on another unit's PR
- **Sliced per-file** - one unit per source file. Do NOT group multiple files into one unit.
- **No two units touch the same file** - if conflict, merge those units

**☐ Shared file check (do this before finalizing units):**
After listing all units, verify that no file appears in more than one unit:
```
Manually verify no file appears in more than one unit before finalizing.
An automated duplicate check will catch any misses later before spawning.
```
If a test file is shared by multiple source units (e.g. `test_users.py` tests `profile.py`, `admin.py`, `preferences.py`) - read the test file imports to understand which source modules it tests. Assign the test file to ONE unit only (the last source file unit in that module). All other source units must NOT include that test file.

IMPORTANT: Before finalizing units, read each test file and check which source modules it imports. Do not assume a test file belongs to only one module based on its name — always verify by reading its imports.

**☐ Orphaned test file check (CRITICAL — do after unit assignment):**
After assigning test files to units, run this check:
```bash
# Find ALL test files that mock the migrated pattern
grep -rn "OLD_PATTERN" tests/ --include="*.py" -l

# For each test file found, verify it's assigned to a unit:
for test_file in $(grep -rn "OLD_PATTERN" tests/ --include="*.py" -l); do
  echo "=== $test_file ==="
  # Check if any unit's "Files to modify" includes this test file
  grep -r "$test_file" /tmp/hermes-batch/*-prompt.txt > /dev/null \
    && echo "✓ Assigned to a unit" \
    || echo "✗ ORPHANED — no unit owns this test file! Must assign to one unit."
done
```
**Every test file that mocks the old pattern MUST be owned by exactly one unit.** Unowned test files will retain stale mocks that pass in worktrees (because `patch()` creates attributes) but FAIL after all PRs merge. This is the #1 cause of post-merge test failures.

Target: 5-20 units. Maximum 30 for large codebases.

### ☐ Step 4 - Determine E2E test recipe

How will workers verify their changes? Fill commands into each worker's
"E2E test recipe" section. Use a template below or write your own.
If no concrete path — ask user via clarify tool.

```bash
# Source-only (default):
python3 -m pytest --tb=short -q

# API:
[server-start] &
sleep 2
curl -s localhost:PORT/api/[endpoint] | jq .
kill %1
python3 -m pytest --tb=short -q

# UI:
[dev-server] &
sleep 3

# Then fill worker's "Browser verification" section with:
#   - browser_navigate("http://localhost:PORT/[page]")
#   - browser_snapshot()
#   - browser_click(ref="@e[N]")
#   - browser_vision(question="Any layout breaks?")
#   - kill %1

# CLI:
OUTPUT=$([cli-cmd] [flags] 2>&1)
echo "$OUTPUT" | grep "[expected]" || { echo "FAIL"; exit 1; }

# Combine — a unit can use multiple:
python3 -c "from src.module import func; print('OK')"
python3 -m pytest tests/unit/test_[name].py -v
```

### ☐ Step 5 - Present plan and wait for approval

```python
clarify(
    question="""## Batch Migration Plan

**Goal:** [original user instruction]
**Pattern to migrate:** [what is being changed]
**Files affected:** [count] files across [count] modules

### Work Units
1. [unit title] - [files] - [brief description]
2. [unit title] - [files] - [brief description]

### E2E Test Recipe
[exact commands workers will run]

### Estimated: [N] PRs will be created

Proceed with migration?""",
    choices=["Yes, proceed", "No, cancel", "Adjust the plan"]
)
```

→ **STOP. Do not proceed to Phase 2 until user selects "Yes, proceed".**

**☐ Step 5.1 - Prepare main branch before creating worktrees:**

Once plan is approved, make all necessary changes to main in ONE commit — before creating worktrees. This ensures all branches inherit these changes and there are no conflicts later.

**☐ Step 5.1.1 - Save ROOT_DIR and update `.gitignore`:**
```bash
ROOT_DIR=$(pwd)
# Add .hermes/ so workers don't add .gitignore noise to PRs:
grep -qxF ".hermes/" $ROOT_DIR/.gitignore 2>/dev/null || echo ".hermes/" >> $ROOT_DIR/.gitignore
# Add .worktrees/ to prevent hermes sub-worktree metadata from being committed:
grep -qxF ".worktrees/" $ROOT_DIR/.gitignore 2>/dev/null || echo ".worktrees/" >> $ROOT_DIR/.gitignore
```

**☐ Step 5.1.2 - Update `requirements.txt` — add new AND remove old dependencies:**
```bash
# Add new dependencies (uncomment and adjust as needed):
# grep -q "httpx" $ROOT_DIR/requirements.txt || echo "httpx>=0.27.0" >> $ROOT_DIR/requirements.txt
# grep -q "pytest-asyncio" $ROOT_DIR/requirements.txt || echo "pytest-asyncio>=0.23.0" >> $ROOT_DIR/requirements.txt
# Remove replaced dependencies (uncomment and adjust as needed):
# sed -i '/^requests/d' $ROOT_DIR/requirements.txt
```

**☐ Step 5.1.3 - Commit all changes together in ONE commit:**
```bash
git add $ROOT_DIR/.gitignore $ROOT_DIR/requirements.txt
git diff --cached --quiet || git commit -m "chore: prepare main for batch migration (gitignore + dependencies)"
```

**☐ Step 5.1.4 - Push main to remote:**
```bash
git push origin main
```

→ This is required so GitHub knows about the main branch. Without this, `gh pr create --base main` will fail because GitHub cannot find main on remote. Always push before creating worktrees.

→ Always run all four sub-steps. Adjust commented lines to match your migration pattern.

---

## Phase 2 - Execute

**FILE EDITING BAN: You are forbidden from editing source files. All changes go to workers.**

### ☐ Step 6 - Prepare worktrees and spawn workers

Complete each sub-step in order. Do not skip any.

**☐ Step 6.1 - Navigate to project root (do this FIRST)**
```bash
cd /path/to/project  # cd into the project root where .git lives
pwd  # verify you are in the right directory
```
→ All subsequent commands must run from the project root. Never work from a sandbox or temp directory.

**☐ Step 6.2 - Verify gh auth**
```bash
gh auth status
```
→ Must show authenticated. If not - stop and fix auth first.

**☐ Step 6.3 - Save ROOT_DIR**
```bash
ROOT_DIR=$(pwd)
echo "ROOT_DIR=$ROOT_DIR"
```
→ Write down the exact value. You will use it everywhere. Never use relative paths.

**☐ Step 6.4 - Create worktrees**
```bash
mkdir -p $ROOT_DIR/.hermes/worktrees
for i in $(seq 1 8); do
  git worktree add $ROOT_DIR/.hermes/worktrees/unit-$(printf "%02d" $i) -b batch/unit-$(printf "%02d" $i) main
done
# Replace 8 with actual number of units
```

**☐ Step 6.5 - Initialize todo tracking**

Create one todo per unit with status `in_progress`. Update status as workers finish.

**☐ Step 6.6 - Write worker prompts**

**☐ Step 6.6.1 - Copy the template below EXACTLY. Fill in ONLY the [bracketed] sections. Do NOT remove or rewrite any steps - especially `/requesting-code-review` and `PR: <url>`.**

**☐ Special rule for units that include a test file:** Before writing the prompt:
1. Run `grep -n "patch(" [test file]` to get ALL mock patches in the file
2. For EACH patch, determine which source module it belongs to and what it must become
3. Fill in the `## Mock patches to update` section as a table — every row is mandatory, no exceptions
4. Run `grep "^from src\." [test file]` to find ALL source modules imported by the test file
5. For EACH imported function from a source module owned by ANOTHER unit:
   - Check the migration instructions for that unit — will it change this function to `async def`?
   - If YES → add to `## Cross-unit async dependencies` table in the worker prompt
   - If NO → skip, test doesn't need async changes for that function

The table must include ALL mocks for ALL source modules covered by the test file — not just the module in this unit. Also add this warning to the prompt: "These mocks MUST be updated regardless of whether tests currently pass — mocks may pass in your worktree because `patch()` creates the attribute, but will fail after all PRs merge."

- Each unit that starts a server must use a unique port. Assign: unit-01 → 8091, unit-02 → 8092, etc. Write the port in both the E2E test recipe and Browser verification sections.

**☐ Step 6.6.2 - Write prompts to `/tmp/hermes-batch/` — NOT inside worktrees. This prevents worker_prompt.txt from ever being committed.**
```bash
mkdir -p /tmp/hermes-batch
cat > /tmp/hermes-batch/unit-01-prompt.txt << 'WORKER_PROMPT'
You are a batch migration worker. Your task is strictly scoped.
You have NO access to the coordinator's context - everything you need is below.

## Your worktree path
[fill in: $ROOT_DIR/.hermes/worktrees/unit-XX - coordinator must replace this with the exact absolute path]
cd into this directory FIRST. Run `pwd` to verify. If not in your worktree - STOP and cd there.

## Overall goal
[user's original migration instruction]

## Your specific task
[unit title and description]

## Files to modify
[exact list of files for this unit only]

## What to change
[exact migration: from X pattern to Y pattern with example]

## Mock patches to update
[Fill this in ONLY if your unit includes a test file. Use a table format — every row is mandatory:]
[Example:
| Test function | Current mock | Must become |
|--------------|-------------|-------------|
| test_get | patch('src.foo.requests.get') | patch('src.foo.httpx.AsyncClient') |
| test_post | patch('src.bar.requests.post') | patch('src.bar.httpx.AsyncClient') |

WARNING: These mocks MUST be updated regardless of whether tests currently pass.
Mocks may pass in your worktree because `patch()` creates the attribute on the module object.
After ALL PRs merge, these mocks WILL fail. Update them NOW, not later.
]
[If no test file in this unit — remove this section entirely]

## Cross-unit async dependencies
[Fill this in ONLY if your test file imports functions from source modules migrated by OTHER units that are changing from sync to async. If yes, list every affected test function:]
[Example:
Your test file calls these functions from OTHER units' source modules that are becoming async:

| Source module | Functions called in tests | Action required |
|--------------|--------------------------|-----------------|
| src.utils.http (unit-2) | get, post | Make test_get async + await get(...) / Make test_post async + await post(...) |

For EACH function listed above:
1. Find the test function that calls it
2. Add @pytest.mark.asyncio decorator
3. Make the test function async def
4. Add await before the function call
5. Import pytest if not already imported
6. **CRITICAL: When making test functions async, ensure ALL calls to async functions are awaited.** This is the #1 post-merge failure pattern — tests pass in worktrees (because `patch()` creates attributes) but fail after merge when async functions return coroutines instead of results.

IMPORTANT: When patching these other modules' httpx, always use `create=True`:
`patch('src.other_module.httpx.AsyncClient', create=True)` — this avoids scope violations.
Do NOT add imports to other units' files. `create=True` works both before and after the other unit's PR merges.
]
[If no cross-unit async dependencies — remove this section entirely]

## Codebase conventions
[conventions discovered during research]

## E2E test recipe
[exact commands to verify the changes work end-to-end]

## Browser verification
[Fill this in ONLY for UI verification patterns. For non-UI patterns write "N/A".
Include browser tool calls to verify the migration:]
[Example:
1. browser_navigate("http://localhost:3000/dashboard")
2. browser_snapshot() — verify form renders
3. browser_click(ref="@e5") — submit form
4. browser_snapshot() — verify result
5. browser_vision(question="Any layout breaks or errors?")
6. kill %1 — stop dev server
]

## Instructions
Safety rules - follow these before anything else:
- ☐ **FIRST: cd into your worktree** - run `cd [absolute path to your worktree]` and verify with `pwd`. If you are NOT in your worktree directory - STOP and cd there. NEVER edit files outside your worktree.
- ☐ NEVER delete source files. If migration fails, revert: git checkout -- <file>
- ☐ Install required dependencies before starting: pip install <package> / npm install. If pip not found - use `source venv/bin/activate` first or `python3 -m pip install`.
- ☐ Set PYTHONPATH if needed: `export PYTHONPATH=$(pwd)` - required if tests import from `src/` directly.
- ☐ Run baseline tests BEFORE any changes: [test command] - note which tests pass
- ☐ **STRICT SCOPE**: Modify ONLY the files listed in "Files to modify". HARD RULE: NEVER edit any file not in your list - not even test files. If tests fail because other files are not yet migrated - commit anyway and note it in the PR. Do NOT fix files assigned to other workers.
- ☐ **If your test file imports source modules assigned to OTHER units** — do NOT migrate those source modules. Only migrate files in your "Files to modify" list. Tests WILL fail until those units complete — this is expected. Commit anyway and note it in the PR.
- ☐ **If your unit includes a test file** — you are the SOLE owner of that file. Update EVERY mock target (`patch('...')` string argument) to match the new pattern. Changing a mock target string does NOT mean you must migrate the corresponding source module — leave source files assigned to other units untouched. Your test functions for other modules' source files WILL pass because `patch()` creates the mocked attribute on the module object.
- ☐ **DO NOT touch `.gitignore` or `requirements.txt`** — coordinator already committed these to main. If you see them modified, run `git checkout -- .gitignore requirements.txt` before committing.
- ☐ **NEVER run `git push origin main`** — only push to your assigned branch: `git push origin HEAD:batch/unit-XX`

## Async mock pattern
When updating mocks for async httpx functions, use this EXACT pattern:
```python
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

@pytest.mark.asyncio
async def test_example():
    mock_response = MagicMock()
    mock_response.json.return_value = {"key": "value"}
    mock_response.raise_for_status.return_value = None
    with patch('src.module.httpx.AsyncClient') as MockClient:
        mock_client = AsyncMock()
        mock_client.METHOD.return_value = mock_response  # replace METHOD with get/post/put/patch/delete
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        MockClient.return_value = mock_client
        result = await function_under_test(...)
        assert result == expected
```

**Cross-unit mock safety — use `create=True`:**
When your test file mocks a source module owned by ANOTHER unit (not yet migrated in your worktree), `patch('src.other_module.httpx.AsyncClient')` will FAIL because `httpx` isn't imported there yet. **Do NOT add imports to other units' files — this is a scope violation.** Instead, use `create=True`:
```python
with patch('src.other_module.httpx.AsyncClient', create=True) as MockClient:
```
`create=True` tells `patch()` to create the attribute if it doesn't exist. This works in your worktree (where the other module still uses `requests`) AND after merge (where the other module uses `httpx`). No scope violation needed.

After implementing the change:
1. Review - run `/requesting-code-review`. **STOP. Do not proceed to step 2 until review is complete and all issues are fixed.**
2. Re-review - if you made ANY additional changes after step 1, run `/requesting-code-review` again before committing.
3. Run unit tests - run: [test command]. If tests fail due to OTHER units' files not being migrated - that is expected, note it in PR.
4. Test end-to-end - follow the e2e test recipe above
5. **Pre-commit scope check** — run this BEFORE committing. Copy-paste EXACTLY — do NOT modify the ALLOWED_FILES line. If ANY file outside your "Files to modify" list appears, remove it with `git checkout -- <file>` before proceeding:
```bash
# DO NOT ADD FILES TO THIS LINE. Only files from your "Files to modify" section.
ALLOWED_FILES="[list your Files to modify here, space-separated]"
for f in $(git diff --name-only); do
  echo "$ALLOWED_FILES" | grep -qw "$f" || { echo "SCOPE VIOLATION: $f — run: git checkout -- $f"; exit 1; }
done
```
**CRITICAL:** NEVER add test files to ALLOWED_FILES unless your unit's "Files to modify" section explicitly lists that test file. Workers that "helpfully" update test files they don't own WILL cause merge conflicts.
6. **Mock verification** — if your unit includes a test file, run this BEFORE committing. If any stale mock is found, update it NOW — do NOT wait for tests to fail after merge:
```bash
grep -n "requests\." [your test file] && { echo "STALE MOCKS FOUND — update before committing"; exit 1; } || echo "✓ No stale mocks"
```
**IMPORTANT:** Mocks that reference `requests` may still PASS in your worktree because `patch()` creates the attribute on the module object. But after ALL PRs merge, these mocks WILL fail. Update them regardless of whether tests currently pass.
7. Commit and push - git add -A && git commit -m "batch: [unit title]" && git push origin HEAD:batch/unit-01
8. Create PR - gh pr create --title "Batch: [unit title]" --body "[brief description]"
   If PR already exists: gh pr edit <number> --body "[brief description]" and use existing URL
9. Report - end your response with exactly one line: PR: <url>
   If no PR was created, end with: PR: none - <reason>
WORKER_PROMPT
```

**☐ Step 6.7 - STOP. Self-verify prompts before spawning**

Run ALL checks. Every check must pass before spawning:
```bash
# Check /requesting-code-review is in every prompt
for f in /tmp/hermes-batch/*-prompt.txt; do
  grep -q "requesting-code-review" "$f" \
    && echo "✓ $(basename $f) has review step" || echo "✗ MISSING REVIEW STEP in $(basename $f) - fix before spawning"
done

# Check PR report line is in every prompt
for f in /tmp/hermes-batch/*-prompt.txt; do
  grep -q "PR: <url>" "$f" \
    && echo "✓ $(basename $f) has PR report" || echo "✗ MISSING PR REPORT in $(basename $f) - fix before spawning"
done

# Check STRICT SCOPE warning is in every prompt
for f in /tmp/hermes-batch/*-prompt.txt; do
  grep -q "STRICT SCOPE" "$f" \
    && echo "✓ $(basename $f) has scope warning" || echo "✗ MISSING SCOPE WARNING in $(basename $f) - fix before spawning"
done

# Check worktree paths in prompts exist on disk
for f in /tmp/hermes-batch/*-prompt.txt; do
  path=$(grep -o "cd /[^ ]*worktrees/unit-[^ ]*" "$f" | head -1 | awk '{print $2}')
  if [ -n "$path" ]; then
    [ -d "$path" ] && echo "✓ $path exists" || echo "✗ MISSING PATH: $path - fix before spawning"
  fi
done

# Manual check — verify each prompt's "Files to modify" contains ONLY its assigned files
echo "=== Files to modify per prompt (verify no cross-unit scope violations) ==="
for f in /tmp/hermes-batch/*-prompt.txt; do
  echo "--- $f ---" && grep -A10 "Files to modify" "$f" | head -10
done

# Check no file appears in more than one unit's scope
echo "=== Duplicate file check ==="
ALL_FILES=$(for f in /tmp/hermes-batch/*-prompt.txt; do
  sed -n '/^## Files to modify$/,/^## What to change$/p' "$f" | grep -v "^## " | grep -v "^$" | while read line; do
    echo "$(basename $f) $line"
  done
done)
DUPES=$(echo "$ALL_FILES" | awk '{print $2}' | sort | uniq -d)
if [ -n "$DUPES" ]; then
  echo "✗ DUPLICATE FILES ACROSS UNITS:"
  for dup in $DUPES; do
    echo "  $dup — assigned to: $(echo "$ALL_FILES" | grep -F " $dup" | awk '{print $1}')"
  done
  echo "  Fix before spawning."
else
  echo "✓ No duplicate files across units"
fi

# Check no port duplicates across units (for API/UI patterns)
echo "=== Port uniqueness check ==="
PORTS=$(grep -oh "localhost:[0-9]\{4\}" /tmp/hermes-batch/*-prompt.txt 2>/dev/null | grep -oP '\d{4}' | sort)
DUPES=$(echo "$PORTS" | uniq -d)
if [ -n "$DUPES" ]; then
  echo "✗ DUPLICATE PORTS: $DUPES — assign unique ports before spawning"
else
  echo "✓ No duplicate ports"
fi

# Check UI units have browser verification filled (not N/A)
echo "=== Browser verification check ==="
for f in /tmp/hermes-batch/*-prompt.txt; do
  FILES=$(sed -n '/^## Files to modify$/,/^## What to change$/p' "$f")
  if echo "$FILES" | grep -q "\.jsx\|\.tsx\|\.vue\|\.html\|\.svelte"; then
    grep -A1 "Browser verification" "$f" | grep -q "N/A" \
      && echo "✗ $(basename $f) has UI files but Browser verification is N/A — fill before spawning" \
      || echo "✓ $(basename $f) browser verification filled"
  fi
done
```
→ If ANY check fails - fix the prompt. Do NOT spawn until all show ✓.

**☐ Step 6.8 - Spawn all workers simultaneously**

**Primary method - always try this first:**
```python
# REQUIRED: use terminal(background=True) - NOT shell &
terminal(
    command='hermes chat -w --yolo -q "$(cat /tmp/hermes-batch/unit-01-prompt.txt)"',
    background=True,
    workdir=ROOT_DIR + '/.hermes/worktrees/unit-01'
)
# Record the session_id returned for each worker
```

Repeat for each unit. Record all session_ids:
```
Unit 1 → session_id: abc123
Unit 2 → session_id: def456
```

**Fallback - delegate_task (ONLY if `hermes chat -w` command itself is unavailable or crashes on startup):**

"Worker got stuck" is NOT a reason to use delegate_task - spawn a new `hermes chat -w` worker instead.
Use delegate_task ONLY when the `hermes` command itself is not found or fails to start. Maximum 3 workers at once.
```python
delegate_task(tasks=[
    {"goal": "Worker 1 prompt", "context": "Your worktree is at [ROOT_DIR]/.hermes/worktrees/unit-01. cd into it before doing anything."},
    {"goal": "Worker 2 prompt", "context": "Your worktree is at [ROOT_DIR]/.hermes/worktrees/unit-02. cd into it before doing anything."},
    {"goal": "Worker 3 prompt", "context": "Your worktree is at [ROOT_DIR]/.hermes/worktrees/unit-03. cd into it before doing anything."},
])
```

---

## Phase 3 - Track Progress

**GATE: DO NOT present final summary until status table has been shown with ALL workers in done/failed state.**

### ☐ Step 7 - Monitor workers and collect PR links

**☐ Step 7.1 - Initialize status table**

As soon as workers are spawned, display the initial table:
```
| # | Unit                     | Status  | PR                        |
|---|--------------------------|---------|---------------------------|
| 1 | [unit title]             | running | -                         |
| 2 | [unit title]             | running | -                         |
| 3 | [unit title]             | running | -                         |
```

**☐ Step 7.2 - Poll workers and update table after every poll**

After every `process(action="poll")` or `process(action="wait")`, complete ALL three sub-steps:

**☐ 7.2a - Parse PR url** from the last 20 lines of output:
```python
tail = output.splitlines()[-20:]
pr_url = None
for line in tail:
    if line.strip().startswith("PR:"):
        pr_url = line.strip()[3:].strip()
        break
```

**☐ 7.2b - Update worker status** in todo and table:
```python
if pr_url and pr_url.startswith("https://"):
    todo(action="update", id="...", status="done")
    # update table: running → done, add PR url
elif pr_url:
    todo(action="update", id="...", status="blocked")
    # update table: running → failed, add reason
else:
    todo(action="update", id="...", status="blocked")
```

**☐ 7.2c - Show updated table immediately** - do NOT wait until all workers finish.

**☐ Step 7.3 - Final summary (ONLY when ALL workers are done/failed)**

DO NOT show final summary until every row in the table is `done` or `failed`.

**☐ Step 7.3.1 - Verify all workers ran `/requesting-code-review`:**
```python
# Check each worker log for requesting-code-review
for session_id in all_session_ids:
    log = process(action="log", session_id=session_id)
    if "requesting-code-review" not in log:
        # Mark that worker as missing review in the summary
```
If any worker skipped review - note it explicitly in the final summary.

**☐ Step 7.3.2 - Show final summary:**

```
## Migration Complete

 Successful: X/N units landed as PRs
 Failed: Y/N units

### PRs created:
- [unit 1] → github.com/.../pull/1
- [unit 2] → github.com/.../pull/2

### Failed units (manual action required):
- [unit 3] - reason: [details]
```

**☐ Step 7.3.3 - Spot-check ALL PR diffs for scope violations:**
```bash
# Check files changed in every PR
for pr_number in [list all PR numbers]; do
    echo "=== PR #$pr_number ===" && gh pr diff $pr_number --name-only
done
```
→ Verify each PR only touches files from its assigned unit. If any PR contains unexpected files - note it in the summary as a scope violation.

Then spot-check at least one full diff:
```bash
gh pr diff [PR number] | head -50
```
→ Verify imports changed, functions are async, no unexpected content.

→ **MANDATORY GATE: Use `clarify()` tool now. Do NOT proceed to Phase 4 until user explicitly confirms.**

```python
clarify("""
Migration summary ready. Here are the PRs created and recommended merge order.

[paste final summary table here]

Shall I proceed with merging the PRs in the order above?
""")
```

**Do NOT run any `gh pr merge` commands until user responds YES to the clarify above.**

---

## Phase 4 - Merge PRs in correct order (MANDATORY - do not skip)

### ☐ Step 8 - Merge PRs in the correct order

Merge order matters when PRs share test files. Follow this order:

1. **First - source-only PRs** (no test file assigned) - merge these first. They change source files only and won't conflict with each other.
2. **Then - PRs with test files** - merge these after all source PRs are in main, so test mocks match the already-migrated source files.

Example:
```bash
# Step 1: merge source-only PRs first
gh pr merge <PR for profile.py> --merge
gh pr merge <PR for preferences.py> --merge
gh pr merge <PR for middleware.py> --merge
# ...

# Step 2: merge PRs with test files
gh pr merge <PR for admin.py + test_users.py> --merge
gh pr merge <PR for sms.py + test_notifications.py> --merge
# ...
```

→ **Note:** Source-only PRs may show failing CI checks (because test mocks haven't been updated yet). This is expected — test-file PRs merged last will fix all tests. Override branch protection or merge with admin privileges if needed.

### ☐ Step 9 - Run full test suite on main after ALL PRs are merged

**This step is mandatory.** After merging all PRs, pull main and run the full test suite:

```bash
git checkout main && git pull origin main
[run test command — same as baseline from Step 2]
```

Compare results against the baseline recorded in Step 2:
- Any NEW failures (not in pre-existing list) = regression. Fix directly on main before declaring migration complete.
- Same failures as baseline = expected. Migration is complete.

**☐ If tests fail:** spawn a cleanup worker to fix remaining issues:
```bash
mkdir -p /tmp/hermes-batch
cat > /tmp/hermes-batch/cleanup-prompt.txt << 'WORKER_PROMPT'
You are a cleanup worker. Fix all remaining test failures after batch migration.

## Your task
Run the test suite and fix any failing tests:
```bash
git checkout main && git pull origin main
[run test command]
```

Find remaining references to old pattern:
```bash
grep -rn "<old_pattern>" . --include="*.py"
```

Fix each failing test — update to the new pattern.

## Safety rules
- Work directly on main branch
- Only edit test files — never source files
- Run tests after each fix to verify
- Commit: git add -A && git commit -m "fix: update stale references after batch migration"
- Push: git push origin main

## Report
End with: CLEANUP: done - X issues fixed
WORKER_PROMPT
hermes chat -w --yolo -q "$(cat /tmp/hermes-batch/cleanup-prompt.txt)"
```

→ Do NOT declare migration complete until test count matches or exceeds baseline.

---

## Phase 5 - Cleanup (MANDATORY - do not skip)

### ☐ Step 10 - Cleanup worktrees and branches

**☐ Step 10.1 - Remove worktrees**

```bash
git worktree remove --force $ROOT_DIR/.hermes/worktrees/unit-01
git worktree remove --force $ROOT_DIR/.hermes/worktrees/unit-02
# repeat for each unit

git worktree prune
```

**☐ Step 10.2 - Verify clean**

```bash
git worktree list  # must show only ONE entry - the main worktree
```

→ **STOP. If more than one entry - run `git worktree prune` again before finishing.**

**☐ Step 10.3 - Remove local and remote branches (after PRs are merged)**

```bash
# Remove local branch
git branch -d batch/unit-01
# Remove remote branch
git push origin --delete batch/unit-01
# repeat for each unit
```

---

## Pitfalls

- **Mock namespace safety** — after a source file is migrated (e.g., `import requests` → `import httpx`), mocks in OTHER test files that still reference `patch('src.module.requests.get')` will still work because `patch()` creates the attribute on the module object. Do NOT panic if you see old-style mocks on migrated modules — they will be updated by their owning unit.
- **Sub-worktrees noise** — if a worker runs hermes inside its worktree, it creates a `.worktrees/` directory that gets committed accidentally. This is prevented by adding `.worktrees/` to `.gitignore` in Step 5.1. If it still appears in a PR diff — amend the commit: `git rm -r --cached .worktrees/ && git commit --amend --no-edit && git push --force-with-lease`.
- **Never edit source files** - coordinator is forbidden from editing `.py/.js/.ts` files. All changes go to workers. If a worker gets stuck - spawn a new worker, NOT edit files yourself.
- **Worker gets stuck** - use `process(action="kill", session_id="...")` then spawn a new `hermes chat -w` worker for that unit. Do NOT edit files yourself.
- **Workers must stay in scope** - each worker modifies ONLY its assigned files. If tests fail because other files are not yet migrated - that is expected. Note it in the PR.
- **ROOT_DIR must be absolute** - always run `ROOT_DIR=$(pwd)` after navigating to project root. Never use relative paths.
- **Use `terminal(background=True)` not shell `&`** - shell `&` does not return session_id; you cannot monitor workers.
- **Copy worker prompt template exactly** - fill in [brackets] only. Do not rewrite or remove steps.
- **Workers CAN invoke skills** - `/requesting-code-review` works in worker sessions via slash command.
- **Never visually extract PR URLs** - always parse `output.splitlines()[-20:]` + `line.startswith("PR:")`.
- **No two units touch the same file** - if conflict, merge those units into one.
- **PR already exists** - worker should run `gh pr edit <number>` and report existing URL. This is not a failure.
- **Branch already exists** - use timestamped names: `batch/unit-01-20260401`.
- **Orphaned worktrees** - always run `git worktree prune` after cleanup.
- **Cleanup worker failed to push to main** - if the cleanup worker pushed to a branch instead of main (e.g., `hermes/hermes-XXX`), fetch and fast-forward merge manually:
```bash
git fetch origin hermes/hermes-XXX
git merge origin/hermes/hermes-XXX --ff-only
git push origin main
git push origin --delete hermes/hermes-XXX
```
- **gh not authenticated** - verify `gh auth status` before spawning.
- **Rate limits** - many concurrent API calls may hit provider rate limits; reduce concurrency if needed.
- **No tests exist** - note in plan; workers commit and create PR but cannot verify correctness.
- **Worker modified test file not in its scope** — if spot-check reveals a worker touched a test file not listed in its "Files to modify", close that PR, spawn a cleanup worker to revert the test file changes (`git checkout -- tests/...`), amend the commit, and re-create the PR. Do NOT leave the scope violation — it will cause merge conflicts with the unit that actually owns that test file.
