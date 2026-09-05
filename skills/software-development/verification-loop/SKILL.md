---
name: verification-loop
description: 'Comprehensive verification system: build, types, lint, tests, security, diff review. Use after completing features, before PRs, or after refactoring.'
version: 1.0.0
author: Hermes Agent (adapted from ECC)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [verification, quality, testing, PR, build, lint]
    related_skills: [systematic-debugging, test-driven-development, gstack-review]
---

# Verification Loop

A comprehensive verification system for Hermes Agent sessions.

## When to Use

Invoke this skill:

- After completing a feature or significant code change
- Before creating a PR
- When you want to ensure quality gates pass
- After refactoring
- Before merging to main

## Verification Phases

### Phase 1: Build Verification

Check if the project builds. Use the project's build command:

```bash
# Python projects
python -m build 2>&1 | tail -20

# Node.js projects
npm run build 2>&1 | tail -20
# OR
pnpm build 2>&1 | tail -20
```

If build fails, STOP and fix before continuing.

### Phase 2: Type Check

```bash
# TypeScript projects
npx --no-install tsc --noEmit 2>&1 | head -30

# Python projects (if using mypy/pyright)
python -m mypy . 2>&1 | head -30
```

Report all type errors. Fix critical ones before continuing.

### Phase 3: Lint Check

```bash
# JavaScript/TypeScript
npm run lint 2>&1 | head -30

# Python
ruff check . 2>&1 | head -30
```

### Phase 4: Test Suite

Run the project's test suite. For Hermes Agent specifically:

```bash
# Use the hermetic test runner
scripts/run_tests.sh 2>&1 | tail -50
```

For other projects:

```bash
npm run test 2>&1 | tail -50
# OR
pytest -x -q 2>&1 | tail -50
```

Report:

- Total tests: X
- Passed: X
- Failed: X
- Coverage: X% (if available)

### Phase 5: Security Scan

```bash
# Check for secrets in code
grep -rn "sk-" --include="*.py" --include="*.ts" --include="*.js" . 2>/dev/null | grep -v node_modules | grep -v .git | head -10
grep -rn "api_key" --include="*.py" --include="*.ts" --include="*.js" . 2>/dev/null | grep -v node_modules | grep -v .git | head -10

# Check for leftover debug statements
grep -rn "console.log" --include="*.ts" --include="*.tsx" src/ 2>/dev/null | head -10
grep -rn "print(" --include="*.py" . 2>/dev/null | grep -v "__init__" | grep -v "logger" | head -10
```

### Phase 6: Diff Review

```bash
# Show what changed
git diff --stat
git diff HEAD~1 --name-only
```

Review each changed file for:

- Unintended changes
- Missing error handling
- Potential edge cases
- Files that shouldn't have changed

### Phase 7: Hermes-Specific Checks

For changes to the Hermes Agent codebase:

```bash
# Check config version consistency
grep "_config_version" hermes_cli/config.py

# Verify no hardcoded ~/.hermes paths (use get_hermes_home())
grep -rn "\.hermes" --include="*.py" . 2>/dev/null | grep -v "get_hermes_home" | grep -v "display_hermes_home" | grep -v test_ | grep -v __pycache__ | head -10

# Check for change-detector tests (should use invariants, not snapshots)
grep -rn "assert.*==" tests/ --include="*.py" | grep -i "model\|provider\|version" | head -10
```

## Output Format

After running all phases, produce a verification report:

```
VERIFICATION REPORT
==================

Build:     [PASS/FAIL]
Types:     [PASS/FAIL] (X errors)
Lint:      [PASS/FAIL] (X warnings)
Tests:     [PASS/FAIL] (X/Y passed, Z% coverage)
Security:  [PASS/FAIL] (X issues)
Diff:      [X files changed]

Overall:   [READY/NOT READY] for PR

Issues to Fix:
1. ...
2. ...
```

## Continuous Mode

For long sessions, run verification at checkpoints:

- After completing each function
- After finishing a component
- Before moving to next task
- Every 15 minutes for autonomous agents

## Hermes Integration

- Use `terminal` tool for all build/test/lint commands
- Use `search_files` for security scans (grep patterns)
- Use `read_file` to review changed files
- Use `delegate_task` to run phases in parallel when possible
- For Hermes Agent development, always use `scripts/run_tests.sh` (never raw `pytest`)

## Pre-PR Checklist

Before creating a PR, verify ALL of these:

- [ ] Build passes
- [ ] Type check passes (0 errors)
- [ ] Lint passes (0 warnings)
- [ ] All tests pass
- [ ] No secrets in code
- [ ] No debug statements left in
- [ ] Diff reviewed for unintended changes
- [ ] New code has tests
- [ ] Documentation updated if API changed
- [ ] No hardcoded `~/.hermes` paths (use `get_hermes_home()`)
- [ ] No change-detector tests added
