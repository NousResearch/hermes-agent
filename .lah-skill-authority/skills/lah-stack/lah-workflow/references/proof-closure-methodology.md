# Proof-Closure Methodology

A **proof-closure mission** is a bounded independent verification that a prior deliverable is safe, deterministic, complete, and durably recorded. It is NOT a re-implementation — its goal is to certify, not to redesign.

## When to run

- After a CODE_CHANGE mission, before closing it out.
- When a prior report makes claims that have not been independently verified.
- When remaining "known gaps" are documented but unfixed.
- Before handoff to a downstream consumer (operator, certification, PR merge).

## Core principles

| Principle | Meaning |
|-----------|---------|
| **Verify first** | Never trust a prior report. Reproduce every claim with direct commands before changing anything. |
| **Isolate** | Use clean worktrees, detached HEADs, or temp checkouts for baseline comparison. Never assume the current workspace state is authoritative. |
| **Fix bounded** | Only repair documented gaps. Do not expand scope or begin the next planned mission. |
| **TDD for corrections** | Every behavioral fix starts with a failing test, then implementation. |
| **No assumptions** | "X is fine because nothing modified it" is not valid — prove it by running the same test at baseline + current HEAD. |
| **Stability before commit** | Run the full targeted suite at least 3 times. Flaky tests are not acceptable for certification. |
| **Post-commit re-verify** | Tests must pass from the committed HEAD, not just the dirty worktree. |

## Standard 5-phase flow

### Phase 1 — Baseline capture

Establish the ground truth before any correction:

```
git merge-base --is-ancestor <base> HEAD  # ancestry check
git status --porcelain=v1                  # clean worktree required
git rev-parse HEAD
```

If a claim about pre-existing failures is made (e.g. "dotenv regression is pre-existing"), create a detached worktree at the reference commit and run the same command:

```bash
git worktree add --detach /tmp/baseline <base_sha>
cd /tmp/baseline && <same_command>
# Compare stdout, stderr, exit code
```

Classification: `BASELINE_EQUIVALENT_FAILURE` | `LOT_A_INTRODUCED_FAILURE` | `ENVIRONMENT_DIFFERENCE`.

### Phase 2 — Gap inventory

Re-read the actual files, not the report. Count:

```
exact files
insertions
deletions
line counts
SHA-256 per file
```

Compare against the prior report. Correct any discrepancies.

For each claimed gap, reproduce it with a direct test or code inspection. Do not trust narrative descriptions.

### Phase 3 — Bounded correction

For each confirmed gap:

1. Write a failing test that proves the gap exists
2. Implement the minimal fix
3. Verify the test now passes
4. Check backward compatibility (existing tests still pass)

Maximum 3 repair cycles per proof-closure mission. If more are needed, the original deliverable was not ready for certification.

### Phase 4 — Multi-regression verification

After all corrections:

```
# Targeted suite × 3
for i in 1 2 3; do node --test --test-concurrency=1 <targeted tests>; done

# Relevant existing tests
node --test --test-concurrency=1 <related tests not modified>

# Baseline equivalence (re-run same baseline command as Phase 1)
# Confirm classification hasn't changed
```

### Phase 5 — Commit and certify

```
git add <exact files>               # never git add .
git commit -m "fix(<scope>): <summary>"
git status --porcelain=v1            # must be empty
git merge-base --is-ancestor <base> HEAD   # ancestry preserved
node --test --test-concurrency=1 <all targeted tests>   # from HEAD
```

## Common pitfalls

| Trap | Symptom | Fix |
|------|---------|-----|
| **Claiming equivalence without evidence** | "The dotenv failure is pre-existing" with no detached baseline comparison | Create a temp worktree at the base commit and run the EXACT same command. Compare exit code + error message. |
| **Trusting the report** | Prior report says "44 files" but `git show --stat` says 43 | Always count from Git, not from narrative. Correct errors openly. |
| **Stability assumed** | Tests pass once → "certified" | Run 3×. Acceptable variance is ±1 test, but 0 failures. |
| **Post-commit not re-run** | Tests pass in dirty worktree, fail in CI | Always run from `HEAD` after commit. `git stash && node --test` is sufficient. |
| **Scope creep** | "Since I'm here, let me also fix X" | Stop. Create a separate mission for anything outside the documented gaps. A proof-closure mission that exceeds 3 repair cycles was launched on the wrong deliverable. |

## Output contract

A certified proof-closure produces:

```
Verdict: <MISSION>_CERTIFIED | _CERTIFIED_WITH_LIMITATIONS | _BLOCKED
Report sections:
  1. Initial git truth (verified)
  2. Gap inventory (verified)
  3. Each gap: test → fix → verify
  4. Targeted test results (3×)
  5. Relevant regression (baseline-compared)
  6. Safety proofs
  7. Commit proof (SHA, parent, message, files)
  8. Post-commit verification
  9. Explicitly not done
```

The boundary between CERTIFIED and CERTIFIED_WITH_LIMITATIONS:

| Condition | Verdict |
|-----------|---------|
| All gaps closed, all tests pass, baseline equivalent | CERTIFIED |
| All security-critical gaps closed, tests pass, but a non-critical env/tooling limitation remains | CERTIFIED_WITH_LIMITATIONS |
| Any gap unresolved, or new regression introduced, or scope broken | BLOCKED |
