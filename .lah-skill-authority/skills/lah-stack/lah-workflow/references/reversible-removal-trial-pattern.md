# Reversible Removal Trial Pattern

Use for governed, sequential, per-candidate operator trials where the goal is to determine whether a statically-detected unused file, export, or dependency can be safely neutralized without causing operator regressions.

## Core Principle

> A candidate may be statically unused and still be operationally necessary. The real operator outcome is authoritative over static analyzer agreement.

## When to Use

- A Knip/dependency-cruiser/similar static analysis identified unused files/exports
- A reconciliation mission produced a certified shortlist of removal candidates
- The shortlist candidates have `removal_safety: UNPROVEN`, meaning no operator proof exists
- The mission boundary is "evaluate removal hypotheses, not perform cleanup"

## 22-Phase Protocol

| Phase | Name | Gate-pass |
|-------|------|-----------|
| 0 | Baseline & preservation | Exact SHA, clean ancestry, shortlist loaded, Git state captured |
| 1 | Shortlist eligibility | Each candidate satisfies original shortlist rules; no new protective evidence |
| 2 | Per-candidate evidence dossier | JSON with all evidence categories (import, exec, dynamic, config, docs, tests, history) |
| 3 | Historical/intent analysis | Creation commit, renames, associated tests/docs, authorship |
| 4 | Impact model | Parent dir, siblings, CLI surfaces, test suites, config surfaces, validation matrix |
| 5 | Operator scenario definition | Command, expected exit code, semantic assertions, forbidden side effects, timeout |
| 5b | Baseline operator trial | **Run operator scenario BEFORE neutralization** — capture exit code, stdout, stderr, semantic assertions. This enables differential regression detection: the trial can distinguish "scenario was already failing" from "removal caused a regression." Required since framework fix v1. |
| 6 | Baseline run | Pass before neutralization or document pre-existing failure |
| 7 | Reversible neutralization | File rename (`.mjs` → `.mjs.trial-removed`), minimal, isolated, precomputed rollback |
| 8 | Immediate structural validation | Git diff allowlist, syntax, module resolution, zero unrelated mutation (candidate-scoped) |
| 9 | Focused tests | Candidate-specific tests, declaring module tests, parent subsystem tests |
| 10 | Repository-wide regression | All existing hygiene tests, fast structural checks |
| 11 | Real operator trial | Actual command executed, real output captured, semantic assertions evaluated |
| 12 | Cold-path review | Alternate CLI options, error paths, dry-run mode, maintenance mode |
| 13 | Rollback verification | Restore exact content, file modes, symlinks, verify hashes |
| 14 | Decision authority | Rules: any regression = KEEP, unexplained diff = KEEP, no operator scenario = KEEP. **IMPORTANT:** `KEEP_DUE_TO_OPERATOR_REGRESSION` now requires a passing baseline operator trial (Phase 5b). Without it, the framework returns `KEEP_DUE_TO_INSUFFICIENT_EVIDENCE`. |
| 15 | Sequential trial isolation | One candidate at a time, restore before next, no batch neutralization |
| 16 | Per-candidate artifacts | evidence-dossier.json, history-review.md, impact-model.json, baseline-report.json, etc. |
| 17 | Aggregate report | Metrics table: candidate/type/static/focused/full/operator/rollback/decision |
| 18 | Zero-unrelated-mutation proof | Before/after Git status, tracked hashes, file modes, no absorbed user work |
| 19 | Implementation boundary | Only trial framework + artifacts committed; no permanent removal |
| 20 | Trial framework tests | Shortlist loading, eligibility, neutralization, rollback, isolation, decision rules |
| 21 | Diff integrity | Compare against baseline SHA, allowlist, confirm all original subjects present |
| 22 | Certification | One of: CERTIFIED, CERTIFIED_WITH_LIMITATIONS, PARTIAL, BLOCKED |

## Architecture Components

```
reversible-trial.mjs
├── ShortlistLoader          — load & validate certified shortlist
├── CandidateEligibility     — check protective evidence rules
├── EvidenceDossier          — comprehensive evidence per candidate
├── ImpactModel              — blast radius definition
├── BaselineCapture          — capture pre-neutralization state
├── NeutralizationEngine     — rename/restore files
├── ValidationMatrixRunner   — run structural checks
├── FocusedTestRunner        — candidate-specific tests
├── RegressionTestRunner     — repo-wide tests
├── OperatorScenarioRunner   — real operator verifications
├── RollbackVerifier         — verify complete restoration
├── DecisionEngine           — classify candidate outcome
├── SequentialTrialOrchestrator — orchestrate N trials sequentially
└── AggregateReporter        — produce aggregate report
```

## Mandatory Safety Fields

```json
{
  "removal_safety": "UNPROVEN|BOUNDED_TRIAL_PASSED",
  "automatic_action_allowed": false,
  "production_removal_authorized": false,
  "future_explicit_removal_mission_required": true
}
```

`REMOVAL_CANDIDATE_CERTIFIED` means only: bounded reversible trial passed; future explicit removal mission required; production removal not authorized.

## Known Pitfalls

- **Structural validation in dirty worktrees**: When the trial framework itself edits files (CLI, tests, docs), pre-existing changes show up in `git diff`. Scope the zero_unrelated_mutation check to the candidate's mutation surface only — not the entire diff.
- **Renamed file extension issues**: `node --check` on `.mjs.trial-removed` files fails because Node doesn't recognize the extension. Define operator scenarios that verify operator-visible behavior (CLI loading, test execution) rather than trying to parse renamed files.
- **Pre-existing test failures**: If the operator scenario is `node --test` on a suite with pre-existing failures, the candidate will be classified as KEEP_DUE_TO_OPERATOR_REGRESSION. Document pre-existing failures at Phase 6 baseline. **Framework fix applied**: `classifyCandidate()` now checks `baseline_operator_trial.passed` before returning `KEEP_DUE_TO_OPERATOR_REGRESSION`. Without a passing baseline, it returns `KEEP_DUE_TO_INSUFFICIENT_EVIDENCE`.
- **npx subprocess timeout**: `spawnSync('npx', ...)` can hang in Node collectors while the terminal command works instantly. Resolve the binary path from npx cache and invoke via `process.execPath` instead.
- **Operator scenario that tests unrelated code**: If the operator scenario tests a completely different module than the candidate (e.g., testing repo-hygiene when the candidate is a git-policy file), the outcome is meaningless. The scenario MUST exercise the candidate's actual functionality. An unrelated scenario that passes does not prove the candidate is safe to remove.
- **classifyCandidate baseline blind spot**: Before the framework fix (commit ba2dc75), `classifyCandidate()` returned `KEEP_DUE_TO_OPERATOR_REGRESSION` whenever `operator_trial.passed === false`, WITHOUT checking whether:
  - The operator scenario was meaningful for this candidate
  - The scenario passed at baseline before neutralization
  - The failure was truly a regression (difference from baseline)
  
  This caused Candidate 2 (`scripts/git-policy-test.mjs`) to be incorrectly classified. The fix adds Phase 5b (baseline operator trial) and updates `classifyCandidate()` to require differential evidence.
