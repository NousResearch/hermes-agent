# Explicit Removal Implementation Pattern

Use after a successful reversible removal trial has certified candidates as `REMOVAL_CANDIDATE_CERTIFIED`. This pattern governs the permanent removal of those files from the repository with bounded risk, per-candidate validation, and truthful failure handling.

## When to Use

- A hygiene reversible trial produced `REMOVAL_CANDIDATE_CERTIFIED` outcomes for one or more candidates
- The mission explicitly authorizes **permanent file removal** (no neutralization, no rollback target)
- You are not responsible for fixing pre-existing test failures, only for proving your removals didn't cause new ones

## Phase Structure

### Phase A (optional) — Truth Repair

Before implementing removals, verify that every candidate's certified outcome is semantically correct. This is especially important when:

- Pre-existing test failures may have caused false `KEEP_DUE_TO_OPERATOR_REGRESSION` outcomes
- The operator scenario was inappropriate for the candidate (tested unrelated code)
- The framework's `classifyCandidate()` is simplistic (checks `!operator_trial.passed` without baseline comparison)

**Verify for each retained candidate:**
1. Was the operator scenario **meaningful** for this candidate? (Does it test the candidate's actual functionality?)
2. Did the operator scenario have **pre-existing failures** that prevented differential measurement?
3. Is the outcome classification semantically correct? (`KEEP_DUE_TO_OPERATOR_REGRESSION` requires differential evidence — pre-existing failures invalidate it)

**Repair actions:**
- Create a correction report explaining exactly why the outcome changed
- Repair the aggregate JSON
- Repair the operator packet
- Repair the continuity JSON
- Add correction record
- Commit Phase A separately from Phase B

### Phase B — Per-Candidate Removal

Process removals **sequentially, one candidate at a time**. Do not batch remove.

For each candidate:

#### 1. Build Evidence Dossier
- File path, size, hash, mode
- All references: imports, dynamic imports, `require` calls, subprocess execution, shell wrappers, npm scripts, configuration files, registries, CLI delegation, tests, documentation
- Git history: introduction commit, last modification, callers, renames, purpose

#### 2. External Consumer and Cold-Path Review
Search all known consumers:
- Local sibling repositories under the workspace
- Shell history or documented commands (when safely available)
- README and operator documentation
- Package scripts
- CLI dispatch tables
- Registries
- Subprocess command strings
- Dynamically constructed paths

Classify each:
- `NO_LOCAL_EXTERNAL_CONSUMER_FOUND`
- `EXTERNAL_CONSUMER_RISK_UNRESOLVED` (must block removal)
- `KNOWN_EXTERNAL_CONSUMER_FOUND`

An unresolved material external-consumer risk **must block** that candidate's permanent removal.

#### 3. Define Impact Model
- Parent directory and sibling modules
- Tests that reference this file
- CLI commands that dispatch to it
- Config registries
- Documentation references that become stale
- Re-export chains

#### 4. Design a Candidate-Specific Operator Scenario

**WARNING: Generic CLI-help scenarios are INSUFFICIENT.** A scenario like `node tools/repo-hygiene-authority/cli.mjs --help` only proves the CLI binary loads — it does not prove anything about an individual candidate's removal. A corrective mission explicitly identified this as a certification deficiency.

Each scenario MUST:
- Exercise the candidate's actual functionality (not an unrelated module)
- Produce different results when the candidate is present vs absent
- Verify semantic outputs: expected structure, commands, errors, generated artifacts, or replacement behavior
- Be specific enough that removing the wrong file would cause the test to detect a meaningful failure

**Good examples:**
- Running a certification/audit script that checks for the candidate's existence via `existsSync` (the check flips PASS→FAIL deterministically)
- Running a launcher that delegates to the candidate (output changes from help text to deprecation)
- Running an acceptance tool that imports the candidate (reports presence vs absence)
- Running the candidate's own tests with and without the candidate present

**Bad example (do NOT use):**
- `node tools/repo-hygiene-authority/cli.mjs --help` — generic CLI loading, unrelated to most candidates

#### 5. Differential Validation Matrix (REQUIRED)

Before removal, design a 4-state differential validation. Execute all 4 states in sequence:

**State A — REFERENCE (candidate present):**
- Restore candidate from base SHA: `git checkout <base_sha> -- <candidate_path>`
- Record file hash: `sha256sum <candidate_path>`
- Run the operator scenario — capture exit code, stdout, stderr, semantic assertions
- Run `node --check` on the candidate (syntax validation)

**State B — REMOVED (candidate absent):**
- Remove candidate: `git rm <candidate_path>`
- Run the SAME operator scenario — same capture
- Result MUST be meaningfully different from State A (different exit code, different output, etc.)

**State C — ROLLBACK (candidate restored):**
- Executed restoration: `git checkout <base_sha> -- <candidate_path>` (NOT theoretical)
- Verify hash matches State A exactly
- Run operator scenario — result MUST match State A

**State D — FINAL (candidate removed permanently):**
- Remove candidate again: `git rm <candidate_path>`
- Run operator scenario — result MUST match State B

**Interpretation rules:**
- A baseline that already fails cannot certify removal (State A must pass)
- Exit code alone is insufficient — capture semantic assertions
- Timeout is not a pass
- An unexplained difference between State A and State C means rollback failed
- State C must match State A byte-for-byte (hash, output, behavior)

#### 6. Remove Candidate + Dependent Files

Use `git rm <file>` to stage deletion. Remove not only the candidate but also:
- Launcher scripts that hardcode the candidate's path and become invalid

**CRITICAL: Non-candidate files require independent authorization.** Do NOT remove these files merely because they referenced the removed candidate:

- Test files that import the candidate — RESTORE and ADAPT to verify absence instead
- Acceptance/diagnostic scripts that import the candidate — RESTORE and ADAPT instead
- Wrapper/launcher scripts — RESTORE with deprecation stub or adapt

**Process for each non-candidate file:**
1. Restore from base SHA
2. Inspect complete content and role
3. Determine valid independent role after authorized removal
4. Choose disposition:
   - `RESTORED_UNCHANGED` — file has valid independent role
   - `RESTORED_AND_ADAPTED` — modify to test absence, handle missing import, or report deprecation
   - `REPLACED_WITH_TRUTHFUL_ABSENCE_STUB` — replacement that reports removal and points to alternatives
   - `REMOVAL_INDEPENDENTLY_VALIDATED` — only if same proof rigor as authorized candidates

**Rule of thumb:** Tests and acceptance tools require a strong presumption of restoration. Do not delete a test merely because the implementation it covered was removed. Prefer adapting tests to assert absence, deprecation, replacement behavior, or repository integrity.

#### 7. Structural and Syntax Checks
Run `node --check` on all remaining files that were potentially affected (importers, siblings, CLI scripts).

#### 8. Run Focused Tests
Run the hygiene reversible-trial focused tests:
```
node --test --test-concurrency=1 tools/repo-hygiene-authority/tests/reversible-trial.test.mjs
```
Expect 35/35 PASS (or current baseline).

#### 9. Run Broader Regressions
Run all repo-hygiene tests. Pre-existing failures must be clearly distinguished from new regressions:
```
node --test --test-concurrency=1 tools/repo-hygiene-authority/tests/repo-hygiene.test.mjs
```

Do not classify a pre-existing failure as removal-caused without differential evidence.

#### 10. Run Operator Scenario After Removal
Same scenario as from the differential matrix (State D). Compare against State B.

#### 11. Commit
One commit per successfully removed candidate. Commit message should include:
- Mission tag (e.g. `LAH_REPO_HYGIENE_EXPLICIT_REMOVAL_V1`)
- Candidate name
- Outcome: `REMOVAL_IMPLEMENTED_AND_VALIDATED`
- Key metrics: operator scenario result, test results, pre-existing failure status
- List of non-candidate files adapted/restored (if any)

## Scope-Authority Review (Required Before Certification)

Before declaring a mission CERTIFIED, audit for these common over-certification patterns:

| Pattern | Symptom | Fix |
|---------|---------|-----|
| Scope expansion | Non-candidate files removed without independent review | Restore and evaluate each individually |
| Generic operator proof | CLI --help used for all candidates | Replace with candidate-specific scenarios |
| Theoretical rollback | "Git can restore from base SHA" claimed as proof | Execute actual restoration and hash verification |
| Untracked artifacts | hygiene-report-*.json or similar left in worktree | Clean before committing evidence |
| Missing baseline | classifyCandidate returns OPERATOR_REGRESSION without checking pre-existing failure | Use the framework fix (Phase 5b baseline operator trial) |

## Pitfalls

### Commit scope contamination from git rm
`git rm <file>` automatically stages the deletion. If you also have Phase A (truth repair) files staged, they get committed together. **Fix:** Use `git reset --soft HEAD~1` to unstage, then re-stage Phase A files separately, commit Phase A, then stage and commit the deletion.

### Dependent files that span multiple candidates
A launcher script (e.g. `bin/hermes-canonical`) may reference both Candidate A and Candidate B. When you remove Candidate A's launcher, Candidate B's reference in that launcher is also removed. This is fine — it simplifies Candidate B's removal later. But be aware of cross-candidate dependency chains that could affect removal ordering.

### Operator scenario that tests unrelated code
If the previous trial's operator scenario tested a completely different module (e.g. testing repo-hygiene when the candidate is git-policy), the outcome classification may be semantically incorrect. Verify scenario relevance before accepting certified outcomes.

### Syntax checks on renamed .trial-removed files
During the reversible trial phase, `node --check` on `.mjs.trial-removed` files fails because Node doesn't recognize the extension. During explicit removal, this is not an issue since the file is actually deleted.

### Subagent paths may contaminate canonical repo
When using `delegate_task` for differential validation, the subagent may operate on the canonical repo path instead of the isolated worktree. Always verify the subagent's `cwd` matches the worktree path, and restore the canonical repo after subagent completion: `git checkout -- <files>`.

### Rollback hash must match byte-for-byte
A restored file may have different metadata (mode bits, timestamps) but identical content. Use `sha256sum` for content verification, not file metadata. If the hash matches, the content is byte-identical.

## Allowed Outcomes

| Outcome | Meaning |
|---------|---------|
| `REMOVAL_IMPLEMENTED_AND_VALIDATED` | File removed, all validations passed |
| `REMOVAL_REJECTED_OPERATOR_REGRESSION` | Before/after operator behavior changed (requires passing baseline) |
| `REMOVAL_REJECTED_STATIC_REGRESSION` | Syntax or module resolution broke |
| `REMOVAL_REJECTED_EXTERNAL_CONSUMER` | External reference found |
| `REMOVAL_REJECTED_INSUFFICIENT_EVIDENCE` | Could not prove safety |
| `REMOVAL_REJECTED_ROLLBACK_FAILURE` | Executed rollback failed or hash mismatch |
| `REMOVAL_BLOCKED` | Precondition failure |
| `REMOVAL_REVALIDATED` | Used in corrective missions — removal re-proven with proper evidence |

## Authorized Removal Outcomes (Corrective Missions)

When auditing a previous mission's removals, use these outcomes:

| Outcome | Meaning |
|---------|---------|
| `REMOVAL_REVALIDATED` | Removal re-proven with candidate-specific scenario, executed rollback, differential matrix |
| `REMOVAL_REJECTED_OPERATOR_PROOF` | Operator scenario was insufficient to prove safety |
| `REMOVAL_REJECTED_EXTERNAL_CONSUMER` | External reference found during review |
| `REPAIR_BLOCKED` | Precondition prevents corrective action |

## Non-Candidate File Dispositions

| Disposition | Meaning |
|-------------|---------|
| `RESTORED_UNCHANGED` | Restored from base SHA without modification |
| `RESTORED_AND_ADAPTED` | Restored and modified to handle absence, test repository state, or report deprecation |
| `REPLACED_WITH_TRUTHFUL_ABSENCE_STUB` | Replaced with script that reports removal and points to alternative tools |
| `REMOVAL_INDEPENDENTLY_VALIDATED` | Removed after independent proof (same rigor as authorized candidates) |

## Rollback Boundaries

For every removed candidate, rollback must be **executed**, not merely theoretically possible:

1. Restore file from base SHA: `git checkout <base_sha> -- <candidate_path>`
2. Verify hash matches: `sha256sum <candidate_path>` (must match pre-removal hash)
3. Verify git diff: `git diff -- <candidate_path>` (should be clean)
4. Rerun the operator scenario from State C of the differential matrix
5. Confirm restored behavior matches State A

Report rollback as:
- `ROLLBACK_VERIFIED` — all steps pass
- `ROLLBACK_FAILED` — hash mismatch or behavior differs
- `ROLLBACK_AMBIGUOUS` — partial recovery, some assertions pass
