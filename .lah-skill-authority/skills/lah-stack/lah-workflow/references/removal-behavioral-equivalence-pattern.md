# Removal Behavioral Equivalence Verification Pattern

Use when you need to determine **final candidate truth** after a reversible trial proved structural cleanliness but behavioral equivalence was unproven.

This pattern sits between `reversible-removal-trial-pattern.md` (which proves clean absence) and `explicit-removal-implementation-pattern.md` (which executes permanent removal). Its purpose: resolve every candidate's state from provisional to final.

## When to use

- A repo-hygiene candidate survived a reversible trial (clean absence) but the operator scenario was generic (e.g. `CLI --help`) — not a meaningful exercise of the original capability.
- Predecessor missions returned `REMOVAL_STRUCTURALLY_CLEAN_BUT_BEHAVIOR_UNPROVEN` for any candidate.
- You need to distinguish between `STRUCTURALLY_CLEAN_ABSENCE` and `BEHAVIORALLY_EQUIVALENT_REPLACEMENT`.
- No provisional classifications may remain at mission completion — every candidate must be either fully supported for removal or restored.

## Core distinction

The pattern enforces a rigorous distinction between:

| Outcome | Meaning |
|---------|---------|
| `STRUCTURALLY_CLEAN_ABSENCE` | No broken imports, adapted tests pass — but capability may be lost |
| `BEHAVIORALLY_EQUIVALENT_REPLACEMENT` | A replacement exists AND exercises the original operator capability |
| `INTENTIONALLY_DEPRECATED_CAPABILITY` | Explicit decision to retire + truthful deprecation stub + alternative path |
| `CAPABILITY_LOSS` | No replacement exists, capability disappears when candidate is removed |
| `UNRESOLVED_OPERATOR_RISK` | No test exercises the original behavior — absence is unproven |

## The 10-phase protocol

### Phase 0 — Preservation and intake

Before any mutation:
1. Verify exact source SHA and branch per mission spec
2. Verify required ancestry and forbidden ancestry absence
3. Create an isolated Git worktree (`git worktree add`)
4. Record Git status and exact tree state (file hashes)
5. Verify Candidate 2 (or any designated "keep" file) still exists
6. Verify each removal candidate is absent
7. Verify supporting files (tests, stubs, acceptance tools) exist
8. Record pre-existing test failure count and details
9. Create a preservation receipt JSON
10. Stop `BLOCKED` if the repository differs materially from the authoritative source

### Phase 1 — Capability inventory

For each candidate, reconstruct the **original capability** from the authoritative pre-removal commit:

1. `git show <pre-removal-sha>:<candidate-path>` — full file content
2. Exported functions and their signatures
3. CLI behavior, expected inputs/outputs
4. Side effects (file writes, network calls, subprocesses)
5. Error handling and failure semantics
6. Security or governance role
7. Tests, documentation, command examples
8. Git introduction commit and last meaningful modification
9. All callers, wrappers, dynamic import strings

Produce a **capability contract** for each candidate with:
- Original purpose and operator entrypoints
- Expected semantic behavior
- Known consumers (files that import, exec, or check existence)
- Observable outputs
- Security/governance role
- Current claimed replacement or deprecation
- Evidence gaps

**Pitfall — Do not proceed from file names alone.** Two files with similar names may have completely different architectures (e.g. `hermes-host-bridge/` is NOT a replacement for `hermes-canonical-host-execution.mjs` — different protocol, different execution model, different receipt schema).

### Phase 2 — Canonical replacement discovery

For each removed candidate, search for a real canonical replacement:

1. Repository source files (same repo)
2. Package scripts (`package.json`, `scripts/`)
3. Command dispatchers, registries, shell wrappers
4. Runtime adapters, host CLI tools, policy executors
5. Sibling repositories (check ALL repos under the workspace)
6. Documentation and migration notes
7. Certification artifacts from predecessor missions

Allowed findings:

| Finding | Meaning |
|---------|---------|
| `CANONICAL_REPLACEMENT_CONFIRMED` | Replacement exercises materially relevant original behavior AND is reachable through a real operator path |
| `PARTIAL_REPLACEMENT_FOUND` | Some capabilities covered but not all |
| `NO_REPLACEMENT_FOUND` | No replacement exists |
| `REPLACEMENT_AUTHORITY_UNCLEAR` | A file exists but its role as replacement is unverified |

**Key rule:** A replacement is confirmed only when ALL of: (a) performs the original behavior, (b) reachable through a current operator path, (c) input/output contract understood, (d) failure behavior understood, (e) tests exercise it, (f) not merely a deprecation stub or absence handler.

**Pitfall — Two systems with the same output concept are not necessarily replacements.** The `hermes-host-bridge` and `hermes-canonical-host-execution.mjs` both produce "execution receipts" but use different schemas, different auth (HMAC vs WebSocket token), and different delivery mechanisms (standalone files vs gateway RPC). They are architecturally distinct — one does not replace the other.

### Phase 3 — Deprecation authority

For candidates proposed as intentionally deprecated, verify explicit authority.

A valid intentional deprecation requires evidence of ALL of:
1. An explicit decision to retire the capability (ADR, operator decision record, mission verdict)
2. The reason for retirement
3. The expected user/operator behavior after retirement (which tool to use instead)
4. An alternative path where applicable
5. Truthful CLI messaging or documentation
6. No silent fallback (no missing-module swallowed silently)
7. No known active consumer requiring the old capability
8. Compatibility with current repository governance

Allowed findings:

| Finding | Meaning |
|---------|---------|
| `DEPRECATION_EXPLICITLY_AUTHORIZED` | Written authorization exists (ADR, operator decision) |
| `DEPRECATION_IMPLICIT_BUT_UNSUPPORTED` | Stub exists, alternatives exist, but no written authorization document |
| `DEPRECATION_CONTRADICTED_BY_ACTIVE_USAGE` | Active consumer discovered — deprecation claim is false |
| `DEPRECATION_AUTHORITY_NOT_FOUND` | No evidence of any deprecation decision |

**Pitfall — A stub that says "this feature is deprecated" does not itself prove authorization.** The stub is the technical mechanism; authorization is the governance decision. They are distinct. If only the stub exists, the finding is `DEPRECATION_IMPLICIT_BUT_UNSUPPORTED`.

**Decision rule:** If deprecation authority is absent or unclear, the candidate must not remain removed solely on that basis. Restore it.

### Phase 4 — Candidate-specific behavioral proof

Build a meaningful behavioral scenario for EVERY candidate. The scenario must exercise the original capability or its claimed replacement.

**Forbidden:** Generic CLI `--help` checks, syntax validation, or import-only verification as sufficient proof.

For each candidate, design a scenario that:
1. Exercises the original capability's core function
2. Produces observable output
3. Has verifiable semantic assertions
4. Can be run in both REMOVED and RESTORED states for comparison

**Example scenario types:**
- For a plan generator: run the generator and verify it produces expected plan files
- For a client: run the client with a real endpoint and verify it connects
- For a validation module: import and call exported functions with test fixtures
- For a test: run the original test suite (not the adapted absence test)

**Pitfall — An adapted test that asserts absence is not proof of behavioral equivalence.** A test that only checks `!existsSync(path)` tests repository structure, not capability preservation.

### Phase 5 — Controlled comparison states

For every candidate, construct and evaluate at minimum two states:

1. **CURRENT_REMOVED_STATE** — run the operator scenario with the candidate absent
2. **REFERENCE_RESTORED_STATE** — restore from pre-removal commit, run the scenario again

Where a replacement is claimed, also construct:
3. **REPLACEMENT_EXERCISED_STATE** — exercise the claimed replacement

For every state capture:
- Exact file hashes
- Git diff
- Command run, exit code, stdout, stderr
- Generated artifacts
- Semantic assertion results
- Timeout or execution status

**Compare semantic behavior, not just exit codes.** A scenario that passes in both states may prove tolerance but not equivalence. A scenario that FAILS in the removed state and PASSES in the restored state proves capability loss.

### Phase 6 — Restoration trials

For every candidate not conclusively proven safe to keep removed:
1. Restore from the authoritative pre-removal commit
2. Restore any required supporting files (launchers, tests, acceptance tools)
3. Run the original tests (not adapted absence tests)
4. Run the meaningful operator scenario
5. Compare removed vs restored behavior
6. Determine whether restoration repairs a capability loss

Restoration is the **default safe final action** when:
- Behavioral equivalence is unproven
- Deprecation authority is absent
- A capability disappears when the candidate is absent
- An active consumer is found
- Operator results differ materially between removed and restored states
- Evidence remains ambiguous

### Phase 7 — External and cold-path review

For each candidate proposed to remain removed, inspect:
- All sibling repositories under the workspace
- Shell scripts, deployment manifests, npm scripts
- Command registries and dynamic import paths
- Subprocess command strings
- README files, operator documentation
- Generated plans, receipts, archived workflows
- Git history for aliases and wrappers

Allowed findings:

| Finding | Meaning |
|---------|---------|
| `NO_LOCAL_EXTERNAL_CONSUMER_FOUND` | No references outside the repository |
| `KNOWN_EXTERNAL_CONSUMER_FOUND` | A file in another repo references this candidate |
| `EXTERNAL_CONSUMER_RISK_UNRESOLVED` | Potential external use but unverifiable |

**Pitfall — Do not claim global absence of consumers.** The scan is local — there may be consumers outside the workspace that you cannot see. Use `FOUND` / `NOT_FOUND` scoped to the local scope only.

### Phase 8 — Supporting-file truth

For each supporting file (stubs, adapted tests, acceptance tools), verify it truthfully reflects the final candidate decision.

Forbidden patterns:
- A test that passes only because it no longer exercises the capability (adapted absence test)
- An acceptance script that silently skips missing behavior
- A launcher that claims an alternative exists when it does not (verify the alternative actually works)
- A deprecation message without a valid supported next action
- Deleting a test to hide a failure

If the candidate is RESTORED: restore supporting files to their original state. The launcher should delegate, tests should test, acceptance tools should exercise.

### Phase 9 — Framework and regression validation

Run ALL relevant test suites:
1. Repo-hygiene authority framework tests
2. Candidate-specific focused tests (original tests for restored candidates)
3. CI-equivalent tests (git-policy tests, host launchers, etc.)
4. Syntax validation (`node --check`) for all modified files
5. Import validation for all restored modules
6. Launcher execution test for CLI tools
7. Acceptance-tool execution test

Report pre-existing failures separately — they must NOT be attributed to the mission without differential proof.

**Key rule:** If a test was adapted to verify absence and now fails because the candidate is restored, that's EXPECTED — the test was wrong (it was encoding absence, not testing capability). Restore the original test.

### Phase 10 — Final tree authority

Every candidate must receive a final, non-provisional state:

| Final outcome | When to use |
|---------------|-------------|
| `REMOVAL_CONFIRMED_BEHAVIORALLY_EQUIVALENT` | Replacement proven, all criteria met |
| `REMOVAL_CONFIRMED_INTENTIONAL_DEPRECATION` | Deprecation explicitly authorized, stub truthful, alternative exists |
| `CANDIDATE_RESTORED_DUE_TO_CAPABILITY_LOSS` | Capability disappears without replacement |
| `CANDIDATE_RESTORED_DUE_TO_UNRESOLVED_OPERATOR_RISK` | Risk exists and deprecation authority absent |
| `CANDIDATE_RESTORED_DUE_TO_INSUFFICIENT_EVIDENCE` | No conclusive proof either way |
| `CANDIDATE_RESTORED_DUE_TO_EXTERNAL_CONSUMER` | Active consumer found outside the repo |
| `FINAL_DECISION_BLOCKED` | Cannot make a determination — requires operator judgment |

**No candidate may remain absent under `STRUCTURALLY_CLEAN_BUT_BEHAVIOR_UNPROVEN`.** That's the provisional state that triggered this pattern.

## Required mission artifacts

Create a dedicated directory `docs/repo-hygiene/<mission-name>/` containing:

1. Preservation receipt (Phase 0)
2. Capability contract per candidate (Phase 1)
3. Canonical replacement discovery report (Phase 2)
4. Deprecation authority report (Phase 3)
5. Per-candidate operator scenario results (Phase 4-5)
6. Per-candidate restoration proof (Phase 6)
7. External consumer review (Phase 7)
8. Supporting-file truth report (Phase 8)
9. Final aggregate JSON report with structured semantics
10. Zero-unrelated-mutation proof
11. Integration-readiness report
12. Continuity JSON at canonical path (`docs/mcporter/`)
13. Markdown aggregate report

## Known pitfalls

- **Adapted test = absence test**: A test that was changed from testing functionality to asserting `!existsSync(path)` does NOT prove the capability is preserved. It proves the repository is structurally consistent. These are different facts.
- **Clean absence ≠ behavioral equivalence**: Two different concepts. A module can be absent without broken imports while still representing a lost operator capability.
- **Deprecation stub ≠ authorization**: The stub is the technical mechanism; authorization is a governance decision. Verify both independently.
- **Interdependent candidates**: If C4 imports C5, restoring C4 without C5 will fail. Restore interdependent candidates together or document the dependency chain.
- **Pre-existing failures**: Before this mission, run and count test failures. Document them separately. Any change in failure count must be proven to be mission-caused, not pre-existing.
- **C1 certify dependency**: When `git-policy-certify.mjs` has `existsSync(...)` for a candidate AND the absence changes the verdict from `CERTIFIED` to `BLOCKED`, that is a **material behavioral change** — not a "non-fatal diagnostic."
- **Same output name, different architecture**: Two systems can both produce "execution receipts" but be architecturally incompatible. Compare protocol, auth model, delivery mechanism, schema fields — not just output concept names.
