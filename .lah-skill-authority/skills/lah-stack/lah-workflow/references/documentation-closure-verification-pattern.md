# Documentation Closure Verification Pattern

## When to use

After a CODE_CHANGE mission's implementation commit (Gate 8), before Memory Lock (Gate 10), when the mission spec requires proving that:

1. The design document accurately matches committed source
2. All tests pass with stable counts
3. Test attribution across multi-lot test suites is exact
4. No secrets or stale evidence exist in documentation
5. The post-commit worktree is clean (own-files = 0)
6. Regressions are ruled out via combined runs × 3 + dotenv baseline + relevant regression suite

## Sequence

### Phase 1 — Git truth verification

```bash
git rev-parse HEAD          # implementation commit
git rev-parse HEAD^         # parent (should be the certified baseline)
git merge-base --is-ancestor <BASELINE_SHA> HEAD   # YES
git status --porcelain=v1   # exactly N untracked = design doc only
git ls-files --others --exclude-standard   # the untracked doc(s)
```

Confirm: HEAD = expected implementation SHA, parent = expected baseline.

### Phase 2 — Implementation inventory

```bash
git show --name-status --format= <IMPLEMENTATION_SHA>
```

Classify every file in the commit:
- `E4_SOURCE` — production code
- `E4_TEST` — test code
- `E4_DOCUMENTATION` — design plans
- `OTHER` — dependencies, configs (should be empty for bounded missions)

### Phase 3 — Document inspection

Read the design document. Check for:
- Correct base commit and implementation commit SHAs
- No secrets (grep `LAHB_ADMIN_API_KEY`, `x-admin-api-key`, `token`, `privateKey`, `mnemonic`, `wallet`, `raw response`)
- No stale or approximate values
- No generated logs

### Phase 4 — Document-to-source consistency

Compare every claim in the document against the source code:

| Claim | Source symbol | Match? |
|-------|---------------|--------|
| Eligible classifications | `checkReconciliationEligibility()` | YES/NO |
| Request schema fields | `buildX402LahbReconciliationRequest()` | YES/NO |
| Identifier selection order | `selectLookupIdentifier()` | YES/NO |
| Status mappings | `classifyRemoteStatus()` | YES/NO |
| Security properties | grep for credentials/network | YES/NO |

Fix documentation to match source — never source to match documentation.

### Phase 5 — Document-to-test consistency

Map every security claim to a test:

| Claim | Test name | Covered? |
|-------|-----------|----------|
| eligible UNKNOWN_OUTCOME | `test('eligibility: SUBMISSION_ATTEMPT_UNKNOWN_OUTCOME...')` | YES |
| no retry | `test('no-resubmission: automaticRetryAllowed always false')` | YES |

Remove or qualify unsupported claims.

### Phase 6 — Test attribution discovery

Run every test file individually, record count, attribute to lot:

```bash
for f in test/<suite>/*.js test/<suite>/**/*.js; do
  count=$(node --test --test-concurrency=1 "$f" 2>&1 | grep "^# tests" | awk '{print $3}')
  echo "$count  $f"
done
```

Then sum by lot attribution. Verify:
- Lot sum + cross-lot = combined suite total
- No file counted twice
- No file omitted

### Phase 7 — Combined suite × 3

```bash
for i in 1 2 3; do
  node --test --test-concurrency=1 test/<suite>/*.js test/<suite>/**/*.js 2>&1 | grep -E "^# tests|^# pass|^# fail"
done
```

Produce table: Run | Tests | Pass | Fail | Duration

### Phase 8 — Relevant regression

Run side-effect-free tests for the subsystem affected:
- Submission tests for LAHB-related changes
- Transport tests for parser changes
- No live network, no receipt claims, no submission

### Phase 9 — Dotenv baseline equivalence

```bash
node --input-type=module -e "try { await import('./src/server.js'); console.log('FULL_PASS'); } catch(e) { console.log('BASELINE_EQUIVALENT_FAILURE: ' + e.message.split('\n')[0]); }"
```

Classification:
- `BASELINE_EQUIVALENT_FAILURE` — same failure as baseline (e.g. missing dotenv)
- `FULL_PASS` — both baseline and new commit import successfully
- `LOT_E4_INTRODUCED_FAILURE` — new commit fails differently

### Phase 10 — Static quality + safety audit

```bash
node --check <each source file>
git diff --check
# Safety audit
grep -rn "process\.env\|fetch\|http\.request\|submitApprovalRaw\|LAHB_URL\|LAHB_ADMIN_API_KEY\|wallet\|signer\|payment" <source-dir>/
```

### Phase 11 — Document correction and commit

Correction policy: fix the document, never the source or tests.
Stage only the document:

```bash
git add -- <document-path>
git commit -m "docs(<scope>): close <mission> documentation"
```

### Phase 12 — Post-commit recheck

```bash
node --test --test-concurrency=1 test/<suite>/*.js test/<suite>/**/*.js
```

Counts must be identical to pre-commit — documentation must not affect runtime.

### Phase 13 — Final Git truth

```bash
git rev-parse HEAD        # = closure commit
git rev-parse HEAD^       # = implementation commit
git status --porcelain=v1 # = empty (0 staged, 0 unstaged, 0 untracked)
```

### Phase 14 — Requirement-to-evidence matrix

Every mission requirement gets a row: file or command, evidence line, result (COVERED/PARTIALLY_COVERED/MISSING/NOT_APPLICABLE), coverage status.

## Pitfalls

| Trap | Fix |
|------|-----|
| Document contains approximate test counts | Run exact per-file attribution before writing final test table |
| Document includes generated logs or stale duration values | Replace with exact values from latest runs |
| Document embeds its own future commit SHA | Document should reference the implementation commit SHA, not the closure commit |
| Worktree shows canonical-checkout dirt | Compare `git diff <base> HEAD --name-only` for E4-owned changes only |
| Security claim says "76 tests" but count changed | Verify exact test count before writing the document; use "79 tests covering [topics]" format |
