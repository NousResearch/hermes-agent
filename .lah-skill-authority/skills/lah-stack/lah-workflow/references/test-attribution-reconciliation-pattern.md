# Test Attribution Reconciliation Pattern

## When to use

After any CODE_CHANGE mission where the test suite spans multiple lots, phases, or components and you need to prove exact per-lot test attribution for certification.

Use this when a mission requires:
- "Prove exact test attribution for Lots A through E4"
- "Run per-lot targeted tests and report counts"
- "Reconcile combined count with individual file counts"
- "Explain a test count delta between two commits"

## Pattern

### Step 1 — Discover all test files

```bash
ls -R test/<suite>/
```

Note: files may be in subdirectories like `test/x402/`, `test/x402/transport/`, etc.

### Step 2 — Run each file individually and record its count

```bash
for f in test/<suite>/*.js test/<suite>/**/*.js; do
  count=$(node --test --test-concurrency=1 "$f" 2>&1 | grep "^# tests" | awk '{print $3}')
  echo "$count  $f"
done
```

Store this as a hash: filename → test count.

### Step 3 — Attribute each file to a lot/component

Read the `@fileoverview` or JSDoc header of each test file to determine which lot created it. When ambiguous:
- Check which commit created the file (`git log --diff-filter=A --follow <file>`)
- Check imports — what components does it test?
- Check the file path — submission/ → Lot E2-E3, transport/ → Lot B, etc.

Produce a table:

| Test file | Tests | Lot | Reason |
|-----------|-------|-----|--------|
| `test-models.js` | 16 | A | Canonical model tests |
| `test-header-normalization.js` | 28 | B | Transport parser |
| `test-injectable-lahb-transport-adapter.js` | 98 | E2 | Submission transport |

### Step 4 — Sum by lot

```text
Lot A:  16+32+21+34 = 103
Lot B:  28+38+17   = 83
...
Cross-lot: 23 (test-proof-closure.js)
────────────────────────────────
Total:  587
```

### Step 5 — Verify combined suite

Run the union of all test files:

```bash
node --test --test-concurrency=1 test/<suite>/*.js test/<suite>/**/*.js
```

The combined count must equal the sum of individual file counts. If it differs, check for:
- Files counted twice (glob overlap)
- Files omitted from the glob
- Test files that dynamically generate subtests

### Step 6 — Run combined suite 3 times for stability

Record: Run | Tests | Pass | Fail | Duration

### Step 7 — Run per-lot targeted tests

For each lot, run only its attributed files:

```bash
node --test --test-concurrency=1 test/<suite>/<lot-file-1>.js test/<suite>/<lot-file-2>.js
```

This proves each lot's tests pass independently.

## Cross-Commit Test Delta Analysis

When a mission claims a test count changed between two commits (e.g. "fell from 85 to 65"), run this delta analysis before modifying any file.

### Step A — Create a temporary detached worktree at the old commit

```bash
git worktree add --detach /tmp/<temp-worktree-name> <old-commit-sha>
```

### Step B — Extract test names at both commits

```bash
# Old commit (from temp worktree)
rg -n '^\s*(test|it)\(' /tmp/<temp-worktree-name>/test/<path>/*.js | sed 's/.*test(//' | sed 's/,.*//' | sort > /tmp/old-tests.txt

# New commit (current)
rg -n '^\s*(test|it)\(' test/<path>/*.js | sed 's/.*test(//' | sed 's/,.*//' | sort > /tmp/new-tests.txt
```

### Step C — Compute the set diff

```bash
comm -23 /tmp/old-tests.txt /tmp/new-tests.txt > /tmp/removed-tests.txt
comm -13 /tmp/old-tests.txt /tmp/new-tests.txt > /tmp/added-tests.txt
wc -l /tmp/old-tests.txt /tmp/new-tests.txt /tmp/removed-tests.txt /tmp/added-tests.txt
```

### Step D — Classify every removed test

For each test in `removed-tests.txt`, determine:

| Classification | Meaning |
|----------------|---------|
| `RESTORE` | Legitimate test that was accidentally removed; restore it |
| `RENAMED` | Same assertion moved to a new test name; find and verify the new name |
| `MERGED` | Assertions absorbed into another test; verify all branches covered |
| `INVALID_OLD` | Old test tested now-invalid behavior (e.g. receipt-only authenticity); replace with equivalent negative test |
| `DISCOVERY_BUG` | Test still exists but wasn't discovered (glob missed it, file not committed, etc.) |

### Step E — Build assertion-equivalence matrix

For every test not classified `RESTORE`, prove:

```text
Original test: <name>
Original assertions: <list>
New test(s): <name>
Covered assertions: <list>
Missing assertions: <list>  // must be empty for certification
```

### Step F — Decide restoration

- Default rule: **restore removed tests**
- A removed test may stay removed **only** when ALL are proven:
  - The old test asserted a now-invalid security behavior
  - A replacement test exists with opposite (correct) expectation
  - Every legitimate old assertion is represented
  - The changed behavior is explicitly documented
- Invalid positive tests (e.g. "receipt-only authenticity accepted") must be replaced by explicit negative tests (e.g. "receipt-only authenticity rejected")
- Replacement negative tests count toward the total — they are not a loss

### Step G — Clean up

```bash
git worktree remove /tmp/<temp-worktree-name>
rm -rf /tmp/<temp-worktree-name>
```

### Delta analysis verification checklist

- [ ] Temporary worktree created and verified
- [ ] Test names extracted at both commits
- [ ] Set diff computed (old vs new)
- [ ] Every removed test classified
- [ ] Assertion-equivalence matrix built
- [ ] All legitimate assertions preserved
- [ ] Invalid old tests replaced with negative security tests
- [ ] Temp worktree cleaned up
- [ ] Final count >= original count (unless exact equivalence proven)
- [ ] Combined suite passes × 3

## Verification checklist

- [ ] Every test file is attributed to exactly one lot (or "cross-lot")
- [ ] No file is counted in two lots
- [ ] Sum of individual file counts = combined suite count
- [ ] Combined suite passes × 3 with stable count
- [ ] Per-lot targeted runs all pass
- [ ] No test file omitted from attribution table

## Pitfalls

| Trap | Fix |
|------|-----|
| Test filename appears in multiple globs | Use explicit file list, not glob expansion for attribution |
| Test file in subdirectory (`test/x402/submission/`) missed by flat glob | Use `test/x402/*.js test/x402/**/*.js` or explicit paths |
| Dynamic test generation changes count between runs | Run 3 times and confirm stable count |
| `grep "^# tests"` shows suite-level count instead of file-level | Run each file individually, not the whole suite |
| Old commit uses different test file structure | Check `git show --name-status` for file addition/removal in the diff |
| `rg` unavailable | Use `grep -rn '^\s*test('` instead |
| Test names with dynamic content (`test('case ' + n)`) | Count by `test(` occurrence, not by unique name |
