# Full-suite test traps & bounded certification (lah-openclaw-mvp)

Observed during CLOE_EXOCLICK_OURDREAM_DEVICE_SUBID_WIRING_REPAIR_V1 (2026-08-01).

## Trap 1 — bare `node --test` hangs forever

In `/home/deploy/openclaw-runtime/lah-openclaw-mvp`, do NOT run `node --test` without a path.

- The default glob scans `releases/` — thousands of archived duplicate test files (2395 *.test.* at the time of observation).
- It ALSO scans `test/fixtures/` — the committed fixture `runner-stdin-echo.mjs` calls `process.stdin.resume()` and echoes; it never exits, so the whole suite hangs indefinitely on it.
- Symptom: `node --test --test-concurrency=1` runs for 12+ minutes stuck on a subprocess `runner-stdin-echo.mjs`, output file stays empty.

Fix — always use explicit globs:
```bash
cd lah-openclaw-mvp
node --test --test-concurrency=1 test/*.test.js test/*.test.mjs
```
(501 active test files at HEAD 9958e41: 479 *.test.js + 22 *.test.mjs.)

## Trap 2 — test runs regenerate tracked `test/reports/*.json`

The full suite writes/updates tracked report files under `test/reports/` (e.g. `cloe-governed-mission-proposals.json`, `lah-runtime-evidence.json`, `cloe-cognitive-answer-quality-*.json`). `git diff HEAD` then shows these as modified — they are test-run churn, NOT mission changes.

Fix before commit:
```bash
git checkout HEAD -- test/reports/
# then re-check git diff --stat HEAD — should show only your mission files
```
Do not `git add .` — stage mission files one by one (Gate 8 staging ciblé).

## Bounded certification basis (operator directive)

When the repo-wide suite has many pre-existing failures, the operator may direct a BOUNDED certification instead of exhaustive per-failure classification. The directive was: "Stop the exhaustive full-suite failure classification... certify on the focused/related/runtime basis... do not spend additional iterations on unrelated full-suite failures."

Certify on:
1. Focused tests PASS (the mission's new test file)
2. Directly-related suites PASS (draft, conversation, adapter, entrypoint, executor, etc.)
3. Runtime integration proof PASS (captured evidence, not narrative)
4. Invariants preserved: paused status, LIVE gate unchanged, zero remote mutation, no secret leak
5. Bounded diff (mission files only)

Then do ONE identity comparison between baseline and with-diff, and document the limitation:
```bash
# extract failing test NAMES from both runs
node --test --test-concurrency=1 test/*.test.js test/*.test.mjs 2>&1 | grep "^not ok" \
  | sed 's/^not ok [0-9]* - //' | sort > /tmp/withdiff-names.txt
# same command on the canonical checkout (baseline, without your diff) → /tmp/baseline-names.txt
comm -23 /tmp/withdiff-names.txt /tmp/baseline-names.txt   # only-in-with-diff = potential regressions
# verify 0 hits touch your modified path (grep for your module names)
```
If 0 only-in-with-diff tests touch the modified path, the limitation is: repo-wide failures are pre-existing/unrelated, not exhaustively classified, all tests in the modified execution path pass. Verdict suffix becomes `_WITH_LIMITATIONS`. Do NOT claim `CERTIFIED` (no suffix) while repo-wide failures are unclassified.

Observed numbers at HEAD 9958e41 (openclaw-runtime): baseline 104 FAIL / 9351 tests; with-diff 123 FAIL / 9334 tests. Differences were non-path tests (identity query, telegram REAL, auth routes) — environmental/flaky.

## Baseline capture note

Run the baseline full suite from a **fresh worktree at the exact origin/main SHA**,
NOT from the canonical checkout. The canonical checkout is usually dirty (100+ files)
and its local `main` can be behind `origin/main` — a baseline captured there is
contaminated and compares the wrong tree.

**Pitfall — the WITH-DIFF worktree must also have `npm ci` before the comparison.**
A mission worktree created with `git worktree add` has NO node_modules. Running the full
suite there first produces a garbage comparison: imports fail across hundreds of files,
test counts collapse (e.g. 7217 vs 10386 on the baseline), and the failure count balloons
(260 vs 92). The `comm -23` name comparison then means nothing. Always:
1. `npm ci --no-audit --no-fund` in BOTH worktrees before running the suite;
2. sanity-check the totals match within ±mission-tests: `# tests` should be
   baseline + your new tests, not drastically fewer.

```bash
# in the canonical checkout:
git worktree add /tmp/verify-baseline <ORIGIN_MAIN_SHA>
cd /tmp/verify-baseline/lah-openclaw-mvp
npm ci --no-audit --no-fund
node --test --test-concurrency=1 "test/*.test.js" "test/*.test.mjs" 2>&1 | grep "^not ok" | sort > /tmp/baseline-names.txt
# same globs on your worktree → /tmp/withdiff-names.txt
comm -23 /tmp/withdiff-names.txt /tmp/baseline-names.txt   # only-in-with-diff = regression candidates
```

Compare by **test NAME** (`not ok N - <name>`), not by TAP number — TAP indices shift
when your mission adds test files. Verify each only-in-with-diff candidate: run it
ISOLATED on the baseline worktree (it may fail there too → pre-existing) and trace
whether any of your modified modules is in its execution path.

Clean up: `git worktree remove --force /tmp/verify-baseline` (npm ci leaves untracked
node_modules; `--force` still removes the worktree metadata).

Note: `echo EXIT:$?` after a pipe captures the LAST command (grep), not the runner —
use `set -o pipefail` or capture the runner exit separately if you need it.

## Trap 3 — flaky-by-state-accumulation misattributed as regression

A full-suite "new" failure may be a test that pollutes a **shared store between runs**.
Observed: `campaign-memory-operator-scenarios.test.js` "Store a failed creative" writes
via `writeCampaignMemory` to the default store `data/memory-events/` (no
`memoryEventsDir` override despite the file header claiming temp dirs). Run 1 passes
with a clean store; subsequent isolated reruns fail with `duplicate_skipped` instead
of `written` because the previous run left an identical record.

Attribution protocol:
1. Run the candidate test ISOLATED on the baseline worktree → if it fails there too, it is pre-existing, not your regression.
2. Clean the shared store (`rm -rf data/memory-events/` — test artifacts YOUR runs created, not committed files) and re-run → passes with clean state.
3. Conclusion: flaky-by-state-accumulation, unrelated to the mission diff. Document the proof rather than "fixing" the unrelated test.

Also: full-suite runs create untracked `data/` artifacts (memory-events/, decision-records/, creative-assets/, cloe-governed-action-packets.json). Clean them before commit and never stage them — the worktree's initial `git status` should be your 9 mission files only.

## Trap 4 — giant test file OOMs the 8GB VPS even with --max-old-space-size

Observed 2026-08-08 (CLOE_LAHB_AUTONOMOUS_AFFILIATE_RUNTIME_E2E_V1, LOT 1 fix):
`test/readonly-operator-cli-client.test.js` (~1300 lines importing the whole
conversation pipeline) crashes with `FATAL ERROR: Ineffective mark-compacts near
heap limit` on this VPS, both with default heap and with
`NODE_OPTIONS="--max-old-space-size=4096"`. The run takes ~4 minutes then OOMs.

Key facts:
- The OOM is PRE-EXISTING, not caused by your patch: verify by running the same
  file on the mission worktree WITHOUT your change (same crash) or by running the
  full suite (where it fails among the documented pre-existing files).
- Because the file can never run here, DO NOT add your new test case to it — the
  test is dead code you can never see pass.
- Fix: write a NEW focused standalone test file (`test/<capability>-path.test.js`)
  that imports the same production module and exercises ONLY the changed path.
  It runs in <1s and actually verifies your fix.
- Watch for cross-file env interference when the full suite runs several of these
  focused files together: one file's test that DELETES env vars (fail-closed test)
  can race with another file's setup in the same node --test process. If a focused
  file passes alone but fails in batch, check `secret_accessed present:false` logs
  — that is env contamination, not a code regression. Keeping setup/teardown
  symmetric (see secret-masking-file-corruption.md env-assignment variant) avoids
  most of it; when it still bites, document the batch-vs-solo difference rather
  than chasing the race.
