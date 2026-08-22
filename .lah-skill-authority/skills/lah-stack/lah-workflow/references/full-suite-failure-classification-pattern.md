# Full-suite failure classification (lah-openclaw-mvp)

## Why this pattern exists

A bounded CODE_CHANGE in openclaw-runtime/lah-openclaw-mvp can NEVER be certified
on "full suite passes" alone:

- The repository-wide suite carries a large PRE-EXISTING failure floor
  (~104 FAIL on main HEAD, 9351 tests). These are unrelated to any mission diff.
- A bare `node --test` from lah-openclaw-mvp root HANGS: it scans `releases/`
  (2000+ archived release test files) and `test/fixtures/runner-stdin-echo.mjs`,
  a stdin-blocking fixture. Use the explicit glob below; kill any hung run with
  `pkill -f "node --test"` and `pkill -f runner-stdin-echo`.
- The workspace clone (/home/deploy/openclaw-runtime) and canonical checkout
  (/home/deploy/lah-stack-repos/openclaw-runtime) are SEPARATE clones that
  legitimately diverge in UNTRACKED local test files (501 vs 503 files →
  9334 vs 9351 tests). Count deltas are explained by file-set divergence,
  NOT by the diff.

## Exact commands

Baseline (canonical checkout, same HEAD, WITHOUT the diff):
```bash
cd /home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp
node --test --test-concurrency=1 test/*.test.js test/*.test.mjs > /tmp/bl-full.txt 2>&1; echo "EXIT:$?"
grep -E '^# (tests|pass|fail)' /tmp/bl-full.txt
```

With-diff (workspace clone, feature branch): same command in
/home/deploy/openclaw-runtime/lah-openclaw-mvp → /tmp/wd-full.txt.

Capture failing identities (compare names, not totals):
```bash
grep '^not ok' /tmp/bl-full.txt | sed 's/^not ok [0-9]* - //' | sort -u > /tmp/bl-names.txt
grep '^not ok' /tmp/wd-full.txt | sed 's/^not ok [0-9]* - //' | sort -u > /tmp/wd-names.txt
comm -12 /tmp/wd-names.txt /tmp/bl-names.txt   # in both
comm -23 /tmp/wd-names.txt /tmp/bl-names.txt   # implementation-only
comm -13 /tmp/wd-names.txt /tmp/bl-names.txt   # baseline-only
```

Explain total deltas by comparing the test FILE SETS:
```bash
ls test/*.test.js test/*.test.mjs | sort > /tmp/wd-files.txt   # (in each clone)
comm -23 /tmp/wd-files.txt /tmp/bl-files.txt   # files only in workspace
comm -13 /tmp/wd-files.txt /tmp/bl-files.txt   # files only in baseline
```

## Classification buckets

| Bucket | Meaning |
|---|---|
| PRE_EXISTING_IDENTICAL | failing in both runs — not the mission's concern |
| PRE_EXISTING_VARIANT | same failure, slightly different message |
| MISSION_REGRESSION | fails ONLY with the diff AND causally linked |
| ENVIRONMENTAL_OR_FLAKY | untracked local files absent from baseline, shared data/ state, REAL network tests, port/ordering flake |
| UNRESOLVED | needs investigation |

## Causality proof for implementation-only failures

1. **Imports grep**: `grep -l "<each-modified-module>" <failing-test-file>` — if no
   failing file imports any modified module, no causal path exists.
2. **Same-commit clone comparison**: run the failing file INDIVIDUALLY in both
   clones at the same commit. If canonical passes while workspace fails at
   identical committed content, the cause is local state (data/, untracked
   files), not the diff.
3. **UNTRACKED files cannot be regressions**: a test file that exists only in
   the workspace (untracked, absent from baseline) was never executed by the
   baseline — its failure is an artifact, by construction.
4. **Bound the server/app diff**: when server.js or a shared entrypoint is in the
   diff, verify the change scope (e.g. only 2 route handlers' logging) — tests
   that mount the app via createApp() still pass individually.

## Test-run report pollution (commit-scope trap)

Running the suite regenerates TRACKED `test/reports/*.json` (timestamps change).
They appear in `git diff` as unrelated modifications. Before committing:

```bash
git checkout HEAD -- test/reports/   # restore all regenerated report files
git diff --stat HEAD                 # verify only mission files remain
```

## Rules

- Repair ONLY proven MISSION_REGRESSION. Never repair pre-existing failures as
  part of the mission.
- Do NOT rerun the 9000+ suite after classification unless a repair touched
  shared infrastructure.
- Keep EXOCLICK_LIVE_ENABLED=false and perform zero external mutation during
  classification runs.
