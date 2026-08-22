# Worktree Dependency and Baseline Pitfalls

Discovered during CLOE_PR633_FINAL_REGRESSION_AND_IDENTITY_PROOF_V1.

## npm ci required after worktree checkout before test execution

**Symptom:** Agent enters an isolated worktree, attempts to run tests, and gets `ERR_MODULE_NOT_FOUND` for `dotenv`, `zod`, `express` etc. The worktree has no `node_modules/` — `npm ci` was never run. The agent wastes time diagnosing dependency failures instead of real test regressions.

**Root cause:** Git worktrees share the canonical checkout's `.git` directory but NOT its `node_modules/`. When the worktree is created fresh, dependencies must be installed from the lockfile.

**Fix:** After `cd` into the worktree (or before any test run), run `npm ci` to restore dependencies from the lockfile. Verify with `npm ls --depth=0`. Do NOT diagnose missing-package errors as test regressions — they are always an environment setup gap.

Note: `npm ci` respects the lockfile exactly and never installs undeclared packages. Playwright or other undeclared tools remain absent by design — do not install them to force tests to pass.

## Full CI sweep vs bounded baseline confusion

**Symptom:** Agent runs `node --test --test-concurrency=1 test/*.test.js` expecting to reproduce a "218 tests pass" baseline, but gets 8,884 tests with 87 failures. The agent wastes 5+ minutes on a full sweep and draws incorrect conclusions about the baseline.

**Root cause:** The openclaw-runtime repo has a massive test suite (~8,800+ tests across 400+ files). The previously certified "218-test baseline" is a specific bounded subset (campaign, Draw Things, creative suites), not the full `test/*.test.js` directory sweep. The full sweep includes 87 pre-existing env-related failures (missing API keys, provider config, services not running) that have nothing to do with the PR.

**Fix:** Always reconstruct the bounded baseline from the merge commit's CI run, not from the full directory sweep:

```bash
# Find exact test files certified at merge time
git diff --name-only <merge_sha>~1..<merge_sha> -- 'test/*.test*'

# Run only those files
node --test --test-concurrency=1 <test_file1> <test_file2> ...
```

For the openclaw-runtime repo, the bounded campaign/Draw Things baseline is:

```bash
node --test --test-concurrency=1 \
  test/campaign-creation-*.test.js \
  test/canonical-intent-classifier.test.js \
  test/creative-import-bridge.test.js \
  test/creative-factory-orchestrator.test.js \
  test/creative-factory-draw-things-job-adapter.test.js \
  test/draw-things-*.test.js
```

This produces ~296-333 tests depending on the PR stage, all passing when the PR is clean.

The full CI sweep (`test/*.test.js`) is reserved for scheduled full-repo validation on `main`, not for PR-level regression verification.
