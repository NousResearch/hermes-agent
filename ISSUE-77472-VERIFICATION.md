# Issue #77472 verification — 2026-09-05

Base: `upstream/main` at `377118af86`. Existing integration work was preserved,
merged with current main, and migrated to the current defining modules.

## Scope and evidence

- Private artifact creation: trajectory/MoA, CLI/Desktop save, batch rows and
  combined output, A2A, DB-divert fallback, spawn-tree snapshots/index. New files
  are 0600 (0660 managed); new internal directories 0700 (0770 managed).
- Legacy implicit append files: owned single-link files tightened before bytes
  are appended; explicit output permissions retained. Symlink/foreign ownership
  cases log warnings rather than modifying another file's permission policy.
- Optional session JSON snapshots are rewritten privately, including legacy files.
- Default trajectories and single sample exports live under profile-specific
  trajectory buckets, not CWD; explicit trajectory filenames remain authoritative.
- Request diagnostics, outbound tool arguments and Kanban metadata use structured
  fail-closed redaction. Credential containers retain sensitivity through nested
  dictionaries/lists. Full-fidelity conversation/replay/training paths stay intact.
- Request dumps use bounded configurable retention, protect the current dump,
  and serialize publication/pruning. Pending shutdown recovery preserves managed
  directory access without dropping messages on chmod reconciliation failures.

## Verification

`scripts/run_tests.sh` across the 20 directly affected/neighbor suites: **911
passed, 0 failed, 11 skipped**, including 640 TUI gateway tests, real SQLite and
filesystem tests, redaction, retention, replay/cache and symlink regressions.
New legacy/explicit-output, nested-secret, sibling-artifact and single-sample
tests were observed failing before their respective fixes.

Desktop build and `npm run typecheck`: passed. Desktop lint: zero errors, 124
existing warnings. Changed Python files: ruff passed. Windows-footgun check
passed. Contributor email audit passed. TUI subprocess-stdin check passed with
an isolated empty HERMES_HOME. Plugin compatibility scan passed for tracked
files; the unfiltered scan reports four hits in pre-existing untracked ` 2.py`
copies, which were preserved untouched.

Real Electron: `npx playwright test e2e/save-command.spec.ts --reporter=list`
passed. The test launches built Electron and real Python backend against a
local mock inference provider, sends a message, receives a reply, submits
`/save` via Send, verifies saved JSON content and 0600 file/0700 directory.
It precreates the directory at 0755 to exercise upgrade behavior. Artifacts:
`apps/desktop/test-results/save-command--save-command-71a7e-ely-with-0700-dir-0600-file/`
(save-confirmed.png, trace.zip; transcript attached to the test report).
The LLM is mocked; Electron, gateway, RPC, command dispatch and disk writes are real.

## Remaining verification/authority boundaries

- Native Windows ACL enforcement and Linux multi-UID/setgid execution were not
  run on this macOS host. chmod is not a Windows confidentiality mechanism.
- This is targeted regression coverage, not a claim that the entire repository
  test suite or remote CI passed. The PR has not been updated or merged.
- Existing CWD artifacts are not retroactively moved/deleted. Arbitrary unknown
  secrets cannot be guaranteed recognized by pattern-based redaction.
- Main checkout and pre-existing untracked duplicate files remain untouched.
