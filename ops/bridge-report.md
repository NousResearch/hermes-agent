# Bridge report: t_151153cc

## Root cause

The three failures came from GNU/Linux command assumptions in one real-binary integration test:

- Apple `sort` builds the compression command through a shell, so a compressor token beginning with `-` cannot be executed directly. Its compressed merge also hangs with the original 10,000-record workload; a bounded probe showed that 300 records still invokes the compressor and exits cleanly.
- BSD `script` uses `script -q <file> <command> [args]`, not util-linux `script -qec <command> <file>`.
- Apple `man` supports `-P` but rejects GNU `--pager`.

## Change

The Darwin path now uses 300 bulk records, asserts the original `sort --compress-program -payload-marker` command shape before substituting the marker's absolute execution path, invokes BSD `script` with its native argument order, and maps unsupported Apple `man --pager` to equivalent `-P`. Every non-Darwin branch retains the original 10,000-record fixture and util-linux command.

No test was deleted, hollowed, or newly skipped. The sole `ag` skip is unchanged because `ag` is not installed on this host.

## Validation

- Baseline on Darwin: 3 failed, 156 passed, 1 skipped.
- Focused real-binary test: 5 passed, 1 unchanged skip.
- Full scoped file: 159 passed, 1 unchanged skip.
- Ruff and `git diff --check`: passed.
- Independent Pi validation initially failed on the mistaken assumption that the adapters were mocked. After correction that real Apple binaries execute and only unsupported syntax is translated, Pi returned `VERDICT_PASS`. A minimal Claude cross-harness policy check also returned `VERDICT_PASS`; longer Claude review calls were silent. DeepSeek validation was unavailable because its configured credential was rejected.
- The requested `venv/bin/python` does not exist in this worktree. Verification used the repository-documented shared fallback `/Users/henru1/.hermes/hermes-agent/venv/bin/python`.

## Better-design option not implemented

A future test refactor could separate command-shape assertions from per-platform real-binary execution adapters and use a shared PTY helper instead of `script`. That is broader than the approved single-file Darwin repair and was not implemented.

**Result** done: the scoped test file exits 0 on Darwin with 159 passed and the same one missing-`ag` skip.
**Changed** tests/tools/test_execution_flag_detection.py:3-87; ops/bridge-report.md:1
**Verified** Light tier: `/Users/henru1/.hermes/hermes-agent/venv/bin/python -m pytest tests/tools/test_execution_flag_detection.py` -> 159 passed, 1 unchanged skip; `/Users/henru1/.hermes/hermes-agent/venv/bin/python -m ruff check tests/tools/test_execution_flag_detection.py` -> passed; `git diff --check` -> passed; independent Pi re-review -> PASS; minimal Claude policy check -> PASS.
**Risk** Linux was not executed locally; unchanged Linux behavior is established by inspection of the explicit Darwin-only branches. The exact requested `venv/bin/python` path remains unavailable, and DeepSeek validation could not authenticate.
