# Gateway timeout log cleanup and timestamped stderr

**Date:** 2026-05-02
**Author:** Hermes Agent
**Branch:** fix/gateway-timeout-log-timestamps-20260502030014
**PR:** https://github.com/NousResearch/hermes-agent/pull/18648

## Goal

Reduce noisy Hermes gateway log output from expected session_search provider timeouts, and add application timestamps to gateway stderr log lines so Docker logs are easier to correlate.

## What Was Done

- Added plan/spec: `.hermes/plans/2026-05-02-gateway-timeout-log-timestamps.md`.
- Updated `tools/session_search_tool.py`:
  - Added `_is_timeout_exception()` to structurally recognize built-in, asyncio, OpenAI, httpx, and httpcore timeout exception shapes without importing optional providers.
  - Converted expected per-session summarization timeouts into concise warnings without traceback spam.
  - Converted outer `_run_async` summarization bridge timeout logging into a concise warning without `exc_info=True` while preserving the existing JSON error response.
- Updated `gateway/run.py`:
  - Added `%(asctime)s` with `datefmt="%H:%M:%S"` to the gateway verbosity stderr handler, preserving `RedactingFormatter` and the existing level/name/message fields.
- Updated tests:
  - `tests/tools/test_session_search.py` now covers timeout logging, timeout classification, and outer bridge timeout logging.
  - `tests/gateway/test_runner_startup_failures.py` now verifies gateway stderr verbosity logs include an HH:MM:SS timestamp.

## Key Decisions

- No ADR: this is an operational logging/triage refinement, not an architectural or API contract change.
- For known provider/network timeouts inside `_summarize_session()`, return after a concise warning rather than retrying and eventually printing a large traceback. This intentionally reduces gateway stalls/log noise for expected timeout failures; unexpected exceptions still keep the existing retry and final traceback path.
- Used a structural timeout classifier so Hermes does not need hard imports of every optional provider client just to classify expected timeout failures.
- Kept timestamps in the application formatter rather than changing Docker logging configuration, because the user specifically wanted the timestamp next to gateway log lines and this works regardless of runtime logging driver.

## Validation

- `python -m pytest tests/tools/test_session_search.py::TestSummarizeSessionTimeouts::test_timeout_failure_logs_concise_warning_without_traceback tests/gateway/test_runner_startup_failures.py::test_start_gateway_stderr_verbosity_includes_timestamp -q` — RED observed before implementation; failed for missing concise timeout handling/timestamped gateway stderr. Log: `.hermes/test-output/2026-05-02-red-session-search-gateway-stderr.log`.
- `python -m pytest tests/tools/test_session_search.py::TestSummarizeSessionTimeouts::test_outer_summarization_timeout_logs_concise_warning_without_traceback -q` — RED observed for outer bridge timeout traceback. Log: `.hermes/test-output/2026-05-02-red-outer-timeout.log`.
- `python -m pytest tests/tools/test_session_search.py::TestSummarizeSessionTimeouts tests/gateway/test_runner_startup_failures.py::test_start_gateway_stderr_verbosity_includes_timestamp -q` — passed, 4 tests. Log: `.hermes/test-output/2026-05-02-green-timeout-expanded.log`.
- `python -m pytest tests/tools/test_session_search.py tests/gateway/test_runner_startup_failures.py tests/test_hermes_logging.py -q` — passed, 99 tests. Log: `.hermes/test-output/2026-05-02-regression-targeted-suites-final.log`.
- `python ${HERMES_HOME:-/work/.hermes-data/hermes-agent}/skills/software-development/requesting-code-review/scripts/static_scan_diff.py --unstaged` — passed, no findings.
- `python -m py_compile tools/session_search_tool.py gateway/run.py tests/tools/test_session_search.py tests/gateway/test_runner_startup_failures.py` — passed.
- `python -m ruff check gateway/run.py tools/session_search_tool.py tests/tools/test_session_search.py tests/gateway/test_runner_startup_failures.py` — failed on pre-existing broad lint issues in `gateway/run.py`/tests plus one pre-existing unused import; no new lint issue was introduced by the edited hunks.
- Independent review via `delegate_task` — passed, no blocking security or logic issues; reviewer noted non-blocking suggestions around retry intent and additional real-provider classifier tests. Retry intent was documented in the plan/devlog.
- `python -m pytest -q` — attempted broad full suite; did not complete cleanly in this environment. It reached ~90%, emitted many xdist `node down: Not properly terminated` failures, then stopped making progress and was killed to unblock PR prep. Partial log: `.hermes/test-output/2026-05-02-full-pytest.log`.
- `HERMES_TDD_EVIDENCE="RED .hermes/test-output/2026-05-02-red-session-search-gateway-stderr.log; RED .hermes/test-output/2026-05-02-red-outer-timeout.log; GREEN .hermes/test-output/2026-05-02-green-timeout-expanded.log; REGRESSION .hermes/test-output/2026-05-02-regression-targeted-suites-final.log" /work/.hermes-data/scripts/code_work_guard.py --mode final` — passed.

## What Skills and Tools Were Used

- Skills: `hermes-agent`, `writing-plans`, `test-driven-development`, `systematic-debugging`, `requesting-code-review`, `devlog`, `github-pr-workflow`.
- Tools: worktree-based git workflow, pytest, static diff scan, py_compile, independent reviewer subagent.

## Artifacts Updated

- `.hermes/plans/2026-05-02-gateway-timeout-log-timestamps.md`
- `.devlogs/2026-05-02-gateway-timeout-log-timestamps.md`
- `.hermes/test-output/` validation logs

## Related Repos

- `/work/.hermes-data/hermes-agent-gateway-timeout-logs-20260502030014`

## Issues & Blockers

- `ruff check` on the selected files reports pre-existing lint violations in large legacy modules/tests; treated as non-blocking because targeted syntax, tests, static scan, and independent review passed.
- No blocker remains before PR once final guard, commit, push, PR creation, and CI check are complete.

## Key Learnings

- The gateway verbosity stderr handler used a non-timestamped formatter even though file logs and verbose logging elsewhere already use timestamps.
- `session_search` has two timeout surfaces: per-session provider/client timeouts inside `_summarize_session()` and an outer `_run_async` bridge timeout in `session_search()`.

## Next Steps

- Final code-work guard passed with RED/GREEN evidence paths.
- Commit, push, open PR, then update this devlog with PR URL and CI/review status.

## Prompting Notes

- **Initial ask:** User pasted Hermes gateway timeout stack traces and asked to improve/fix them and add timestamps next to gateway log lines while following full dev process through PR.
- **Clarifications needed:** None; scope was clear enough to proceed.
- **Corrections made:** Independent review identified an outer timeout log path still using `exc_info=True`; added a RED regression and fixed it.
- **Scope drift:** Small, directly related expansion to cover the outer `session_search()` bridge timeout path.

## Session Quality

- **Faithfulness:** Stayed on track — followed worktree/spec/TDD/review/devlog workflow and kept changes focused on logging/timeout behavior.
- **Prompt patterns:** Strong operational evidence in the pasted log excerpt made the problem concrete; requested process requirements were explicit.

---
*Generated by Hermes Agent — devlog skill*
