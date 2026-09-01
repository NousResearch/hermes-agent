# Task 1 Report: Artifact Oracle and Typed Failure Vocabulary

## Files changed

- `tests/fakes/mcp_oauth_peer.py`
- `tests/tools/test_mcp_oauth_reauth_regression.py`
- `.superpowers/sdd/2026-09-01-mcp-oauth-chunk-0-implementation-plan/task-1-report.md`

The pre-existing unrelated change to `contributors/emails/agent@Agents-Mac-mini.local` was preserved and not staged.

## TDD evidence

- RED: `HERMES_PYTHON=/private/tmp/hermes-mcp-oauth-chunk-0-venv/bin/python scripts/run_tests.sh tests/tools/test_mcp_oauth_reauth_regression.py -q` failed during collection with `ModuleNotFoundError: No module named 'tests.fakes.mcp_oauth_peer'`.
- GREEN: the same command passed with `4 tests passed, 0 failed`.
- Review RED: after adding the repr safety test, the same command failed 1 test because the dataclass repr exposed `OLD_ACCESS_TOKEN_FOR_TEST_ONLY`.
- Review GREEN: after adding the label-only `__repr__`, the full Task 1 test file passed with `5 tests passed, 0 failed`.

## Implementation

Added the typed OAuth failure vocabulary, immutable artifact state snapshot, safe label-only summaries, deterministic legacy OAuth fixture seeding/capture through `HermesTokenStorage`, and known-mutation classification.

The metadata fixtures identify the legacy server via `old-auth.invalid` rather than an `OLD_` token, so the labeler recognizes that fixture marker while never exposing its payload in `safe_summary()`.

## Concerns

None. The test runner emitted no test failures in the green run; generated runner caches were not staged.
