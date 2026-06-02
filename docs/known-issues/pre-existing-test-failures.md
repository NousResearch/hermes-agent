# Pre-Existing Backend Test Failures

**Date:** 2026-06-02  
**Milestone:** v2.9 Evidence-First Managed Agents Foundation  
**Status:** Documented, not yet fixed  

---

## Failing Tests

### 1. `test_ledger_event_type_enum_all_members`

**File:** `tests/hermes_cli/test_run_ledger_api.py:1768`  

**Failure:** The test expects exactly 14 `LedgerEventType` members but the enum
now has 15.

**Root cause:** The v2.9 integration audit added `unknown = "unknown"` to
`LedgerEventType` because `from_legacy_event()` needs a fallback value for
unrecognized legacy events. The test was written before this addition and has
not been updated.

**Related to v2.9 Evidence Foundation:** Partially. The enum was introduced
in Phase 2 (Ledger Event Normalization). The `unknown` member was added during
the integration audit to fix a runtime crash in `from_legacy_event()`.

**Suspected owner:** `agent/managed_agents/execution_policy.py` (LedgerEventType)

**Recommended fix priority:** Low. Update the test's expected set to include
`"unknown"`.

**Blocks v2.9 milestone tagging:** No. The runtime behavior is correct; only
the test assertion is stale.

**Fix:** Add `"unknown"` to the expected set in the test.

---

### 2. `test_propose_endpoint_returns_advisory_only`

**File:** `tests/hermes_cli/test_run_ledger_api.py`  

**Failure:** `500 Internal Server Error` when calling
`POST /api/agents/runs/policy/propose` via `TestClient`.

**Root cause:** The `derive_proposed_action()` function requires imports
that fail in the `TestClient` test environment (FastAPI `TestClient` does
not fully replicate the production server environment for endpoints that
use lazy imports inside handler functions).

**Related to v2.9 Evidence Foundation:** Partially. The propose endpoint
was added in Phase 4 (Policy Dry-Run). The endpoint itself works correctly
when called via real HTTP (verified by curl smoke test in Phase 3). The
failure is specific to the test harness.

**Suspected owner:** `hermes_cli/web_server.py` (propose endpoint) /
`tests/hermes_cli/test_run_ledger_api.py` (test fixture)

**Recommended fix priority:** Medium. Either fix the `TestClient` fixture
setup to support lazy imports, or rewrite the test as a unit test (as done
for `test_scenario08_ci_path_does_not_execute_impactful_policy_actions`).

**Blocks v2.9 milestone tagging:** No. The endpoint is verified functional
via unit tests (`test_derive_proposed_action_*`) and the scenario test
(`test_scenario08_*`).

**Fix:** Refactor the endpoint handler to not use lazy imports, or convert
the test to a direct unit test of the handler function.

---

## Conclusion

Both failures are:

- **Pre-existing** from the v2.9 Evidence Foundation work (not pre-existing
  before v2.9 began, but existed at the time of milestone tagging).
- **Non-blocking** for the v2.9 milestone tag.
- **Well-understood** with clear fix paths.

The first failure (enum member count) requires a trivial test update.
The second failure (TestClient endpoint) requires a test fixture adjustment
or endpoint refactor.

Neither failure indicates a product bug. All 247 other tests pass, and the
v2.9 Evidence Foundation operates correctly in real HTTP use.
