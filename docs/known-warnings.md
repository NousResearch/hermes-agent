# Known Warnings

## Test suite: Starlette `TestClient` / HTTPX deprecation

- **Observed:** 2026-08-14 during Kanban provider-meter remediation tests.
- **Message:** `Using httpx with starlette.testclient is deprecated; install httpx2 instead.`
- **Impact:** No test failure; the focused suite completed successfully.
- **Follow-up trigger:** Revisit when the warning becomes an error, when the project's Starlette/HTTPX dependency versions are upgraded, or when a test starts failing in `TestClient` setup.
- **Current action:** Record only; do not change dependency versions as part of provider-meter remediation.
