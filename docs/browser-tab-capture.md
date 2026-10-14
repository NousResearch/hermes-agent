# Browser Harness exact tab ownership and cleanup

The opt-in capture prerequisite has been extended to native lifecycle cleanup,
explicit owner-scoped release, exact current-page handoff, daemon startup
creation attribution, and bounded recovery.

See [Browser Automation](../website/docs/user-guide/features/browser.md#exact-tab-ownership-and-automatic-cleanup-opt-in-linux-cdp) for configuration,
ownership authority, retention, holds, disable behavior and supported backends.

Only literal `browser.tab_cleanup_enabled: true` enables the feature; it defaults
to false. Unknown owners remain retained, never inferred from final answers or
idle time. Ambiguous IPC responses remain quarantined. Cleanup never stops a
shared relay or closes a snapshot-derived tab.

## Verification

Run the canonical wrapper serially (`-j 1`) over `tests/tools/test_browser_tab*.py`,
`tests/tools/test_browser_use_ipc_timeout.py`, and existing Browser Use, vault,
supervisor, delegation and gateway startup regression suites. Real Harness checks
use disposable unauthenticated Chromium and temporary HOME/HERMES_HOME, never
existing browser profiles. The live browser lane is opt-in; see the test fixture
for its entrypoint and prerequisites.
