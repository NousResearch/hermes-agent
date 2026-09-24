# Desktop MCP OAuth integration checks

Run from a checkout with the Python MCP dependencies and Desktop Node dependencies installed.
Outputs are local receipts; keep them outside the checkout. Neither harness uses real provider credentials.

```bash
python3 evals/desktop_mcp_oauth/backend_http_fixture.py --repo . --output /tmp/mcp-backend.json
node evals/desktop_mcp_oauth/renderer_lifecycle.mjs "$PWD" /tmp/mcp-renderer.json approved
node evals/desktop_mcp_oauth/renderer_lifecycle.mjs "$PWD" /tmp/mcp-scope.json cancel
node evals/desktop_mcp_oauth/renderer_lifecycle.mjs "$PWD" /tmp/mcp-unmount.json unmount
```

The backend harness creates disposable HOME/HERMES_HOME directories and a local HTTP OAuth/MCP provider. It exercises production session functions, discovery, dynamic registration, PKCE code exchange, token persistence and fresh-process authenticated MCP access. Wrong state, callback replay, wrong server, wrong profile and cancellation are negative controls. It removes temporary token stores and reports only booleans/paths, never token values.

The renderer harness bundles the real McpTab and its dependencies in Chromium, runs the production native loopback listener with a registration-only Electron IPC adapter, and uses a fixture WebSocket backend. `cancel` changes the component's scope; `unmount` removes it entirely. Both must cancel the original backend session and close its native listener without relaying a callback. `approved` must use the native relay, not REST auth. Set `CHROMIUM_EXECUTABLE` to use an existing Chromium executable; otherwise Playwright's installed browser is used.

These are complementary integration probes, not a single end-to-end Electron/provider test. The renderer backend and Electron registration boundary are fixtures; the Python probe uses production session functions rather than gateway RPC transport. No native Electron application launch or hosted-provider consent is claimed.

For A/B, run the same renderer harness against a baseline checkout by replacing its first positional argument. The approved assertion fails before the shared relay change. For the lifecycle regression, a checkout immediately before the scoped-tab cleanup fails both abandonment assertions. Backend cross-profile callbacks must be rejected even when the session ID and valid state are supplied; the original owner must still be able to finish afterward.
