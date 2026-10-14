# Native pool retirement integration

From `apps/desktop`, after installing the repository-pinned Node dependencies and a
Hermes Python environment (including the serve dependencies):

```sh
HERMES_TEST_REAL_SERVE=1 \
HERMES_TEST_PYTHON=/absolute/path/to/venv/bin/python \
HERMES_TEST_ELECTRON=/absolute/path/to/native/electron \
node ../../node_modules/vitest/vitest.mjs run \
  --config electron/pool-retirement-live-fixture/vitest.config.ts
```

`HERMES_TEST_ELECTRON` is optional when Electron resolves from the desktop/root
Node installation. Point it at the native executable, not `electron/cli.js`.
The Python interpreter supplies dependencies; imports use this checkout's source.
The test is opt-in and POSIX-only (real process-group signals), not a Windows proof.
No full Electron app or Vite renderer build is required. Esbuild bundles only the
fixture entry points and their production imports into a fresh temporary directory.

## What it proves

- Native Electron's browser/main process imports production `pool-retire.ts`,
  `pool-spawn-coordinator.ts`, `pool-retire-http.ts`, and `backend-child.ts`.
- Three actual isolated Python children run headless `start_server(..., port=0)`.
  The oldest is busy with a real scheduler-dispatched, `no_agent=True` cron script:
  no fake running ledger, inference responses, or provider/model calls.
- Real session-token-authenticated HTTP covers busy refusal, unauthenticated refusal,
  prepare/cancel/reprepare, invalid-token commit refusal, commit recovery, and the
  committed fence's refusal to cancel. No retirement response is mocked.
- At cap three, a direct foreground waiter plus an already-queued background ticket
  promoted to foreground both obtain live successors. Renderer activity flags are
  false and timestamps recent: backend-only cron work still vetoes retirement.
- A fixture lifespan barrier holds each real idle child alive after SIGTERM.
  Neither its lease nor successor capacity is released during that barrier.
  The event receipt asserts commit → park → signal → held shutdown → actual exit →
  lease release → acquisition → successor spawn, and OS liveness is false at release.
- A hidden, sandboxed native renderer imports the real `src/store/gateway.ts` and
  `HermesGateway`. It opens four actual WebSockets (legacy and registry-local scopes
  for both idle children). Main sends retirement over fixture IPC before signalling.
  Both scopes stay parked after socket closure and ordinary/forced wake sweeps;
  no new descriptor request is made. Renderer Node integration is disabled.
- The cron process and script remain alive and its heartbeat advances after both
  retirements. Releasing the script completes the real job and restores HTTP idle.
- Cleanup waits for every owned backend exit and verifies the coordinator has zero
  live leases. The temporary HOME, per-child HERMES_HOME, Electron userData,
  sessionData, config/cache, random auth tokens, and OS-assigned ports are isolated.
  The launch environment is allowlisted; no real credentials are copied.

The JSON stdout receipt includes actual PIDs, ephemeral ports, HTTP outcomes, event
ordering, maximum live child count, renderer parking, and cleanup status. Failed
runs additionally report child output. The launcher captures the process handle
before exit. Keeping a `window-all-closed` listener is essential: otherwise native
Electron may exit while asynchronous cleanup is still writing the receipt.

This is a live seam integration test, **not** a full `main.ts` startup/onboarding
or application-window E2E. Descriptor/retirement IPC and shutdown pacing are fixture
wiring. The retirement arbiter, HTTP protocol, OS child exit, scheduler, renderer
parking, and WebSocket connections are real production implementations.
