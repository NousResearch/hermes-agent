# HTTP Route Runtime Proof Pattern

Prove a production HTTP route exercises its full canonical chain through real
network requests on ephemeral ports, without external side effects.

## Simple Express Test Harness (createApp + ephemeral port)

For most route-level proof missions, the pattern is: import `createApp()` from
`server.js`, bind to `127.0.0.1:0`, and issue real HTTP requests with Node's
built-in `http` module. No Express mock needed — the real middleware chain runs.

### buildServer helper

```javascript
// Returns { url(pathFn), close() }
function buildServer() {
  const { createApp } = await import('../src/server.js');
  const app = createApp();
  return new Promise((res) => {
    const srv = app.listen(0, '127.0.0.1', () => {
      const port = srv.address().port;
      res({
        url: (p) => `http://127.0.0.1:${port}${p}`,
        close: () => { srv.close(); },
      });
    });
  });
}
```

### HTTP request helper (CORRECT pattern)

**CRITICAL PITFALL — `http.get(urlString, options)` does NOT pass headers reliably.**

```javascript
// WRONG — headers may be silently dropped:
http.get('http://127.0.0.1:3000/path', { headers: { ... } }, callback);
// The first argument is a string URL AND an 'options' object. Node.js
// merges them, but the URL's hostname/port from the string override
// any in the options object. Headers in the options object may or may
// not be applied depending on internal resolution order.

// CORRECT — always use http.request() with explicit fields:
function req(method, url, body, headers) {
  const u = new URL(url);
  const opts = {
    hostname: '127.0.0.1',       // NOT 'localhost' (avoids IPv6 issues)
    port: u.port,                 // from the full URL
    path: u.pathname,             // NOT full URL — just the path
    method,
    headers: headers || {},
  };
  return new Promise((resolve, reject) => {
    const r = request(opts, (res) => {
      let d = '';
      res.on('data', c => d += c);
      res.on('end', () => {
        let b;
        try { b = JSON.parse(d); } catch { b = d; }
        resolve({ status: res.statusCode, headers: res.headers, body: b });
      });
    });
    r.on('error', reject);
    if (body) { opts.headers['Content-Type'] = 'application/json'; r.write(JSON.stringify(body)); }
    r.end();
  });
}
```

The `http.request()` form passes `{ hostname, port, path, method, headers }`
as a pure options object — no URL string to confuse the resolution. Use this
for ALL route proof tests.

### Env isolation per test run

Each test suite gets its own temp directory and env overrides:

```javascript
function setupRun() {
  const id = randomUUID().slice(0, 8);
  const tmp = `/tmp/dt-http-proof-${id}`;
  mkdirSync(tmp, { recursive: true });

  const saved = {};
  function setEnv(key, val) {
    saved[key] = process.env[key];
    process.env[key] = val;
  }
  setEnv('ADMIN_API_KEY', 'test-key');
  setEnv('STORE_FILE', join(tmp, 'store.json'));
  setEnv('NODE_ENV', 'test');
  setEnv('ALLOWED_ORIGIN', 'http://localhost');

  return {
    tmp,
    restore: () => {
      for (const [k, v] of Object.entries(saved)) {
        if (v === undefined) delete process.env[k];
        else process.env[k] = v;
      }
      try { rmSync(tmp, { recursive: true, force: true }); } catch {}
    },
  };
}
```

### Specific-claim vs next-claim ordering

When the job store has multiple QUEUED jobs (from prior tests), `GET /next`
returns the FIRST in the internal `job_ids` array order — NOT the most recently
created one. For deterministic tests, use the **specific claim endpoint**:

```javascript
// Deterministic — claims EXACTLY the job we just created
const claim = await req('POST', url(`/job/${jid}/claim`), { worker_id: 't1' }, bridgeHeaders);
assert.equal(claim.status, 200);
```

Compare with the ambiguous form:
```javascript
// Non-deterministic — may return a different QUEUED job if store is not empty
const claim = await req('GET', url('/job/next'), null, bridgeHeaders);
```

Use specific-claim anytime test isolation matters (rejection matrix, replay
proof, concurrent claim tests). Use next-claim only for "poll as worker" tests
where the answer "is there any QUEUED job at all" suffices.

### Lifecycle

```javascript
test('suite', async (t) => {
  const env = setupRun();
  const svc = await buildServer();

  // ... all subtests use svc.url() and env.tmp ...

  svc.close();
  env.restore();      // removes temp dir, restores env vars
});
```

A single `before()` + `after()` works when all subtests share the same server.
Use per-test servers only when env isolation requires it.

Established during: `CLOE_DRAW_THINGS_BRIDGE_HTTP_AND_ASSET_FLOW_PROOF_V1`, refining
the pattern from `CLOE_EXPLICIT_APPROVED_EXECUTION_ENTRYPOINT_RUNTIME_BEHAVIORAL_PROOF_V1`.

## Architecture

```
┌─────────────────────────────┐     ┌──────────────────────┐
│  Test (node:test)            │────▶│  Real Express App    │
│                             │     │  (createApp, port 0) │
│  • Seed temp packet store   │     │  POST /cloe/...      │
│  • Set env vars             │     │  → auth middleware   │
│  • Issue real HTTP requests │     │  → entrypoint        │
│  • Verify responses         │     │  → executor          │
│  • Shutdown & clean         │     └──────────┬───────────┘
│                             │                │
│                             │     ┌──────────▼───────────┐
│                             │     │  Mock LAHB Server    │
│                             │     │  (127.0.0.1:0)       │
│                             │     │  GET /approvals/:id  │
│                             │     └──────────────────────┘
└─────────────────────────────┘
```

## When to Use

Any MIXED mission that needs to prove a real HTTP endpoint works end-to-end
through the registered production route. Do NOT use when unit tests alone
can prove the behavior (e.g. pure functions, DI-injectable services).

## Components

### Mock dependency server
Node built-in `http.createServer` on `127.0.0.1:0`. Key design:
- `setResponse(fn)` — override the response logic per-test
- `resetResponse()` — restore the default (reject `null` assignment which crashes the server)
- The server must handle multiple requests (entrypoint + executor may both call it)
- Return a JSON object with the shape the production consumer expects
- Use GET/POST as the production caller uses

### Env isolation
Use a scoped env helper:
```javascript
function withScopedEnv(envVars) {
  const prev = new Map();
  for (const [k, v] of Object.entries(envVars)) {
    prev.set(k, process.env[k]);
    if (v === undefined) delete process.env[k];
    else process.env[k] = String(v);
  }
  return {
    restore: () => { /* restore all */ },
  };
}
```

Always redirect:
- Store/DB files to temp paths via env vars
- Service URLs to mock server (`127.0.0.1:{port}`)
- Live-mode flags to `'false'`
- Auth keys to test values
- `NODE_ENV` to `'test'`

### Store isolation
The production store reads its path from `process.env`:
```javascript
function getStoreFile() {
  return process.env.STORE_FILE || join(__dirname, '../../data/production.json');
}
```

If the store reads its path at module-load time as `const STORE_FILE = process.env.STORE_FILE || '...'`, the env override will NOT work. See pitfall below.

## The Module-Level Env Constant Trap

**Symptom**: The test sets `process.env.LAHB_URL`, but the production module
still reaches the real URL. The test fails with unexpected HTTP 401/403.

**Root cause**: The module declares `const LAHB_URL = process.env.LAHB_URL || 'https://...'` at the top level. This is evaluated ONCE when the module is first imported — before `test.before()` runs. The test env var has no effect.

**Fix**: Replace with a getter function:
```javascript
const DEFAULT_LAHB_URL = 'https://leanframeworklab.com';
function getLahbUrl() {
  return process.env.LAHB_URL || DEFAULT_LAHB_URL;
}
```

Then use `getLahbUrl()` at every call site instead of the constant. This defers the env read to call time. The change is:
- ~5 lines, backward-compatible (same default)
- A valid testability seam — not a security bypass
- Required for ANY test that redirects service URLs via env vars

## Safe Executor Path

When the executor has live-disabled behavior and the mission must prove the
full chain without provider calls:

| Action Type | Live Disabled Behavior | Safe for Mock Proof? |
|-------------|----------------------|----------------------|
| CAMPAIGN_PLAY | Returns `DRY_RUN_BLOCKED` before any provider call | ✅ Yes |
| CAMPAIGN_PAUSE | Returns `DRY_RUN_BLOCKED` (bypass removed per `CLOE_CAMPAIGN_PAUSE_EMERGENCY_BYPASS_AUTHORITY_AND_FAIL_CLOSED_REPAIR_V1`) | ✅ Yes |
| CAMPAIGN_CREATE_PAUSED | Preflight only, no live send | ⚠️ Partial (needs ExoClick mock) |
| VARIATION_CREATE_VISIBLE | Returns `LIVE_DISABLED` | ⚠️ Partial |

**Rule**: For a pure HTTP proof (no provider mock needed), use CAMPAIGN_PLAY or CAMPAIGN_PAUSE.

## Sentinel Mock for Zero-Provider-Call Proof

When proving a fail-closed behavior (e.g. `DRY_RUN_BLOCKED`) through real HTTP,
use a **sentinel fetch** that throws on any provider URL to fail the test
immediately if an unexpected code path is reached:

```javascript
let loginCalls = 0;
let pauseApiCalls = 0;
let approvalCalls = 0;

function sentinelFetch() {
  return async (url, opts) => {
    const u = String(url);
    if (u.includes('/approvals/')) {
      approvalCalls++;              // LAHB approval — expected
      return mockApprovalResponse();
    }
    if (u.includes('/v2/login')) {
      loginCalls++;                 // ExoClick auth — should be 0 in blocked proof
      return mockLoginResponse();
    }
    if (u.includes('/campaigns/pause')) {
      pauseApiCalls++;              // pauseCampaigns — should be 0 in blocked proof
      throw new Error('SENTINEL HIT: pauseCampaigns called — fail-closed broken');
    }
    throw new Error('SENTINEL HIT: unexpected fetch to ' + u);
  };
}
```

Assert after every blocked-path request:
- `loginCalls === 0` — no provider auth attempted
- `pauseApiCalls === 0` — no pause function called
- No execution receipt created
- `live_sent === false` in the response body

This pattern was established during `CLOE_CAMPAIGN_PAUSE_EMERGENCY_BYPASS_AUTHORITY_AND_FAIL_CLOSED_REPAIR_V1`.

## 90-Scenario Proof Taxonomy

The proof test covers 10 suites (A–J) plus a success scenario:

| Suite | Focus | # Tests |
|-------|-------|---------|
| A | Server + route registration | 5 |
| B | Authentication (auth before execution) | 7 |
| C | Request contract (validation, forbidden fields) | 11 |
| D | Persisted packet truth (schema, completeness) | 8 |
| E | Trusted approval (LAHB mock, mismatch, expiry) | 12 |
| F | Adapter + validation (unsupported types, order) | 11 |
| G | Safe execution (live_sent=false, no provider) | 8 |
| H | Receipt linkage + idempotency | 8 |
| I | Persistence + cleanup | 7 |
| J | Architectural negative proof | 12 |
| SUCCESS | Complete end-to-end scenario | 1 |

## Concurrent Idempotency Proof with Deferred Provider Mocks

Proving `EXECUTION_IN_PROGRESS` for concurrent duplicates requires holding the
provider response in-flight while a second request arrives.

### Pattern: holdPromise + try/finally

```javascript
let holdPromise = null;
let holdResolve = null;

function holdProvider() {
  holdPromise = new Promise((resolve) => { holdResolve = resolve; });
}

function releaseProvider() {
  if (holdResolve) {
    holdResolve();
    holdResolve = null;
    holdPromise = null;
  }
}

// In the mock fetch: await holdPromise BEFORE incrementing the call counter,
// so the counter is 0 while the request is blocked.
if (u.includes('/v2/campaigns/pause')) {
  if (holdPromise) { await holdPromise; }
  pauseCalls++;
  return mockResponse();
}
```

### Critical: try/finally around the concurrent section

If an assertion fails while the deferred provider is blocking, the test throws
and `releaseProvider()` is never called. The first request stays blocked and the
test runner hangs (Node.js event loop never drains).

```javascript
let firstResult;
try {
  // Assertions on the concurrent duplicate response
  assert.equal(second.body.status, 'EXECUTION_IN_PROGRESS');

  // Release provider — first request completes
  releaseProvider();
  firstResult = await firstPromise;
} finally {
  // ALWAYS release, even if assertions failed
  releaseProvider();
  if (!firstResult) {
    // Timeout-safe drain: give the first request a chance to settle
    try {
      firstResult = await Promise.race([
        firstPromise,
        new Promise((resolve) => setTimeout(() => resolve(undefined), 2000))
      ]);
    } catch { /* ok */ }
  }
}
```

HTTP status for `EXECUTION_IN_PROGRESS` is **202**, not 200 (the route handler
maps in-flight to async-accepted semantic).

## Cleanup Protocol

Always in `test.after()`:
1. Restore env vars (the scoped helper's `restore()`)
2. Close the app server — `await new Promise(r => server.close(r))`
3. Close mock servers
4. Remove temp store files
5. Verify no orphan processes remain

## Pitfalls

| Pitfall | Sign | Fix |
|---------|------|-----|
| `setResponse(null)` crashes mock | Next test hits `TypeError: responseFn is not a function` | Add `resetResponse()` that restores the default function |
| Mock port changes after restart | Closing + recreating the mock assigns a new port | Update `process.env.SERVICE_URL` after each restart |
| `createApp()` import evaluates env constants | Module-level `const X = process.env.X` evaluated at import time | Use getter functions (see above) |
| CAMPAIGN_PAUSE now returns DRY_RUN_BLOCKED (bypass removed) | Pre-existing proof test expects CAMPAIGN_PAUSE + gate false to call provider | Update test to use `EXOCLICK_LIVE_ENABLED='true'` explicitly when testing live execution, or assert `DRY_RUN_BLOCKED` for fail-closed proof |
| Repeated requests create state leakage | Receipts/packets persist between tests | `cleanStores()` in every test, `seedCompletePacket()` in every test |
| `ADMIN_API_KEY` missing from env save/restore | Runtime proof script sets `process.env.ADMIN_API_KEY` but forgets to include it in the `prev` save block. The key leaks across the try/finally boundary. | Always save AND restore EVERY env var you set, including `ADMIN_API_KEY`, `EXOCLICK_API_TOKEN`, and any auth-related vars. Check the save block against the set calls — they must be a 1:1 match. |
| **LAHB mock action_type mismatch when live=true** | All existing runtime proof tests use `EXOCLICK_LIVE_ENABLED='false'` so `executeAction` returns `DRY_RUN_BLOCKED` before its internal `verifyApprovalWithLAHB` call. When switching to `EXOCLICK_LIVE_ENABLED='true'`, the LAHB mock runs and its default `action_type` (CAMPAIGN_PLAY) may not match the packet's action_type (CAMPAIGN_PAUSE). | Configure `lahbServer.setResponse()` per-test to return the exact `action_type` from the packet. Extract a helper: `function setLahbApproval(actionType) { lahbServer.setResponse((id) => ({ ok: true, approval: { status: 'APPROVED', action_type: actionType, ... } })); }` |
| **Entrypoint intercepts ALREADY_EXECUTED before executor** | The entrypoint (`executeApprovedGovernedAction`) checks packet linkage at Step 8 and returns `ALREADY_EXECUTED` before calling `executeAction`. This means `incrementDuplicateAttempt` inside `executeAction` is never called for sequential duplicates through the entrypoint. | `duplicate_attempt_count` stays at 0 for entrypoint-driven idempotency. This is correct — the entrypoint provides the idempotency shield, not the executor's redundant check. Assert on `idempotent: true` instead of on `duplicate_attempt_count`. |
| **ExoClick token cache is module-level** | `cachedAccessToken` in `exoclick-login.js` persists across test cases. Mock counters (`loginCalls++`) only increment on actual fetch calls. On cache hits, `loginCalls` stays at 0. | Assert on the real mutation metric (`pauseCalls`, `playCalls`) rather than on login fetch count. Or call `clearExoClickAccessTokenCache()` between test suites to force fresh login fetches. |
| **Concurrent proof without try/finally hangs runner** | Deferred provider mock blocks the first request. An assertion failure in the concurrent duplicate check throws before `releaseProvider()` is reached. The first request stays blocked, Node event loop never drains, test runner hangs until timeout. | Always wrap concurrent-proof sections in `try/finally` that calls `releaseProvider()` and drains the first promise with a timeout. (See "Concurrent Idempotency Proof" section above for the pattern.) |
