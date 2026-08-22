# Cross-Process Runtime Proof Pattern

Use when a MIXED mission requires proving that two independent OS processes sharing a remote state backend cannot both execute the same approved mutation.

## When to use

- Previous idempotency proof was **mono-process** (same Node.js heap, shared in-memory state)
- The production deployment runs multiple processes (PM2 cluster, multiple containers, worker threads)
- The state backend is remote (HTTP-based: LAHB, Redis, database)
- You need to prove that the remote claim protocol provides cross-process isolation

## Architecture

```
Parent Test
  ├── spawn LAHB mock ──HTTP──► LAHB State Server (claim/receipt/approval)
  ├── spawn sentinel  ──HTTP──► Provider Sentinel (hold/unhold)
  ├── spawn child A   ──► separate process (wins claim, blocked at provider)
  ├── spawn child B   ──► separate process (loses claim → EXECUTION_IN_PROGRESS)
  ├── spawn child C   ──► separate process (post-completion → ALREADY_EXECUTED)
  └── poll LAHB mock for CLAIMED state before spawning B
```

## Key technique: `spawn()` with file-based IPC

**Don't use `fork()` + IPC.** The IPC channel can have timing issues with the test runner, and child stderr is swallowed by TAP output. Use `child_process.spawn()` with temp config/output files:

```javascript
// Parent writes config
writeFileSync(configPath, JSON.stringify({ env, payload, lahbPort, providerPort }));

// Spawn child as independent process
const child = spawn('node', [CHILD_PATH, configPath, outputPath], {
  cwd: join(__dirname, '..'),
  stdio: ['pipe', 'pipe', 'pipe'],
});

// On close, read result from output file
child.on('close', () => {
  const result = JSON.parse(readFileSync(outputPath, 'utf8'));
});
```

## Child script structure

```javascript
// 1. Read config from file
const config = JSON.parse(readFileSync(process.argv[2], 'utf8'));

// 2. Set env BEFORE dynamic import (modules read env at eval time)
for (const [k, v] of Object.entries(config.env)) {
  process.env[k] = String(v);
}

// 3. Dynamic import (env now visible to all imported modules)
const { executeAction } = await import('../../src/services/executor.js');

// 4. Mock fetch — capture realFetch first to avoid recursion
const realFetch = globalThis.fetch;
globalThis.fetch = (url, opts) => {
  if (String(url).includes(`127.0.0.1:${lahbPort}`)) return realFetch(url, opts);
  if (String(url).includes('exoclick') && String(url).includes('/v2/')) {
    return realFetch(url.replace(/https?:\/\/[^/]+/, `http://127.0.0.1:${providerPort}`), opts);
  }
  throw new Error(`SENTINEL: ${url}`);
};

// 5. Call production function
const result = await executeAction(config.payload, {});

// 6. Write result to output file
writeFileSync(process.argv[3], JSON.stringify({ type: 'result', pid: process.pid, result }));
process.exit(0);
```

## LAHB mock protocol quirk

The production `requestState()` throws on **any** non-2xx HTTP status:

```javascript
if (!response.ok) {
  const error = new Error(body.error || `LAHB_STATE_HTTP_${response.status}`);
  throw error;
}
```

So "not found" must return **HTTP 200** with `{ ok: false, error: 'RECEIPT_NOT_FOUND' }`, not HTTP 404. The caller handles this via `result.receipt ?? null`.

## Provider sentinel with deferred gate

```javascript
let _hold = false, _holdResolve = null;

// In pause handler, block if held:
if (_hold) {
  await new Promise(r => { _holdResolve = r; });
}

// Parent controls:
sentinel.hold();
sentinel.unhold();  // releases blocked calls
```

## State-polling synchronization

Poll the shared state backend instead of using `setTimeout` sleeps:

```javascript
async function awaitReceiptStatus(approvalId, targets, timeoutMs = 20_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const r = lahbServer.getState(approvalId);
    if (r && targets.includes(r.execution_status)) return r;
    await new Promise(r => setTimeout(r, 100));
  }
  throw new Error(`Timeout`);
}
```

## Pitfalls

| Trap | Fix |
|------|-----|
| `import()` resolves relative to file path, not CWD | Use `../../src/` from `test/helpers/` |
| `EXOCLICK_API_BASE` set to sentinel host removes 'exoclick' substring | Don't set it — use default base URL |
| Mock fetch calls itself recursively | Capture `realFetch` before replacement |
| Child process leaks | Track PIDs, kill all in `test.after()` and each finally block |
| Module-level `const X = process.env.X` evaluated at import time | Replace with getter function |
