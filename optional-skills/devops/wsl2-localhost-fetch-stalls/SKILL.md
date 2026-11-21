---
name: wsl2-localhost-fetch-stalls
description: "Diagnose WSL2 localhost fetch stalls that fake timing-test failures."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [WSL2, Networking, Testing, Node, Performance]
    related_skills: [hermes-agent, systematic-debugging]
---

# WSL2 Localhost Fetch Stalls

A test that asserts on wall-clock time can fail on a WSL2 machine while the code
under test is completely correct. Before editing production constants, prove which
side is lying.

## When to use

- A timing-sensitive test (staleness pruning, rate windows, debounces, lease
  expiry) passes in CI and fails only on a WSL2 dev box.
- `fetch()` against `127.0.0.1` sometimes takes 200–450 ms while back-to-back
  calls to the same server take 1–3 ms.
- You are deciding whether a latency measurement is the application or the
  platform.

## The artifact: idle-socket reuse, not "slow fetch"

Node's built-in `fetch` (undici) pools connections. On WSL2, reusing a socket that
sat idle for ~60 ms can stall 200–450 ms before the response is handed to the
caller, against a trivial `node:http` server. Back-to-back requests stay at 1–3 ms,
and `setTimeout` accuracy is unaffected, so neither the network stack nor the code
is slow — the cost is attached to socket reuse.

The consequence matters more than the mechanism: a test that sleeps, then asserts
on elapsed time while holding an idle pooled socket can fail here with correct
logic. The signature is an assertion showing **more** wall time elapsed than the
test's own sleeps allow.

## Confirm it in 20 lines

```js
import http from 'node:http'

const srv = http.createServer((req, res) => res.end('ok'))
await new Promise((r) => srv.listen(0, '127.0.0.1', r))
const url = `http://127.0.0.1:${srv.address().port}/`
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const run = async (label, opts) => {
  const rows = []
  for (let i = 0; i < 4; i += 1) {
    const t = performance.now()
    await fetch(url, opts)
    rows.push(Math.round(performance.now() - t))
    await sleep(60)
  }
  console.log(label.padEnd(26), rows.map((x) => `${x}ms`).join(' '))
}

await run('default (pooled)', {})
await run('connection: close', { headers: { connection: 'close' } })
srv.close()
```

Measured on WSL2 (kernel `6.6.87.2-microsoft-standard-WSL2`, Node 26.7.0):

```
default (pooled)      16ms  437ms  345ms  37ms
connection: close     451ms  3ms    3ms    8ms
```

The pooled row is the artifact; the second row shows it disappears once the socket
is not reused.

## Fix the transport, not the production constant

- In the test client, stop reusing sockets: `fetch(url, { headers: { connection: 'close' } })`,
  or replay the scenario with `node:http` and one fresh request per call.
- For window-based assertions, make the clock injectable in the test process and
  advance it explicitly, instead of lengthening sleeps. Widening a margin is the
  trap: the window has two ends, so a longer wait expires the entity that was
  supposed to survive, and the failure changes shape rather than going away.
- Assert both directions — the idle entity IS reclaimed and the heartbeating one
  is NOT — otherwise the over-aggressive threshold never shows up in tests.
- Never widen a production timeout or prune window to satisfy a test that only
  fails on WSL2.

## Rules

- A test that fails only on WSL2 is presumptively environmental; confirm on a
  plain Linux runner before touching application code.
- `undici` is bundled inside Node and is not importable from application code
  unless the package is installed separately (verified on Node 26.7.0), so
  `new Agent({ keepAliveTimeout })` is not available as a one-line test fix —
  disable reuse at the HTTP layer instead.
- Rule out load first (`uptime`), then re-run the file alone. `node --test` runs
  test files concurrently: a single-file failure cannot be contention, while a
  single-file pass does not clear a flake the full suite still hits.
- Do not summarise this as "fetch is slow on WSL2". Busy sequential fetches are
  full speed; the stall is specifically idle-socket reuse to `127.0.0.1`.
