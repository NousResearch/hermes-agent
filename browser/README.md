# browser/ — Hermes backend inside a browser tab

Runs the **stock Hermes agent core** — `tui_gateway`, `web_server`, plugins,
profiles, skills — under [Pyodide](https://pyodide.org) in a Web Worker. A
page-side shim (`page.js`) replaces `fetch()` and `WebSocket` so the existing
dashboard webapp talks to the real backend with **no server process**.

This is complementary to `apps/web/` (PR #93508): that work serves the
Desktop renderer from a *server-hosted* backend. This package puts the
*backend itself* in the tab — zero-server Hermes for demos, docs sites,
e-ink terminals, and other environments where installing a Python process
isn't possible or wanted.

## Layout

| File | Role |
|---|---|
| `runtime.py` | Cooperative runtime: `threading`/`queue`/`time.sleep` emulation over a single wasm thread. Daemon threads suspend on blocking primitives and are rescheduled by a deadline scheduler; the asyncio loop is stepped from the same pump. |
| `gateway.py` | In-process server: drives the real `/api/ws` ASGI route per page socket via `httpx-ws` and the real `web_server.app` via `httpx.ASGITransport`. Also owns the page-mediated fetch path used by httpx/urllib3/urllib. |
| `bootstrap_py.py` | Runs before any upstream import: native-module stubs (pty, termios, resource, ...), socket connect guards (fail fast, never wedge), EOF stdin, exit guards, and the transport patch that routes all HTTP egress through the page's `fetch()`. |
| `worker.mjs` | Web Worker: loads Pyodide, unpacks the Python tree + deps, mounts OPFS for `~/.hermes`, boots `browser.gateway`, then pulls inbound frames into Python forever via `proxy.pullInbound`. |
| `page.js` | Page-side half: installs `window.fetch`/`window.WebSocket` shims for same-origin `/api/*`, answers `pullInbound` with buffered frames, and services `fetchRequest` proxy calls from the backend. |
| `vendor/` | Vendored `coincident` dist (MIT) — the sync worker↔page transport. |
| `index.html` | Minimal demo host page: boots the worker and answers a `gateway.ping` JSON-RPC over `/api/ws`. |
| `serve.mjs` | Static dev server with the required COOP/COEP headers + a `/mock-llm` OpenAI-compatible endpoint for smoke tests. |

## Transport

```
page (page.js)                worker.mjs                 Python (Pyodide)
fetch()/WebSocket ──inbound──> proxy.pullInbound ───────> gateway.handle()
                     postMessage <── emit/restReply <── ws tasks / _handle_rest
fetch() real net  <──proxy.fetchRequest <────────────── fetch_blocking()
```

- **Transport is [coincident](https://github.com/WebReflection/coincident)
  (MIT)** — synchronous worker→page proxy calls over
  `SharedArrayBuffer`/`Atomics`. `postMessage` can't reach a worker
  blocked inside Python, so inbound frames ride the pull channel instead.
- **Inbound (page→worker):** `postToWorker` buffers frames; the worker's
  `bridge.pump(ms)` resolves as a sync `proxy.pullInbound(ms)` call once
  frames exist or the timeout elapses.
- **Outbound (worker→page):** `postMessage`, fire-and-forget.
- **Blocking Python waits** (approvals, fetches, thread joins) are sync
  proxy calls that park the worker the same way, so the interpreter never
  wedges while waiting on the page — and no req/resp id matching exists
  on either side.

## Requirements

- Cross-origin isolation for `SharedArrayBuffer`/`Atomics`:
  `Cross-Origin-Opener-Policy: same-origin` +
  `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`).
  Without it, no SAB — the backend cannot block without losing inbound
  frames.
- OPFS (`navigator.storage.getDirectory`) is used for a persistent
  `~/.hermes` when available; the boot falls back to MEMFS.

## Build & run

`worker.mjs` expects these assets next to `index.html` (produced by whatever
build step packages them — see the standalone `hermes-browser` repo for a
reference `assemble`/`pack` pipeline):

- `pyodide/` — Pyodide distribution (`pyodide.mjs` + stdlib)
- `hermes-py.zip` — the Hermes source tree + this `browser/` package
- `hermes-env.zip` — pure-Python wheels (openai, httpx[socks], httpx-ws,
  wsproto, requests, jinja2, websockets, rich, ...)
- `vendor/` — `coincident-main.js` + `coincident-worker.js` (already in
  this tree; the build just copies it next to `index.html`)
- `overlay/manifest.json` — `[{"name": "browser/runtime.py", "url": "..."}]`
  listing files written into `/hermes-py/` before boot

Then:

```sh
node browser/serve.mjs dist 8471   # serves with COOP/COEP + /mock-llm
open http://localhost:8471/
```

## Tests

```sh
pytest tests/browser/
```

`test_runtime.py` pins the cooperative-runtime invariants under plain
CPython (no Pyodide needed): non-blocking primitive calls must not suspend
daemons, suspension is opt-in for loop daemons only (linear work never
replays), `join()` pumps rather than suspends, and the pump tolerates
nested drains.

`test_gateway.py` exercises the frame router, the queue-backed socket glue,
REST → `ASGITransport` dispatch, and the synchronous `fetch_blocking`
round-trip against a fake `js` bridge.

## Security boundaries

- **No raw sockets.** `socket.connect`/`bind`/`getaddrinfo`/`connect_ex`
  raise `EHOSTUNREACH` immediately — Emscripten's WS-proxy fallback can
  block inside a C call forever, wedging the cooperative interpreter.
  All egress is `fetch()`, subject to normal browser CORS.
- **No code across the boundary.** The page bridge carries data frames
  only; there is no eval/exec channel in either direction.
- **Session token.** The page mints a per-origin `sessionStorage` token
  and the REST/ws shims inject it — same credential model as `hermes serve`.
- **Loopback semantics.** The ASGI scope reports client `127.0.0.1`; the
  app's own Host/Origin DNS-rebinding guards run unchanged (`Host` is
  synthesized from the page origin).
