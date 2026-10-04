# browser/ — Development Guide

In-browser Hermes backend under Pyodide. Read `browser/README.md` first.

## Invariants

- **Upstream files are never modified.** Everything here works by shimming
  the *environment* (threads, sockets, transports), not by patching upstream
  code. If a change seems to require editing `tui_gateway`/`web_server`,
  stop and reconsider — the goal is the unmodified upstream path.
- **Never block the wasm thread without a pump.** Any Python-side wait must
  re-enter `runtime.pump()` — a raw `Atomics.wait` or JS-level sleep wedges
  the interpreter permanently.
- **Suspension is opt-in.** Only daemon threads whose name/target matches
  a known idle-loop pattern may `SleepBreak`-restart. Everything else
  pump-waits. Never widen the suspendable list to silence a hang — a missed
  loop daemon parks loudly in `start()`; fix the classifier, don't replay
  linear work.
- **No code crosses the bridge.** Frames carry data only. There is no eval,
  no `exec`, no page-supplied JavaScript or Python — the debug channels that
  existed during bring-up were deliberately removed.
- **Fail fast, never wedge.** Anything the wasm build can't do (raw sockets,
  subprocess spawn, PTY) raises a real OS-style error immediately rather
  than blocking.

## Conventions

- `runtime.py` is the single place that patches `threading`/`time`/`queue`/
  `asyncio` primitives — `install()` must run before any upstream import.
- `bootstrap_py.py` owns environment shims (native-module stubs, socket
  guards, the httpx/urllib3/urllib page-fetch transport).
- `gateway.py` owns the frame router (`handle()`), the `_FakeWS` socket
  glue, REST→ASGI dispatch, and `fetch_blocking`.
- JS side uses ES5-style `var`/`function` in `page.js` (it runs before any
  bundler step and must parse in the oldest host that might embed it);
  `worker.mjs` is a real module.
- Comments explain *why* (the browser constraint), not *what*.

## Testing

`pytest tests/browser/` runs under plain CPython — no Pyodide download.
A test that needs the JS bridge injects a fake `js.hermesBridge` module
into `sys.modules` (see `test_gateway.py`).
