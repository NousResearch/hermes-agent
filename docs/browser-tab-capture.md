# Browser Harness tab-ownership capture prerequisite

This is an opt-in creation ledger and admission fence, **not automatic cleanup**.
It changes no tool schema, dependency, configuration default or task lifecycle.
Only an explicit `browser.tab_cleanup_enabled: true` enables capture; absent,
false or non-boolean values preserve the existing Browser Use behavior. The
legacy key name does not mean this prerequisite deletes tabs.

## Scope and identity

- Requires an explicit `BU_CDP_WS` or `BU_CDP_URL`. HTTP discovery resolves once
  to an immutable `/devtools/browser/...` WebSocket URL and pins the daemon to it.
- Stores the ledger at `get_hermes_home()/browser_tabs.sqlite` for the current
  profile. Browser identity, unresolved caller identity and named mission feed a
  stable hashed daemon name. Different named missions for the same caller do not
  attach each other's targets; repeated calls in one mission reuse its live
  captured target.
- Admission is committed before the CLI executes. Linux `/proc` verification
  checks the actual daemon's PID/start time and pinned launch endpoint/name.
  Other platforms or unavailable identity evidence fail closed when opted in.
- Only `Target.createTarget` replies observed through synchronous
  `browser_harness.helpers._send` establish ownership. Target listings are never
  evidence of creation. Entry attaches only a captured target from the same
  daemon, owner, generation and immutable browser identity, or creates a new one.
- Daemon bootstrap tabs, arbitrary Python CDP clients, background threads,
  manually selected tabs and navigations are not ownership evidence. This is an
  attribution prerequisite, not a sandbox restricting arbitrary Python code.

## Completion and uncertainty

A completed CLI process drains an active call with zero tracked requests even
when Python returns a nonzero exit code (for example, `NameError` before or after
completed synchronous sends). Existing quarantine and outstanding requests stay
fenced regardless of the CLI exit code. Timeout, interrupted dispatch, transport
failure or ambiguous creation replies cannot release the admission fence. A
late reply clearing the request counter does not clear quarantine.

Task IDs provide stable routing, **not authoritative task-lifetime evidence**.
Owners remain unresolved. No inferred termination, stale-call expiry, tab sweep,
close operation, lifecycle adapter, scheduler or native-browser cleanup ships in
this change. Ledger records are intentionally retained; a quarantined mission
requires investigation rather than an automatic retry across its fence. Turning
capture off does not certify a quarantined call as safe.

## Verification

Run with the canonical test wrapper:

```sh
scripts/run_tests.sh tests/tools/test_browser_tab_ownership.py \
  tests/tools/test_browser_tab_capture.py \
  tests/tools/test_browser_tab_ownership_integration.py \
  tests/tools/test_browser_use_cli.py
```

Use a temporary `HOME`/`HERMES_HOME` and a disposable unauthenticated Chromium
for runtime checks. Never use an existing relay or a real profile. When the
Harness runtime directory is shared by daemon names, its existing
`BH_RUNTIME_DIR_SHARED=1` option must be set, and its path must fit the platform's
Unix socket length limit.
