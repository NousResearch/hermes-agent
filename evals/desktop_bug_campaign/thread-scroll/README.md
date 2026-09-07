# Thread reading-intent probe

This renders the production `Thread`, real assistant-ui external-store runtime,
and production styles in Chromium, with deterministic synthetic transcript
records. No backend or provider is started. It is not a full native-app or
reporter-transcript reproduction.

From the repository root, copy `fixture.tsx` to
`apps/desktop/src/scroll-campaign-probe.tsx` and `fixture.html` to
`apps/desktop/scroll-campaign-probe.html`. Both are temporary files, not production
entries. Start Vite from `apps/desktop` with its normal config and a free loopback
port. Set a private Vite cache **under a node_modules directory**; a cache outside
node_modules can be transformed again by the React compiler.

Use isolated HOME and HERMES_HOME for the server and probe. Set
`PLAYWRIGHT_BROWSERS_PATH` explicitly if HOME changes Chromium's cache lookup.
Both root and apps/desktop dependencies must be available in a worktree.

Run under the campaign's shared test lock:

```sh
THREAD_SCROLL_OUTPUT=/path/to/artifacts \
THREAD_SCROLL_URL=http://127.0.0.1:18480/scroll-campaign-probe.html?thread \
PROBE_TAG=before \
node evals/desktop_bug_campaign/thread-scroll/probe.mjs
```

The probe records real scroll geometry for fresh load, wheel escape, A→B→A,
and a delayed empty→loaded refresh. It asserts preservation of distance from
bottom and a fresh session's tail position. Run against unchanged main first,
then the candidate in a fresh browser context. Each run writes `<PROBE_TAG>.json`.

The optional `?twins` fixture mounts two independent production Threads with
different runtime IDs; it is useful for diagnosis but the automated probe above
expects one viewport. It does not substitute for a tiled pane-shell reproduction.

`ownership-probe.mjs` extends the same fixture (`?ownership`) with real
Thread remounts after changing the production profile and connection-scope
stores, a real document reload, and a partial transcript whose remaining
history is released with `startTransition`. The namespace controls use 12
messages to exclude virtualization estimation from that ownership assertion;
the hydration release expands to the original 200-message history. No DOM
geometry is overridden. Wheel input comes from Playwright's native input path.

The probe deliberately retains a strict document-reload assertion: current
receipts show a 17px reload drift, so an overall nonzero exit is NOT a failure
of the separately recorded profile/gateway/remount and hydration controls.
Do not relax that assertion to call the whole issue solved. The earlier
200-message full remount also drifted by one estimated turn (323px).

Remove the two temporary Desktop entry files after stopping the server.
