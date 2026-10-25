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

The probe retains strict document-reload assertions. It also exercises the
200-message full remount/reload and live response growth after restoration.
A continuation now observes late Markdown/intrinsic-row resizes until reader
input or a live run takes over: 12-message reload drift changed from 17px to
0px, and the current 200-message reload control changed from 72px to 0px.
Earlier historical long-remount drift was 323px; the current control retains
its distance through remount, but this is not native-app or issue acceptance.

Remove the two temporary Desktop entry files after stopping the server.
