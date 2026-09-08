# Saved-pane fixture import-delay control

From a root `npm ci` installation, run in `apps/desktop`:

```sh
npx vitest run --config ../../evals/desktop-pane-fixture/vitest.config.mts --reporter verbose
npx vitest run --project ui src/store/session-pane-focus.test.ts --sequence.shuffle.tests --sequence.seed=12345 --reporter verbose
```

The Vite plugin appends an 11-second asynchronous wait to the real pane-tree
store's evaluation. It does not mock pane functions, inspect source text in an
assertion, lower a deadline, or alter the two test bodies. The UI test deadline
remains 15 seconds and the default hook deadline remains 10 seconds. This is a
scheduling-controlled reproduction of import latency, not a claim that a local
machine reproduced the original CI load unassisted. It is opt-in, not a slow CI
test.

For A/B, apply this eval directory to the parent revision and run the same
command, then run on the fixture fix. Base evaluates the graph inside each
`beforeEach`; fixed evaluates it once during test collection. The delay marker
appears twice on base and once on fixed. Expected: base fails with `Hook timed
out in 10000ms`; fixed runs both behavioral assertions successfully.

Measured on the initial base `fb3446a281e`:

| Control | Result |
| --- | --- |
| Base + evaluation delay | 2 hook timeouts, 20.04s test phase |
| Fixed + identical delay | 2 passed, 5ms test phase; 11.84s import phase |
| Fixed normal and reversed test order (seed 12345) | 2 passed in each order |
| Store directory, 8 workers | 123 files / 1491 tests passed |
| Complete UI project, 8 workers | 753 files / 7355 assertions passed, but exit 1 from an unrelated Radix focus-scope timer's wrong-realm Event in `statusbar-visibility.test.tsx` |

Mutation controls, applied separately to `session-states.ts` and then restored:

- Remove `revealTreePane(paneId)` in `focusOpenSession`: the overlay re-adoption
  case fails (`null` instead of `canonical-chat`); the miss case passes.
- Make `focusWorkspaceOwnerSessionTile` return the stored id regardless of
  `focusOpenSession`'s result: the miss case fails (`canonical-chat` instead of
  `null`); the overlay case passes.

The fixture uses the real registry, tree watcher, pane mirror, overlay, tile
store, and focus helpers. Install app-lifetime watchers once per isolated Vitest
file graph. Discard the fixture tile through the real action to clear both the
reactive source and its persisted in-memory bucket; reset layout, focus, preset,
selection, read baseline, and localStorage between cases. The mirror observes the
discard and unregisters its contribution. No production API needs a test-only
reset or disposer.

If the host has exhausted inotify watches, prefix commands with
`CHOKIDAR_USEPOLLING=true`; this changes file watching, not test deadlines. The
recorded store/full-suite and final focused controls used that workaround.
