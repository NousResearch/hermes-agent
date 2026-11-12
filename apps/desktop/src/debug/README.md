# Dev-only state diagnostics

Two counters that answer, for any interaction: **what re-rendered, why, and
which store pushed it?**

```
window.__RENDER_COUNTS__   what re-rendered, attributed to props / state / parent
window.__ATOM_CHURN__      which store published it, and whether it mattered
```

Both are inert until `start()`, so the idle cost is one no-op branch per commit
and per notify. Neither ships: `vite.config.ts` aliases `@/debug/dev-only` to a
no-op module for any build that isn't the dev server (or `VITE_PERF_PROBE=1`).

## Using it

From the devtools console, while an agent streams:

```js
__RENDER_COUNTS__.start(); __ATOM_CHURN__.start()
// ...let it run for a few seconds...
__RENDER_COUNTS__.stop();  __ATOM_CHURN__.stop()
console.table(__RENDER_COUNTS__.report())
console.table(__ATOM_CHURN__.report())
```

The `wasted` column in each is the fix list:

- **render `wasted`** — re-rendered with neither changed props nor changed hook
  state, i.e. purely because a parent did. A `memo()` or a narrower subscription
  removes it.
- **atom `wasted`** — published a value deep-equal to the previous one.
  `@nanostores/react` bails out on *reference* equality only, so this
  re-renders every subscriber for nothing. This is the
  "preserve reference identity on no-ops" rule in `apps/desktop/AGENTS.md`.

Or as a gated measurement, driving 5 concurrent streaming tabs synthetically
(no backend, no credits):

```bash
node scripts/perf/run.mjs render-churn --spawn --tiles 5 --tokens 240
```

## Why bippy, and not the obvious choices

**React 19.2 removed `injectProfilingHooks` from react-dom.** Verified:
`grep -c injectProfilingHooks node_modules/react-dom/cjs/react-dom-client.development.js`
→ `0`. Only `onCommitFiberRoot` / `onPostCommitFiberRoot` remain. The entire
`mark*` profiling family (`component-render-start`, `state-update`,
`render-scheduled`) is dead on this stack — anything built on it is out.

**`<Profiler>` cannot answer "did the sidebar re-render?"** React invokes
`onRender` for *every* Profiler in a committed tree, including subtrees that
bailed out. Counting those callbacks "proves" a re-render that never happened.
`actualDuration` is not a discriminator either: a bailed-out subtree still
reports a small nonzero duration, so there's no safe threshold. `didFiberRender`
is the honest signal. (Note `app/chat/perf-probe.tsx` exports a `PerfProbe`
Profiler wrapper that is used nowhere — that's why.)

**react-scan** ships the right idea in its undocumented `react-scan/lite`
subpath, but `lite` is a thin wrapper whose only imports are `bippy` and
`bippy/source`. The package pulls ~217 transitive deps (babel, preact) and
floats `react-grab` / `react-doctor` on `latest`, so installs aren't
reproducible, and its main entry currently breaks Vite with a JSON
import-attribute error (upstream issues #448, #467, both open). We take bippy
directly: MIT, zero dependencies.

## The import-order constraint

`main.tsx` imports `@/debug/dev-only` **statically, above `react-dom`**. This is
load-bearing, not stylistic.

react-dom captures the devtools hook at **module init**, not at `createRoot`.
Installing afterwards leaves a hook object in place but `bippy._renderers`
empty, and every commit goes unseen. Verified both directions:

| order | `_renderers` | commits |
|---|---|---|
| bippy installs first | 1 | 1 |
| react-dom evaluates first | 0 | 0 |

A dynamic `import()` behind an `import.meta.env.DEV` guard would resolve a
microtask *after* main.tsx's static graph (react-dom included) had evaluated —
too late. Hence a static import, with production exclusion handled by the
build-time alias instead of tree-shaking.

If you add a test that imports these counters, note the `ui` vitest project's
`setupFiles` pulls in `@testing-library/react` (and thus react-dom) before any
test body runs, so the hook can never install there. Use a config without
`setupFiles`.

## Baseline (5 tabs × 240 tokens, dev renderer, darwin-arm64)

```
sidebar_renders          6
sidebar_wasted           0
wasted_renders      10,432
total_renders       78,385
commits              1,566
wasted_notifies          0
```

The sidebar hypothesis is **refuted**: 6 renders across the whole run, all
attributable to hook state on genuine busy/needsInput edges, none wasted. The
`stableArray` guards on `$workingSessionIds` / `$attentionSessionIds`
(`store/session-states.ts:236-259`) are doing their job.

The real cost is elsewhere — see `topRenders` in the scenario's `detail`.
`$sessionStates` notified 1,200 times with **10 listeners** (fan-out 12,000)
and zero wasted notifies, so the publishing side is honest; the waste is in
components that re-render on a parent commit without their own inputs changing.
