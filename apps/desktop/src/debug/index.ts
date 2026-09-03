// Dev-only state diagnostics: one import, two counters.
//
//   window.__RENDER_COUNTS__  — what re-rendered, and why (props/state/parent)
//   window.__ATOM_CHURN__     — which store published it, and whether it mattered
//
// Imported FIRST in `main.tsx`, before `react-dom`. That ordering is
// load-bearing: react-dom decides at module-init whether a devtools hook
// exists, so bippy must install during THIS module's evaluation. A dynamic
// `import()` from main.tsx would resolve a microtask too late and every commit
// would go unseen — hence a plain static import chain, whose evaluation order
// ESM guarantees.
//
// Both counters are inert until `start()` is called, so the idle cost in dev is
// a single no-op branch per commit / per notify.
//
// Typical session, from the devtools console:
//
//   __RENDER_COUNTS__.start(); __ATOM_CHURN__.start()
//   // ...let an agent stream for a few seconds...
//   __RENDER_COUNTS__.stop();  __ATOM_CHURN__.stop()
//   console.table(__RENDER_COUNTS__.report())
//   console.table(__ATOM_CHURN__.report())
//
// The `wasted` column in each is the fix list: components that re-rendered
// with no changed input, and stores that published a value equal to the last.

import './render-counter'
// Live interaction profiler — arms on real resize/typing so we can measure the
// app under REAL sessions instead of a synthetic scenario's toy transcripts.
// window.__PERF_LIVE__.on() in the console, then just use the app.
import './perf-live'

import { watchSessionAtoms } from './watched-atoms'

watchSessionAtoms()
