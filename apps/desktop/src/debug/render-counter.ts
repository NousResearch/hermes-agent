// Dev-only render counter — answers "what actually re-rendered, and why?".
//
// Loaded from `main.tsx` BEFORE `react-dom` (see the import-order note there).
// That ordering is load-bearing: react-dom decides at module-init whether a
// devtools hook exists, so installing after it has already initialised leaves
// `bippy._renderers` empty and every commit goes unseen.
//
// Why not `<Profiler>`: React invokes `onRender` for EVERY Profiler in a
// committed tree, including subtrees that bailed out. Counting those callbacks
// "proves" the sidebar re-rendered when it did not. `actualDuration` is not a
// discriminator either — a bailed-out subtree still reports a small nonzero
// duration. `didFiberRender` is the honest signal.
//
// Why not react-scan: its `lite` subpath is a thin wrapper over bippy, while
// the package pulls ~217 transitive deps (babel, preact) and floats
// `react-grab`/`react-doctor` on `latest`, which makes installs
// non-reproducible and breaks Vite with a JSON import-attribute error.

import { didFiberRender, type Fiber, getDisplayName, instrument, isCompositeFiber, traverseRenderedFibers } from 'bippy'

/** Why a component re-rendered, attributed per commit. */
export interface RenderRecord {
  /** Commits in which this component actually re-rendered (mount excluded). */
  renders: number
  /** ...of those, how many had at least one changed prop reference. */
  propsChanged: number
  /** ...of those, how many had changed hook state (useState/useMemo/store). */
  stateChanged: number
  /**
   * ...of those, how many consumed a context whose value changed. `memo()`
   * cannot block these — the fix is to narrow or split the provider, not to
   * add a memo boundary.
   */
  contextChanged: number
  /**
   * ...of those, how many had NEITHER changed props, changed state, nor a
   * changed context. These re-rendered purely because a parent did — the
   * wasted work a `memo()` or a narrower store subscription would eliminate.
   */
  wasted: number
  /** Sum of `actualDuration` across counted renders, in ms. */
  totalMs: number
}

const counts = new Map<string, RenderRecord>()
let commits = 0
let recording = false

const blank = (): RenderRecord => ({
  contextChanged: 0,
  propsChanged: 0,
  renders: 0,
  stateChanged: 0,
  totalMs: 0,
  wasted: 0
})

/** Did any prop's reference identity change between the two fiber versions? */
function propsChanged(fiber: Fiber): boolean {
  const prev = fiber.alternate?.memoizedProps as Record<string, unknown> | null | undefined
  const next = fiber.memoizedProps as Record<string, unknown> | null | undefined

  if (!prev || !next) {
    return false
  }

  for (const key of Object.keys(next)) {
    if (!Object.is(prev[key], next[key])) {
      return true
    }
  }

  return Object.keys(prev).length !== Object.keys(next).length
}

/** Did any hook's memoizedState change? Covers useState, useSyncExternalStore
 *  (so nanostores `useStore`), useMemo, and useReducer alike. */
function stateChanged(fiber: Fiber): boolean {
  let next: Fiber['memoizedState'] | null | undefined = fiber.memoizedState
  let prev: Fiber['memoizedState'] | null | undefined = fiber.alternate?.memoizedState

  while (next && prev) {
    if (!Object.is(next.memoizedState, prev.memoizedState)) {
      return true
    }

    next = next.next
    prev = prev.next
  }

  return false
}

/** Did any consumed context value change? A `memo()` cannot block a re-render
 *  caused by context, so distinguishing this from a parent-driven render is
 *  the difference between "add a memo" and "split the provider". */
function contextChanged(fiber: Fiber): boolean {
  let dep = fiber.dependencies?.firstContext

  while (dep) {
    const context = dep.context as { _currentValue?: unknown } | undefined

    if (context && 'memoizedValue' in dep && !Object.is(dep.memoizedValue, context._currentValue)) {
      return true
    }

    dep = dep.next
  }

  return false
}

function record(fiber: Fiber) {
  const name = getDisplayName(fiber)

  if (!name) {
    return
  }

  const entry = counts.get(name) ?? blank()
  const props = propsChanged(fiber)
  const state = stateChanged(fiber)
  const context = contextChanged(fiber)

  entry.renders += 1
  entry.totalMs += fiber.actualDuration ?? 0

  if (props) {
    entry.propsChanged += 1
  }

  if (state) {
    entry.stateChanged += 1
  }

  if (context) {
    entry.contextChanged += 1
  }

  if (!props && !state && !context) {
    entry.wasted += 1
  }

  counts.set(name, entry)
}

/** Rows sorted by wasted renders, then total renders — the fix list, in order. */
function report(limit = 40) {
  return [...counts.entries()]
    .map(([name, r]) => ({ name, ...r, totalMs: Math.round(r.totalMs * 100) / 100 }))
    .sort((a, b) => b.wasted - a.wasted || b.renders - a.renders)
    .slice(0, limit)
}

declare global {
  interface Window {
    __RENDER_COUNTS__?: {
      /** Per-component render attribution since the last `clear()`. */
      counts: Map<string, RenderRecord>
      /** Commits observed since the last `clear()`. */
      commits: () => number
      clear: () => void
      /** Start counting. Cheap no-op until called — zero cost while idle. */
      start: () => void
      stop: () => void
      recording: () => boolean
      /** Sorted worst-offenders table; `console.table`-friendly. */
      report: (limit?: number) => Array<RenderRecord & { name: string }>
      /** Attribution for one component by display name. */
      get: (name: string) => RenderRecord | undefined
    }
  }
}

if (typeof window !== 'undefined' && !window.__RENDER_COUNTS__) {
  instrument({
    onCommitFiberRoot(_id, root) {
      if (!recording) {
        return
      }

      commits += 1
      traverseRenderedFibers(root, fiber => {
        if (isCompositeFiber(fiber) && didFiberRender(fiber)) {
          record(fiber)
        }
      })
    }
  })

  window.__RENDER_COUNTS__ = {
    clear: () => {
      counts.clear()
      commits = 0
    },
    commits: () => commits,
    counts,
    get: name => counts.get(name),
    recording: () => recording,
    report,
    start: () => {
      counts.clear()
      commits = 0
      recording = true
    },
    stop: () => {
      recording = false
    }
  }
}
