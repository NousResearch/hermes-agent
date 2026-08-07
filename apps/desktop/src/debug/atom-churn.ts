// Dev-only nanostores churn counter — the state-side companion to
// `render-counter.ts`. Render counts tell you WHAT re-rendered; this tells you
// WHICH ATOM pushed the update, and whether the push was worth making.
//
// The headline metric is `wasted`: notifications whose new value is deep-equal
// to the old one. `@nanostores/react`'s `useStore` bails out on REFERENCE
// equality only (`snapshotRef.current === value`), so publishing a fresh array
// or object with identical contents re-renders every subscriber for nothing.
// That is the exact failure `apps/desktop/AGENTS.md` names: "Preserve reference
// identity on no-ops."
//
// `listeners` is read from the store's own `lc` (listener count) at notify
// time, because `onNotify` fires even when a store has zero subscribers — a
// raw notify count over-reports. `notifies x listeners` is the real fan-out.

import { onNotify, type Store } from 'nanostores'

export interface AtomChurn {
  /** Times the store notified its listeners. */
  notifies: number
  /**
   * ...of those, how many pushed a value deep-equal to the previous one.
   * Pure waste: every subscriber re-rendered and nothing actually changed.
   */
  wasted: number
  /** Peak listener count seen at notify time (`store.lc`). */
  peakListeners: number
  /** Sum of listeners across notifications — the true re-render fan-out. */
  fanout: number
}

const churn = new Map<string, AtomChurn>()
const unsubscribes: Array<() => void> = []
let recording = false

const blank = (): AtomChurn => ({ fanout: 0, notifies: 0, peakListeners: 0, wasted: 0 })

/** Structural equality, depth-capped so a long transcript array doesn't make
 *  the instrumentation itself the bottleneck. Beyond the cap we compare by
 *  reference, which under-reports waste rather than inventing it. */
function equal(a: unknown, b: unknown, depth = 0): boolean {
  if (Object.is(a, b)) {
    return true
  }

  if (depth > 3 || typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) {
    return false
  }

  if (Array.isArray(a) !== Array.isArray(b)) {
    return false
  }

  const ka = Object.keys(a as object)
  const kb = Object.keys(b as object)

  if (ka.length !== kb.length) {
    return false
  }

  return ka.every(k => equal((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], depth + 1))
}

/** Watch one store. Call at module scope for each atom you want attributed. */
export function watchAtom(name: string, store: Store) {
  const off = onNotify(store, ({ oldValue }) => {
    if (!recording) {
      return
    }

    const entry = churn.get(name) ?? blank()
    // `lc` is nanostores' listener count — see node_modules/nanostores/atom/index.js.
    const listeners = (store as unknown as { lc?: number }).lc ?? 0
    const next = (store as unknown as { value?: unknown }).value

    entry.notifies += 1
    entry.fanout += listeners
    entry.peakListeners = Math.max(entry.peakListeners, listeners)

    if (equal(oldValue, next)) {
      entry.wasted += 1
    }

    churn.set(name, entry)
  })

  unsubscribes.push(off)

  return off
}

/** Rows sorted by wasted notifications, then fan-out — the fix list, in order. */
function report(limit = 40) {
  return [...churn.entries()]
    .map(([name, c]) => ({ name, ...c }))
    .sort((a, b) => b.wasted - a.wasted || b.fanout - a.fanout)
    .slice(0, limit)
}

declare global {
  interface Window {
    __ATOM_CHURN__?: {
      churn: Map<string, AtomChurn>
      clear: () => void
      start: () => void
      stop: () => void
      recording: () => boolean
      report: (limit?: number) => Array<AtomChurn & { name: string }>
      get: (name: string) => AtomChurn | undefined
      /** Names of every watched store, whether or not it has notified. */
      watched: () => string[]
    }
  }
}

if (typeof window !== 'undefined' && !window.__ATOM_CHURN__) {
  window.__ATOM_CHURN__ = {
    churn,
    clear: () => churn.clear(),
    get: name => churn.get(name),
    recording: () => recording,
    report,
    start: () => {
      churn.clear()
      recording = true
    },
    stop: () => {
      recording = false
    },
    watched: () => [...churn.keys()]
  }
}
