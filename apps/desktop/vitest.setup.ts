import { configure } from '@testing-library/react'

// Keep ONE storage implementation across every supported Node version.
//
// Node 26 defines its own `localStorage` accessor on the global object, which
// returns `undefined` unless the process was started with --localstorage-file
// (it warns: "localStorage is not available because --localstorage-file was
// not provided"). In the jsdom environment `globalThis` IS the window, so that
// accessor shadows jsdom's Storage and every `localStorage.getItem(...)` in a
// test throws "Cannot read properties of undefined".
//
// Node 22/24 do not define that accessor, so jsdom's proxy Storage used to
// leak through instead: it mirrors stored keys as named properties and
// silently ignores instance-level patches (`vi.spyOn(localStorage, 'setItem')`
// never fires while the write goes straight through). Storage-failure tests
// (e.g. voice-prefs' quota/denied paths) therefore only misbehaved outside
// CI. Install the same in-memory Storage on every supported Node, before any
// test module reads it, so local runs match CI exactly.
const store = new Map<string, string>()

const storage: Storage = {
  get length() {
    return store.size
  },
  key: (i: number) => [...store.keys()][i] ?? null,
  getItem: (k: string) => store.get(String(k)) ?? null,
  setItem: (k: string, v: string) => void store.set(String(k), String(v)),
  removeItem: (k: string) => void store.delete(String(k)),
  clear: () => store.clear(),
}

for (const target of [globalThis, (globalThis as any).window].filter(Boolean)) {
  Object.defineProperty(target, 'localStorage', {
    value: storage,
    configurable: true,
    writable: true,
  })
}

// jsdom has no layout or intersection delivery. Tests of observer behavior
// supply their own callbacks; ordinary component tests only need the lifecycle.
globalThis.IntersectionObserver = class {
  readonly root = null
  readonly rootMargin = '0px'
  readonly thresholds = [0]
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords(): IntersectionObserverEntry[] { return [] }
} as typeof IntersectionObserver

// React 19 + Testing Library 16: opt into the act environment so render(),
// fireEvent(), and findBy* queries automatically flush state updates without
// spurious "not wrapped in act(...)" warnings.
;(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true

// findBy*/waitFor default to a 1000ms deadline — too tight for async-heavy
// panels (radix menus, refetch chains) when the full suite runs under xdist
// CPU contention in CI. Success still resolves the instant the node appears;
// the wider deadline only absorbs a starved runner, killing timing flakes.
// 5s proved insufficient on saturated runners (2026-08-31: gateway-settings,
// messaging, session-unread-tile, toolset-config-panel each tripped a
// waitFor(mock-called) deadline on runs whose only common factor was load —
// including a plugins-only commit on main). 12s mirrors the same reasoning
// as the 15s testTimeout above it while still finishing below it, so a
// genuinely hung await still surfaces as this assertion, not a test timeout.
configure({ asyncUtilTimeout: 12_000 })
