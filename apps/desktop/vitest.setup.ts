import { setTimeout as nodeSetTimeout } from 'node:timers'

import { configure } from '@testing-library/react'
import { afterAll, afterEach } from 'vitest'

import { clearPlaybackSourceCache } from './src/lib/playback-source-cache'
import { InertResizeObserver } from './src/test/jsdom'

// Shared tooltips measure their arrow through Radix's useSize hook.
// Geometry assertions still belong in a real browser, not this inert observer. Assigned,
// not `vi.stubGlobal`, so a test's `vi.unstubAllGlobals()` cannot strip it.
globalThis.ResizeObserver ??= InertResizeObserver as unknown as typeof ResizeObserver

// Node 26 defines its own `localStorage` accessor on the global object, which
// returns `undefined` unless the process was started with --localstorage-file
// (it warns: "localStorage is not available because --localstorage-file was
// not provided"). In the jsdom environment `globalThis` IS the window, so that
// accessor shadows jsdom's Storage and every `localStorage.getItem(...)` in a
// test throws "Cannot read properties of undefined". Install a real in-memory
// Storage when the global resolves to nothing, before any test module reads it.
if (typeof (globalThis as any).localStorage === 'undefined') {
  const store = new Map<string, string>()

  const storage: Storage = {
    get length() {
      return store.size
    },
    key: (i: number) => [...store.keys()][i] ?? null,
    getItem: (k: string) => store.get(String(k)) ?? null,
    setItem: (k: string, v: string) => void store.set(String(k), String(v)),
    removeItem: (k: string) => void store.delete(String(k)),
    clear: () => store.clear()
  }

  for (const target of [globalThis, (globalThis as any).window].filter(Boolean)) {
    Object.defineProperty(target, 'localStorage', {
      value: storage,
      configurable: true,
      writable: true
    })
  }
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
  takeRecords(): IntersectionObserverEntry[] {
    return []
  }
} as typeof IntersectionObserver

// Idle prefetches and feature queries (`CSS.supports`) run on mount.
// jsdom has neither API; idle work never fires and every feature reads as absent.
globalThis.requestIdleCallback ??= () => 0
globalThis.cancelIdleCallback ??= () => undefined
globalThis.CSS ??= { escape: (value: string) => value, supports: () => false } as unknown as typeof CSS

if ('Element' in globalThis) {
  Element.prototype.scrollTo ??= () => undefined

  // React 19 + Testing Library 16: opt into the act environment so render(),
  // fireEvent(), and findBy* queries automatically flush state updates without
  // spurious "not wrapped in act(...)" warnings.
}

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

// Resolved stream URLs outlive a mount on purpose; a test file reusing one path must not
// inherit the previous case's source.
afterEach(() => clearPlaybackSourceCache())

// Radix overlays (menus, popovers, dialogs) hand focus back from a
// `setTimeout(0)` scheduled when their FocusScope unmounts. When a file's last
// test closes one, that timer can outlive the file and fire after the jsdom
// environment is torn down: it builds a Node `CustomEvent`, dispatches it on a
// jsdom node, and the run fails with an unhandled "parameter 1 is not of type
// 'Event'" even though every test passed. Registered here, this hook runs after
// the file's own hooks (cleanup included), and a same-delay timer created later
// fires later, so every unmount timer already queued runs inside the live
// environment. Node's own timer, not the global: a test that stubs or fakes
// `setTimeout` and is not restored (a `-t` filter skipping the restore) must not
// hang this hook; the real timer shares the queue Radix's timer sits in.
afterAll(() => new Promise<void>(resolve => nodeSetTimeout(resolve, 0)))
