import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createLayoutPersistence, LAYOUT_KEYS } from './layout-persistence'
import { Codecs } from './persisted'

beforeEach(() => window.localStorage.clear())
afterEach(() => vi.unstubAllGlobals())

const stored = () =>
  Object.fromEntries(
    Array.from({ length: window.localStorage.length }, (_, i) => {
      const key = window.localStorage.key(i)!

      return [key, window.localStorage.getItem(key)]
    })
  )

it('retains legacy data in memory when migration writes fail and retries on the next launch', () => {
  const legacy = { sidebar: { open: false, widthOverride: 311 } }
  window.localStorage.setItem(LAYOUT_KEYS.panes, JSON.stringify(legacy))
  const storage = window.localStorage
  vi.stubGlobal('localStorage', {
    getItem: storage.getItem.bind(storage),
    removeItem: storage.removeItem.bind(storage),
    setItem: () => {
      throw new DOMException('Full', 'QuotaExceededError')
    }
  })
  const scope = createLayoutPersistence('simple', true)
  const $panes = scope.atom(LAYOUT_KEYS.panes, () => ({}), Codecs.json())
  expect($panes.get()).toEqual(legacy)
  $panes.set({ sidebar: { open: true, widthOverride: 220 } })
  scope.change('advanced', () => {})
  expect($panes.get()).toEqual(legacy)
  scope.change('simple', () => {})
  expect($panes.get()).toEqual({ sidebar: { open: true, widthOverride: 220 } })
  expect(window.localStorage.getItem('hermes.desktop.layoutModeScopes.v1')).toBeNull()
  expect(JSON.parse(window.localStorage.getItem(LAYOUT_KEYS.panes)!)).toEqual(legacy)

  vi.unstubAllGlobals()
  createLayoutPersistence('simple', true)
  expect(JSON.parse(window.localStorage.getItem(`${LAYOUT_KEYS.panes}.simple`)!)).toEqual(legacy)
  expect(window.localStorage.getItem('hermes.desktop.layoutModeScopes.v1')).toBe('true')

  window.localStorage.clear()
  window.localStorage.setItem(LAYOUT_KEYS.hiddenTabs, '["bots"]')
  vi.stubGlobal('localStorage', {
    getItem: storage.getItem.bind(storage),
    removeItem: storage.removeItem.bind(storage),
    setItem: (key: string, value: string) => {
      if (key === 'hermes.desktop.layoutModeScopes.v1') {
        throw new DOMException('Full', 'QuotaExceededError')
      }

      storage.setItem(key, value)
    }
  })
  const first = createLayoutPersistence('simple', true)
  first.atom(LAYOUT_KEYS.hiddenTabs, () => [] as string[], Codecs.stringArray).set([])
  const reloaded = createLayoutPersistence('simple', true)
  expect(reloaded.atom(LAYOUT_KEYS.hiddenTabs, () => [] as string[], Codecs.stringArray).get()).toEqual([])
})

it('isolates invalid records and keeps auxiliary layouts off primary storage', () => {
  window.localStorage.setItem(LAYOUT_KEYS.preset, 'user-custom')
  window.localStorage.setItem(`${LAYOUT_KEYS.panes}.simple`, 'invalid-json')
  const primary = createLayoutPersistence('advanced', true)
  const $panes = primary.atom(LAYOUT_KEYS.panes, () => ({ width: 200 }))
  $panes.set({ width: 300 })
  primary.change('simple', () => {})
  expect($panes.get()).toEqual({ width: 200 })
  primary.change('advanced', () => {})
  expect($panes.get()).toEqual({ width: 300 })

  const before = stored()
  const aux = createLayoutPersistence('simple', false)
  const $preset = aux.atom(LAYOUT_KEYS.preset, () => 'default', Codecs.text)
  expect($preset.get()).toBe('default')
  $preset.set('auxiliary')
  aux.change('advanced', () => {})
  $preset.set('auxiliary-advanced')
  aux.change('simple', () => {})
  expect($preset.get()).toBe('auxiliary')
  expect(stored()).toEqual(before)
})

// --- Per-window storage isolation (t_7690aaea follow-up) -------------------
//
// Electron partitions localStorage by origin, not by BrowserWindow: the main
// window AND every "File > New Window" peer share the exact same keys here
// unless the renderer namespaces them itself. These tests prove the
// namespacing (windowScopedKey, threaded through every readKey/writeKey call
// in this module) actually isolates two windows' layout state, that the main
// window's pre-fix unscoped key still round-trips exactly as before, and
// that a malformed/missing peer id degrades safely instead of throwing or
// scoping on garbage.
describe('per-window storage isolation', () => {
  afterEach(() => {
    window.history.replaceState(null, '', '/')
  })

  it('gives two different peer windows independent layout state that does not collide', () => {
    window.history.replaceState(null, '', '/?peer=1&windowId=win-a')
    const winA = createLayoutPersistence('advanced', true)
    const $treeA = winA.atom(LAYOUT_KEYS.tree, () => null as unknown, Codecs.json())
    $treeA.set({ id: 'a' })

    window.history.replaceState(null, '', '/?peer=1&windowId=win-b')
    const winB = createLayoutPersistence('advanced', true)
    const $treeB = winB.atom(LAYOUT_KEYS.tree, () => null as unknown, Codecs.json())
    expect($treeB.get()).toBeNull() // Window B starts empty, not window A's tree.
    $treeB.set({ id: 'b' })

    expect(window.localStorage.getItem(`${LAYOUT_KEYS.tree}::win:win-a`)).toBe(JSON.stringify({ id: 'a' }))
    expect(window.localStorage.getItem(`${LAYOUT_KEYS.tree}::win:win-b`)).toBe(JSON.stringify({ id: 'b' }))
    // Neither window ever touched the bare legacy key.
    expect(window.localStorage.getItem(LAYOUT_KEYS.tree)).toBeNull()

    // Re-reading window A's own scope still sees only its own tree.
    window.history.replaceState(null, '', '/?peer=1&windowId=win-a')
    const winAReload = createLayoutPersistence('advanced', true)
    expect(winAReload.atom(LAYOUT_KEYS.tree, () => null as unknown, Codecs.json()).get()).toEqual({ id: 'a' })
  })

  it("the main window's legacy unscoped key round-trips exactly as before (no regression)", () => {
    // No ?peer=1 at all: the main window.
    window.history.replaceState(null, '', '/')
    const main = createLayoutPersistence('advanced', true)
    const $tree = main.atom(LAYOUT_KEYS.tree, () => null as unknown, Codecs.json())
    $tree.set({ id: 'main' })

    expect(window.localStorage.getItem(LAYOUT_KEYS.tree)).toBe(JSON.stringify({ id: 'main' }))
    expect(Object.keys(stored()).some(key => key.includes('::win:'))).toBe(false)

    const reloaded = createLayoutPersistence('advanced', true)
    expect(reloaded.atom(LAYOUT_KEYS.tree, () => null as unknown, Codecs.json()).get()).toEqual({ id: 'main' })
  })

  it('falls back to the shared legacy key when a peer window has no/a malformed windowId', () => {
    for (const search of ['/?peer=1', '/?peer=1&windowId=', '/?peer=1&windowId=bad/id..with spaces']) {
      window.localStorage.clear()
      window.history.replaceState(null, '', search)
      const scope = createLayoutPersistence('advanced', true)
      const $tree = scope.atom(LAYOUT_KEYS.tree, () => null as unknown, Codecs.json())
      $tree.set({ id: 'fallback' })

      // Degrades to the pre-fix shared key rather than throwing or writing a
      // key with unsanitized input baked into it.
      expect(window.localStorage.getItem(LAYOUT_KEYS.tree)).toBe(JSON.stringify({ id: 'fallback' }))
    }
  })
})
