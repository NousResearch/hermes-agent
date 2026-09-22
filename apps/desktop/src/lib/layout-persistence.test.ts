import { afterEach, beforeEach, expect, it, vi } from 'vitest'

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
