import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { InterfaceMode } from '@/store/interface-mode'

import { createLayoutPersistence, LAYOUT_KEYS, type LayoutScope } from './layout-persistence'
import { Codecs } from './persisted'

const classic = (mode: InterfaceMode): LayoutScope => ({ mode, shell: 'classic' })
const portal = (mode: InterfaceMode): LayoutScope => ({ mode, shell: 'portal' })

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
  const scope = createLayoutPersistence(classic('simple'), true)
  const $panes = scope.atom(LAYOUT_KEYS.panes, () => ({}), Codecs.json())
  expect($panes.get()).toEqual(legacy)
  $panes.set({ sidebar: { open: true, widthOverride: 220 } })
  scope.change(classic('advanced'), () => {})
  expect($panes.get()).toEqual(legacy)
  scope.change(classic('simple'), () => {})
  expect($panes.get()).toEqual({ sidebar: { open: true, widthOverride: 220 } })
  expect(window.localStorage.getItem('hermes.desktop.layoutModeScopes.v1')).toBeNull()
  expect(JSON.parse(window.localStorage.getItem(LAYOUT_KEYS.panes)!)).toEqual(legacy)

  vi.unstubAllGlobals()
  createLayoutPersistence(classic('simple'), true)
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
  const first = createLayoutPersistence(classic('simple'), true)
  first.atom(LAYOUT_KEYS.hiddenTabs, () => [] as string[], Codecs.stringArray).set([])
  const reloaded = createLayoutPersistence(classic('simple'), true)
  expect(reloaded.atom(LAYOUT_KEYS.hiddenTabs, () => [] as string[], Codecs.stringArray).get()).toEqual([])
})

it('isolates invalid records and keeps auxiliary layouts off primary storage', () => {
  window.localStorage.setItem(LAYOUT_KEYS.preset, 'user-custom')
  window.localStorage.setItem(`${LAYOUT_KEYS.panes}.simple`, 'invalid-json')
  const primary = createLayoutPersistence(classic('advanced'), true)
  const $panes = primary.atom(LAYOUT_KEYS.panes, () => ({ width: 200 }))
  $panes.set({ width: 300 })
  primary.change(classic('simple'), () => {})
  expect($panes.get()).toEqual({ width: 200 })
  primary.change(classic('advanced'), () => {})
  expect($panes.get()).toEqual({ width: 300 })

  const before = stored()
  const aux = createLayoutPersistence(classic('simple'), false)
  const $preset = aux.atom(LAYOUT_KEYS.preset, () => 'default', Codecs.text)
  expect($preset.get()).toBe('default')
  $preset.set('auxiliary')
  aux.change(classic('advanced'), () => {})
  $preset.set('auxiliary-advanced')
  aux.change(classic('simple'), () => {})
  expect($preset.get()).toBe('auxiliary')
  expect(stored()).toEqual(before)
})

it('keeps the redesign arrangement out of the classic keys in both directions', () => {
  window.localStorage.setItem(LAYOUT_KEYS.tree, '"classic-tree"')
  window.localStorage.setItem(LAYOUT_KEYS.preset, 'default')
  const scope = createLayoutPersistence(classic('advanced'), true)
  const $tree = scope.atom(LAYOUT_KEYS.tree, () => 'declared-default', Codecs.json<string>())
  const $preset = scope.atom(LAYOUT_KEYS.preset, () => 'default', Codecs.text)
  const classicRecord = stored()

  // The redesign starts from the declared default, never from the classic arrangement.
  scope.change(portal('advanced'))
  expect($tree.get()).toBe('declared-default')
  $tree.set('merged-tool-panel')
  $preset.set('custom')

  scope.change(classic('advanced'))
  expect($tree.get()).toBe('classic-tree')
  expect($preset.get()).toBe('default')
  expect(window.localStorage.getItem(LAYOUT_KEYS.tree)).toBe(classicRecord[LAYOUT_KEYS.tree])
  expect(window.localStorage.getItem(LAYOUT_KEYS.preset)).toBe('default')

  // A relaunch straight into the redesign reopens its own arrangement.
  const relaunched = createLayoutPersistence(portal('advanced'), true)
  expect(relaunched.atom(LAYOUT_KEYS.tree, () => 'declared-default', Codecs.json<string>()).get()).toBe(
    'merged-tool-panel'
  )
  expect(relaunched.atom(LAYOUT_KEYS.preset, () => 'default', Codecs.text).get()).toBe('custom')
})

it('scopes the redesign per interface mode without touching the classic Simple layout', () => {
  window.localStorage.setItem(`${LAYOUT_KEYS.preset}.simple`, 'sidebar-left')
  const scope = createLayoutPersistence(portal('simple'), true)
  const $preset = scope.atom(LAYOUT_KEYS.preset, () => 'fallback', Codecs.text)
  expect($preset.get()).toBe('fallback')
  $preset.set('portal-simple-pick')
  scope.change(portal('advanced'))
  expect($preset.get()).toBe('fallback')
  scope.change(classic('simple'))
  expect($preset.get()).toBe('sidebar-left')
  scope.change(portal('simple'))
  expect($preset.get()).toBe('portal-simple-pick')
  expect(window.localStorage.getItem(`${LAYOUT_KEYS.preset}.simple`)).toBe('sidebar-left')
})

it('never writes another window’s layout when a scope switch arrives from that window', () => {
  // Two main windows share storage; the flag flips in A and reaches B by storage event.
  const a = createLayoutPersistence(classic('advanced'), true)
  const b = createLayoutPersistence(classic('advanced'), true)
  const $treeA = a.atom(LAYOUT_KEYS.tree, () => 'default', Codecs.json<string>(), true)
  const $treeB = b.atom(LAYOUT_KEYS.tree, () => 'default', Codecs.json<string>(), true)
  a.change(portal('advanced'))
  b.change(portal('advanced'))
  $treeA.set('tool-panel')
  a.write(LAYOUT_KEYS.tree, JSON.stringify('tool-panel'))
  a.change(classic('advanced'))
  b.change(classic('advanced'))
  expect(window.localStorage.getItem(`${LAYOUT_KEYS.tree}.portal`)).toBe('"tool-panel"')
  expect($treeB.get()).toBe('default')

  // A change this window made in memory only (a resize mid-gesture) is still saved on the way out.
  b.change(portal('advanced'))
  $treeB.set('mid-resize')
  b.change(classic('advanced'))
  expect(window.localStorage.getItem(`${LAYOUT_KEYS.tree}.portal`)).toBe('"mid-resize"')
})
