import { beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { onPersistenceEvent } from '@/lib/storage'

import {
  $sidebarNavHidden,
  applySidebarNavPrefs,
  applyUserNavHidden,
  setSidebarNavHidden,
  SIDEBAR_NAV_IDS,
  SIDEBAR_NAV_PREFS_AREA
} from './sidebar-nav'

const rows = [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }, { id: 'e' }, { id: 'capabilities' }]

const prefs = (id: string, data: { hide?: string[]; order?: string[] }, order?: number) => ({
  area: SIDEBAR_NAV_PREFS_AREA,
  data,
  id,
  order
})

describe('SIDEBAR_NAV_IDS', () => {
  // Cheap invariant on the canonical core-id list the Settings row list and
  // the sidebar-render test both bind to: a duplicate or blank id would
  // silently break that binding (a Settings toggle naming nothing).
  it('is a duplicate-free list of non-empty strings', () => {
    expect(new Set(SIDEBAR_NAV_IDS).size).toBe(SIDEBAR_NAV_IDS.length)
    SIDEBAR_NAV_IDS.forEach(id => expect(typeof id === 'string' && id.trim() !== '').toBe(true))
  })
})

describe('applySidebarNavPrefs', () => {
  // The arbitration rule two plugins live under: neither can un-hide the
  // other's row; the first contribution's order owns the placement it names
  // and a later order only places what is still unplaced; unknown ids are
  // inert; rows nobody names keep their default relative order after the
  // named ones; the row that hosts the Plugins tab (a plugin's own off-switch)
  // can be moved but never hidden.
  it('unions hides, lets the first order win, and keeps the capabilities row', () => {
    const merged = applySidebarNavPrefs(rows, [
      prefs('first', { hide: ['b', 'capabilities'], order: ['d', 'a'] }),
      prefs('second', { hide: ['c', 'missing'], order: ['a', 'd', 'b', 'nope'] })
    ])

    expect(merged.map(r => r.id)).toEqual(['d', 'a', 'e', 'capabilities'])
    expect(merged[0]).toBe(rows[3])
    expect(applySidebarNavPrefs(rows, []).map(r => r.id)).toEqual(['a', 'b', 'c', 'd', 'e', 'capabilities'])
  })

  // "First" is the registry's order for the area — lowest `Contribution.order`,
  // then registration — not registration alone; the docs state that rule.
  it('reads contributions in registry order: lowest `order` first, then registration', () => {
    const dispose = registry.registerMany([
      prefs('later-wins', { order: ['e'] }, -1),
      prefs('registered-first', { order: ['a'] })
    ])

    try {
      expect(applySidebarNavPrefs(rows, registry.getArea(SIDEBAR_NAV_PREFS_AREA)).map(r => r.id)).toEqual([
        'e',
        'a',
        'b',
        'c',
        'd',
        'capabilities'
      ])
    } finally {
      dispose()
    }
  })
})

describe('user-side row visibility (#119965)', () => {
  // The user's own hide is a core-side persisted preference like every other
  // sidebar setting — plugin-independent, filtered at render BEFORE the
  // contribution arbitration, which then only sees what remains.
  beforeEach(() => {
    window.localStorage.clear()
    $sidebarNavHidden.set([])
  })

  it('applyUserNavHidden drops the named rows, keeping order and object identity', () => {
    const visible = applyUserNavHidden(rows, ['b', 'd'])

    expect(visible.map(r => r.id)).toEqual(['a', 'c', 'e', 'capabilities'])
    expect(visible[0]).toBe(rows[0])
    expect(visible[1]).toBe(rows[2])
  })

  it('applyUserNavHidden is inert for unknown ids and keeps everything when hidden is empty', () => {
    expect(applyUserNavHidden(rows, ['gone', 'never-existed']).map(r => r.id)).toEqual(rows.map(r => r.id))

    const all = applyUserNavHidden(rows, [])
    all.forEach((row, i) => expect(row).toBe(rows[i]))
  })

  it('applyUserNavHidden ignores non-string junk in the hidden list (cleanIds spirit)', () => {
    const junk = ['b', 42, null, undefined, {}, '  '] as unknown as string[]

    expect(applyUserNavHidden(rows, junk).map(r => r.id)).toEqual(['a', 'c', 'd', 'e', 'capabilities'])
  })

  // Deliberate contract, unlike the contribution path: `NEVER_HIDDEN` guards
  // PLUGIN contributions, because a plugin must not remove the row hosting the
  // Plugins tab — the user's path to that plugin's off-switch. The USER hiding
  // their own capabilities row is a legitimate choice, recoverable through the
  // same Settings toggle, ⌘K (which routes to the Plugins tab regardless of
  // row rendering), and the `nav.capabilities` keybind. Pin that split.
  it('the user CAN hide capabilities — NEVER_HIDDEN guards plugins, not the user', () => {
    expect(applyUserNavHidden(rows, ['capabilities']).map(r => r.id)).toEqual(['a', 'b', 'c', 'd', 'e'])
  })

  it('composition: a user-hidden row stays gone even when a contribution orders it; contribution hide still works', () => {
    const visible = applyUserNavHidden(rows, ['d'])

    const merged = applySidebarNavPrefs(visible, [
      prefs('revive-attempt', { order: ['d', 'a'] }),
      prefs('own-hide', { hide: ['c'] })
    ])

    // `d` filtered out before arbitration, so `order` can only place `a`
    // first; `c` still falls to the contribution's own hide.
    expect(merged.map(r => r.id)).toEqual(['a', 'b', 'e', 'capabilities'])
  })

  it('setSidebarNavHidden stores a deduped, trimmed list and re-setting the same content writes nothing', () => {
    setSidebarNavHidden([' b ', 'b', '', 'c'])

    expect($sidebarNavHidden.get()).toEqual(['b', 'c'])
    expect(JSON.parse(window.localStorage.getItem('hermes.desktop.sidebarNavHidden.v1')!)).toEqual(['b', 'c'])

    const writes: string[] = []

    const off = onPersistenceEvent(event => {
      if (event.op === 'write') {
        writes.push(String(event.value))
      }
    })

    try {
      setSidebarNavHidden([' b ', 'b', '', 'c'])
      expect($sidebarNavHidden.get()).toEqual(['b', 'c'])
      expect(writes).toEqual([])
    } finally {
      off()
    }
  })

  // A stored value is hand-editable: the decode must sanitize like the setter.
  // Fresh import (resetModules) so the atom seeds from the polluted record.
  it('the atom decodes a hand-edited storage value down to clean ids', async () => {
    window.localStorage.setItem(
      'hermes.desktop.sidebarNavHidden.v1',
      JSON.stringify(['b', 42, null, {}, ' b ', 'b', '  '])
    )
    vi.resetModules()

    const fresh = await import('./sidebar-nav')

    expect(fresh.$sidebarNavHidden.get()).toEqual(['b'])
  })
})
