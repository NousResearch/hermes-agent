import { describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { applySidebarNavPrefs, SIDEBAR_NAV_PREFS_AREA } from './sidebar-nav'

const rows = [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }, { id: 'e' }, { id: 'capabilities' }]

const prefs = (id: string, data: { hide?: string[]; order?: string[] }, order?: number) => ({
  area: SIDEBAR_NAV_PREFS_AREA,
  data,
  id,
  order
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
