import { describe, expect, it } from 'vitest'

import { applySidebarNavPrefs, SIDEBAR_NAV_PREFS_AREA } from './sidebar-nav'

const rows = [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }, { id: 'e' }]

const prefs = (id: string, data: { hide?: string[]; order?: string[] }) => ({ area: SIDEBAR_NAV_PREFS_AREA, id, data })

describe('applySidebarNavPrefs', () => {
  // The arbitration rule two plugins live under: neither can un-hide the
  // other's row; the first-registered order owns the placement it names and a
  // later order only places what is still unplaced; unknown ids are inert;
  // rows nobody names keep their default relative order after the named ones.
  it('unions hides and lets the first-registered order win', () => {
    const merged = applySidebarNavPrefs(rows, [
      prefs('first', { hide: ['b'], order: ['d', 'a'] }),
      prefs('second', { hide: ['c', 'missing'], order: ['a', 'd', 'b', 'nope'] })
    ])

    expect(merged.map(r => r.id)).toEqual(['d', 'a', 'e'])
    expect(merged[0]).toBe(rows[3])
    expect(applySidebarNavPrefs(rows, []).map(r => r.id)).toEqual(['a', 'b', 'c', 'd', 'e'])
  })
})
