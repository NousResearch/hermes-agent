/**
 * `sidebar.nav.trailing` ownership (#141451).
 *
 * The registry namespaces a plugin's contribution ids (`kanban:nav`), so a
 * trailing decorator declaring `navId: 'nav'` silently decorated nothing — the
 * Kanban pause chip never appeared in the rail. Own the row by ROUTE when the
 * plugin gives one, and stay tolerant of the namespacing when it doesn't.
 */

import { describe, expect, it } from 'vitest'

import { rowOwnsTrailing } from './nav-trailing'

const row = { icon: () => null, id: 'kanban:nav', label: 'Kanban', route: '/kanban' }

describe('rowOwnsTrailing', () => {
  it('matches on the route, whatever the namespaced id', () => {
    expect(rowOwnsTrailing(row, { navPath: '/kanban' })).toBe(true)
    expect(rowOwnsTrailing(row, { navPath: '/elsewhere' })).toBe(false)
  })

  it('matches an unqualified navId against its namespaced row id', () => {
    expect(rowOwnsTrailing(row, { navId: 'nav' })).toBe(true)
  })

  it('matches a fully qualified navId', () => {
    expect(rowOwnsTrailing(row, { navId: 'kanban:nav' })).toBe(true)
  })

  it('does not decorate another plugin’s row', () => {
    expect(rowOwnsTrailing(row, { navId: 'other:nav' })).toBe(false)
    expect(rowOwnsTrailing(row, {})).toBe(false)
    expect(rowOwnsTrailing(row, undefined)).toBe(false)
  })
})
