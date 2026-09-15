/**
 * Behavior-contract test for the board's client-side filter predicate.
 *
 * Regression for #110249: the ASSIGNEE filter dropdown had no way to isolate
 * tasks with no assignee. `matchesBoardFilters` is the pure predicate the
 * dropdown's sentinel state (UNASSIGNED_LANE) and the "All profiles" state ('')
 * both feed into — tested directly (no rendering) per the root AGENTS.md rule
 * against reading/asserting on source shape.
 */
import { describe, expect, it } from 'vitest'

import { matchesBoardFilters, UNASSIGNED_LANE } from './board'
import type { KanbanTask } from './types'

const task = (overrides: Partial<KanbanTask>): KanbanTask => ({
  id: 't_1',
  title: 'Untitled',
  status: 'ready',
  ...overrides
})

describe('matchesBoardFilters', () => {
  const assigned = task({ id: 't_assigned', assignee: 'butters' })
  const unassigned = task({ id: 't_unassigned', assignee: null })
  const otherAssigned = task({ id: 't_other', assignee: 'default' })

  it('"All profiles" (empty assignee filter) keeps every task, assigned or not', () => {
    const filters = { assignee: '', search: '', tenant: '' }

    expect(matchesBoardFilters(assigned, filters)).toBe(true)
    expect(matchesBoardFilters(unassigned, filters)).toBe(true)
    expect(matchesBoardFilters(otherAssigned, filters)).toBe(true)
  })

  it('the unassigned sentinel keeps only tasks with a null assignee', () => {
    const filters = { assignee: UNASSIGNED_LANE, search: '', tenant: '' }

    expect(matchesBoardFilters(unassigned, filters)).toBe(true)
    expect(matchesBoardFilters(assigned, filters)).toBe(false)
    expect(matchesBoardFilters(otherAssigned, filters)).toBe(false)
  })

  it('a named profile still filters to exactly that assignee (unaffected by the sentinel)', () => {
    const filters = { assignee: 'butters', search: '', tenant: '' }

    expect(matchesBoardFilters(assigned, filters)).toBe(true)
    expect(matchesBoardFilters(unassigned, filters)).toBe(false)
    expect(matchesBoardFilters(otherAssigned, filters)).toBe(false)
  })

  it('composes with search and tenant filters unchanged', () => {
    const withTenant = task({ id: 't_tenant', assignee: null, tenant: 'acme' })

    expect(
      matchesBoardFilters(withTenant, { assignee: UNASSIGNED_LANE, search: '', tenant: 'acme' })
    ).toBe(true)
    expect(
      matchesBoardFilters(withTenant, { assignee: UNASSIGNED_LANE, search: '', tenant: 'other' })
    ).toBe(false)
  })

  it('a profile literally named "unassigned" is filtered as a real profile, not as null-assignee', () => {
    // The sentinel must not collide with a legal profile name ([a-z0-9][a-z0-9_-]{0,63}),
    // which is why it carries a leading ':'.
    const literal = task({ id: 't_literal', assignee: 'unassigned' })

    expect(matchesBoardFilters(literal, { assignee: 'unassigned', search: '', tenant: '' })).toBe(true)
    expect(matchesBoardFilters(unassigned, { assignee: 'unassigned', search: '', tenant: '' })).toBe(false)
    expect(matchesBoardFilters(literal, { assignee: UNASSIGNED_LANE, search: '', tenant: '' })).toBe(false)
    expect(matchesBoardFilters(unassigned, { assignee: UNASSIGNED_LANE, search: '', tenant: '' })).toBe(true)
  })
})
