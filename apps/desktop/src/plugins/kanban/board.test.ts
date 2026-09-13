/**
 * Behavior-contract test for the board's client-side filter predicate.
 *
 * Regression for #110249: the ASSIGNEE filter dropdown had no way to isolate
 * tasks with no assignee. `matchesBoardFilters` is the pure predicate the
 * dropdown's sentinel state ('unassigned') and the "All profiles" state ('')
 * both feed into — tested directly (no rendering) per the root AGENTS.md rule
 * against reading/asserting on source shape.
 */
import { describe, expect, it } from 'vitest'

import { matchesBoardFilters } from './board'
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

  it('the "unassigned" sentinel keeps only tasks with a null assignee', () => {
    const filters = { assignee: 'unassigned', search: '', tenant: '' }

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
      matchesBoardFilters(withTenant, { assignee: 'unassigned', search: '', tenant: 'acme' })
    ).toBe(true)
    expect(
      matchesBoardFilters(withTenant, { assignee: 'unassigned', search: '', tenant: 'other' })
    ).toBe(false)
  })
})
