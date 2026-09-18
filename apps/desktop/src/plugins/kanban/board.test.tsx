import { describe, expect, it } from 'vitest'

import { buildFilterQuery, CARD_ACTIONS, type CardActionKey, matchesFacetFilters, parseFilterQuery } from './board'
import type { KanbanTask } from './types'

const visible = (status: string): CardActionKey[] =>
  CARD_ACTIONS.filter(action => action.when(status)).map(action => action.key)

describe('card action menu', () => {
  // The menu is the board's only place to block/unblock/send-to-review, so its
  // visibility map is the behavior contract: one row per status.
  it('maps every status to the actions it can run', () => {
    expect(visible('todo')).toEqual(['block', 'comment', 'reassign', 'addLink', 'addChild'])
    expect(visible('running')).toEqual(['block', 'comment', 'reassign', 'addLink', 'addChild'])
    expect(visible('review')).toEqual(['block', 'requestChanges', 'comment', 'reassign', 'addLink', 'addChild'])
    expect(visible('blocked')).toEqual(['unblock', 'comment', 'reassign', 'addLink', 'addChild'])
    expect(visible('done')).toEqual(['block', 'requestReview', 'comment', 'reassign', 'addLink', 'addChild'])
  })

  it('names the rows that go inert for a multi-card selection', () => {
    // Everything that collects a per-card note, plus the single-card drawer
    // jumps, disables (not hides) when several cards are selected.
    expect(CARD_ACTIONS.filter(action => !action.batchSafe).map(action => action.key)).toEqual([
      'block',
      'requestChanges',
      'comment',
      'addLink',
      'addChild'
    ])
  })
})

describe('filter facets', () => {
  const task = (over: Partial<KanbanTask>): KanbanTask => ({ id: 't_1', priority: 0, status: 'todo', title: 'x', ...over })

  it('round-trips through the hash query, keeping the route and other params', () => {
    const built = buildFilterQuery('#/kanban?group=1', {
      priority: [5, -5],
      status: ['todo', 'blocked'],
      triage: true
    })

    expect(built).toBe('/kanban?group=1&status=todo,blocked&priority=high,low&triage=1')
    expect(parseFilterQuery(built)).toEqual({ priority: [5, -5], status: ['todo', 'blocked'], triage: true })
  })

  it('drops empty facets, unknown priority keys, and absent queries', () => {
    expect(buildFilterQuery('#/kanban?status=todo', { priority: [], status: [], triage: false })).toBe('/kanban')
    expect(parseFilterQuery('#/kanban?status=todo,done&priority=urgent')).toEqual({
      priority: [],
      status: ['todo', 'done'],
      triage: false
    })
    expect(parseFilterQuery('')).toEqual({ priority: [], status: [], triage: false })
  })

  it('stacks the three facets, each empty dimension passing everything', () => {
    const all = { priority: [], status: [], triage: false }

    expect(matchesFacetFilters(task({}), all)).toBe(true)
    expect(matchesFacetFilters(task({ status: 'done' }), { ...all, status: ['todo', 'blocked'] })).toBe(false)
    expect(matchesFacetFilters(task({ priority: 5 }), { ...all, priority: [5, 0] })).toBe(true)
    expect(matchesFacetFilters(task({ priority: -5 }), { ...all, priority: [5] })).toBe(false)
    expect(matchesFacetFilters(task({ triage_signal: true }), { ...all, triage: true })).toBe(true)
    expect(matchesFacetFilters(task({}), { ...all, triage: true })).toBe(false)
    // Stacked: a todo with normal priority matches both dimensions; a done
    // card with high priority misses on status.
    expect(matchesFacetFilters(task({}), { priority: [0], status: ['todo'], triage: false })).toBe(true)
    expect(matchesFacetFilters(task({ priority: 5, status: 'done' }), { priority: [0], status: ['todo'], triage: false })).toBe(
      false
    )
  })
})
