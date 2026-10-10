import { describe, expect, it } from 'vitest'

import { buildFilterQuery, CARD_ACTIONS, type CardActionKey, matchesFacetFilters, parseFilterQuery, seedHash } from './board'
import type { KanbanTask } from './types'

const visible = (status: string): CardActionKey[] =>
  CARD_ACTIONS.filter(action => action.when(status)).map(action => action.key)

describe('card action menu', () => {
  // The menu is the board's only place to block/unblock/send-to-review, so its
  // visibility map is the behavior contract: one row per status.
  it('maps every status to the actions it can run', () => {
    expect(visible('todo')).toEqual(['comment', 'reassign', 'addLink', 'addChild'])
    expect(visible('running')).toEqual(['block', 'requestReview', 'comment', 'reassign', 'addLink', 'addChild'])
    expect(visible('ready')).toEqual(['block', 'requestReview', 'comment', 'reassign', 'addLink', 'addChild'])
    expect(visible('review')).toEqual(['requestChanges', 'comment', 'reassign', 'addLink', 'addChild'])
    expect(visible('blocked')).toEqual(['unblock', 'comment', 'reassign', 'addLink', 'addChild'])
    expect(visible('done')).toEqual(['comment', 'reassign', 'addLink', 'addChild'])
  })

  it('never offers a verb the backend state machine rejects', () => {
    // Mirrors the SQL guards: block_task and request_review only accept
    // running/ready source states (kanban_db.py), so the menu must not show
    // them anywhere else — done→review would 409, and blocking out of
    // todo/review/done/triage would too.
    for (const action of CARD_ACTIONS) {
      if (action.key !== 'block' && action.key !== 'requestReview') {
        continue
      }

      for (const status of ['todo', 'review', 'done', 'triage', 'archived', 'blocked']) {
        expect(action.when(status as KanbanTask['status']), `${action.key} on ${status}`).toBe(false)
      }
    }
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

describe('filter seed', () => {
  // AC4: a shared URL reopens the same view. Boot-time navigation assigns the
  // whole hash, so a mount must fall back to the query the page was opened with
  // — once — and never resurrect it after the user clears the filters.
  it('falls back to the boot query while the live hash has lost it', () => {
    expect(seedHash('triage=1&status=blocked', '#/kanban')).toBe('#/kanban?triage=1&status=blocked')
    expect(seedHash('triage=1', '#/kanban?board=ops')).toBe('#/kanban?triage=1')
  })

  it('uses the live hash once the boot query is spent', () => {
    expect(seedHash('triage=1', '#/kanban?status=todo', false)).toBe('#/kanban?status=todo')
    expect(seedHash('', '#/kanban')).toBe('#/kanban')
  })

  it('parses a seeded hash into the facets the filter kebab shows', () => {
    expect(parseFilterQuery(seedHash('triage=1&status=blocked', '#/kanban'))).toEqual({
      priority: [],
      status: ['blocked'],
      triage: true
    })
  })
})
