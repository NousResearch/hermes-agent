import { describe, expect, it } from 'vitest'

import { filterKanbanBoard, type KanbanBoardFilters, pruneTaskSelection } from './board-filter'
import type { KanbanBoard, KanbanTask } from './types'

const task = (
  id: string,
  status: string,
  collapseParentIds: string[] = [],
  extra: Partial<KanbanTask> = {}
): KanbanTask => ({
  collapse_parent_ids: collapseParentIds,
  id,
  link_counts: { children: 0, parents: collapseParentIds.length },
  status,
  title: id,
  ...extra
})

const board = (tasks: KanbanTask[]): KanbanBoard => ({
  assignees: ['builder', 'reviewer'],
  columns: [
    { name: 'ready', tasks: tasks.filter(candidate => candidate.status === 'ready') },
    { name: 'done', tasks: tasks.filter(candidate => candidate.status === 'done') },
    { name: 'archived', tasks: tasks.filter(candidate => candidate.status === 'archived') }
  ],
  latest_event_id: 1,
  now: 1,
  tenants: ['product']
})

const defaults: KanbanBoardFilters = {
  assignee: '',
  search: '',
  showCompletedChildren: false,
  tenant: ''
}

const ids = (filtered: KanbanBoard) => filtered.columns.flatMap(column => column.tasks.map(candidate => candidate.id))

describe('completed-child board filtering', () => {
  it('hides only done child cards with a visible parent', () => {
    const source = board([
      task('workflow-root', 'done', [], {
        link_counts: { children: 2, parents: 0 },
        progress: { done: 1, total: 2 }
      }),
      task('finished-step', 'done', ['workflow-root']),
      task('next-step', 'ready', ['workflow-root']),
      task('standalone-result', 'done'),
      task('archived-step', 'archived', ['workflow-root'])
    ])

    const result = filterKanbanBoard(source, defaults)

    expect(ids(result.board)).toEqual(['next-step', 'workflow-root', 'standalone-result', 'archived-step'])
    expect(result.completedChildren).toBe(1)

    const root = result.board.columns
      .flatMap(column => column.tasks)
      .find(candidate => candidate.id === 'workflow-root')

    expect(root?.progress).toEqual({ done: 1, total: 2 })
  })

  it('reveals completed child tasks when the user enables the history toggle', () => {
    const source = board([task('workflow-root', 'done'), task('finished-step', 'done', ['workflow-root'])])
    const result = filterKanbanBoard(source, { ...defaults, showCompletedChildren: true })

    expect(ids(result.board)).toEqual(['workflow-root', 'finished-step'])
    expect(result.completedChildren).toBe(1)
  })

  it('lets an explicit search find matching completed child tasks without another toggle', () => {
    const source = board([
      task('workflow-root', 'done'),
      task('hidden-step', 'done', ['workflow-root'], { body: 'render the final proposal' }),
      task('other-step', 'done', ['workflow-root'])
    ])

    const result = filterKanbanBoard(source, { ...defaults, search: 'final proposal' })

    expect(ids(result.board)).toEqual(['hidden-step'])
    expect(result.completedChildren).toBe(0)
  })

  it('keeps a completed child visible when its parent is filtered out', () => {
    const source = board([
      task('workflow-root', 'done', [], { assignee: 'builder' }),
      task('review-result', 'done', ['workflow-root'], { assignee: 'reviewer' })
    ])

    const result = filterKanbanBoard(source, { ...defaults, assignee: 'reviewer' })

    expect(ids(result.board)).toEqual(['review-result'])
    expect(result.completedChildren).toBe(0)
  })

  it('keeps a completed child visible when its parent is omitted from the board', () => {
    const source = board([task('orphaned-result', 'done', ['archived-parent'])])
    const result = filterKanbanBoard(source, defaults)

    expect(ids(result.board)).toEqual(['orphaned-result'])
    expect(result.completedChildren).toBe(0)
  })

  it('collapses under any visible parent in a multi-parent graph', () => {
    const source = board([
      task('visible-parent', 'done', [], { tenant: 'product' }),
      task('filtered-parent', 'done', [], { tenant: 'ops' }),
      task('joined-result', 'done', ['visible-parent', 'filtered-parent'], { tenant: 'product' })
    ])

    const result = filterKanbanBoard(source, { ...defaults, tenant: 'product' })

    expect(ids(result.board)).toEqual(['visible-parent'])
    expect(result.completedChildren).toBe(1)
  })

  it('collapses a completed dependency chain to its visible root', () => {
    const source = board([
      task('workflow-root', 'done'),
      task('middle-step', 'done', ['workflow-root']),
      task('final-step', 'done', ['middle-step'])
    ])

    const result = filterKanbanBoard(source, defaults)

    expect(ids(result.board)).toEqual(['workflow-root'])
    expect(result.completedChildren).toBe(2)
  })

  it('keeps reopened child work visible', () => {
    const source = board([task('workflow-root', 'done'), task('reopened-step', 'ready', ['workflow-root'])])
    const result = filterKanbanBoard(source, defaults)

    expect(ids(result.board)).toEqual(['reopened-step', 'workflow-root'])
    expect(result.completedChildren).toBe(0)
  })

  it('counts only collapsible children inside the active assignee and tenant filters', () => {
    const source = board([
      task('matching-parent', 'done', [], { assignee: 'reviewer', tenant: 'product' }),
      task('matching-step', 'done', ['matching-parent'], { assignee: 'reviewer', tenant: 'product' }),
      task('other-profile', 'done', ['matching-parent'], { assignee: 'builder', tenant: 'product' }),
      task('other-tenant', 'done', ['matching-parent'], { assignee: 'reviewer', tenant: 'ops' })
    ])

    const result = filterKanbanBoard(source, { ...defaults, assignee: 'reviewer', tenant: 'product' })

    expect(ids(result.board)).toEqual(['matching-parent'])
    expect(result.completedChildren).toBe(1)
  })

  it('removes hidden cards from bulk selection without churning unchanged state', () => {
    const source = board([task('workflow-root', 'done'), task('finished-step', 'done', ['workflow-root'])])
    const filtered = filterKanbanBoard(source, defaults).board
    const selected = new Set(['workflow-root', 'finished-step'])
    const pruned = pruneTaskSelection(selected, filtered)

    expect([...pruned]).toEqual(['workflow-root'])
    expect(pruneTaskSelection(pruned, filtered)).toBe(pruned)
  })
})
