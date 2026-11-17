import { describe, expect, it } from 'vitest'

import { todoProgress } from './todo-progress'

const item = (id: string, status = 'pending', parent?: string) => ({ id, parent, status })

describe('todoProgress', () => {
  it('keeps the phase pending when its checks pass before acceptance', () => {
    expect(
      todoProgress([item('phase', 'in_progress'), item('a', 'completed', 'phase'), item('b', 'cancelled', 'phase')])
    ).toEqual({
      phases: { total: 1, completed: 0 },
      subtasks: { total: 1, completed: 1 },
      unresolved: 1
    })
  })
  it('reopening a descendant prevents completed ancestors from reporting success', () => {
    expect(
      todoProgress([
        item('phase', 'completed'),
        item('review', 'completed', 'phase'),
        item('probe', 'pending', 'review')
      ])
    ).toEqual({
      phases: { total: 1, completed: 0 },
      subtasks: { total: 2, completed: 0 },
      unresolved: 1
    })
  })
  it('keeps an active check under a cancelled phase visible', () => {
    expect(todoProgress([item('phase', 'cancelled'), item('check', 'pending', 'phase')])).toEqual({
      phases: { total: 0, completed: 0 },
      subtasks: { total: 1, completed: 0 },
      unresolved: 1
    })
  })
  it('handles old snapshots with dangling or cyclic parents without hiding pending work', () => {
    expect(
      todoProgress([item('a', 'completed', 'b'), item('b', 'pending', 'a'), item('orphan', 'completed', 'missing')])
    ).toEqual({
      phases: { total: 3, completed: 1 },
      subtasks: { total: 0, completed: 0 },
      unresolved: 1
    })
  })
  it('does not mutate the persisted snapshot or invent completion for an empty list', () => {
    const todos = Object.freeze([Object.freeze(item('a')), Object.freeze(item('b', 'completed', 'a'))])
    expect(todoProgress(todos).unresolved).toBe(1)
    expect(todoProgress([])).toEqual({
      phases: { total: 0, completed: 0 },
      subtasks: { total: 0, completed: 0 },
      unresolved: 0
    })
  })
})
