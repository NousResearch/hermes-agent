import { describe, expect, it } from 'vitest'

import { latestSessionTodos, mergeTodoItems, nextTodosFromToolEvent, parseTodoPatch, parseTodoRevision, parseTodos, todoTree } from './todos'

describe('todoTree', () => {
  it('orders parents before children with depths', () => {
    const tree = todoTree([
      { content: 'WP1', id: 'wp1', status: 'in_progress' },
      { content: 'WP2', id: 'wp2', status: 'pending' },
      { content: 'T1', id: 't1', parent: 'wp1', status: 'pending' },
      { content: 'T2', id: 't2', parent: 'wp1', status: 'pending' }
    ])

    expect(tree.map(([t, d]) => [t.id, d])).toEqual([
      ['wp1', 0],
      ['t1', 1],
      ['t2', 1],
      ['wp2', 0]
    ])
  })

  it('degrades dangling and self parents to roots', () => {
    const tree = todoTree([
      { content: 'A', id: 'a', parent: 'ghost', status: 'pending' },
      { content: 'B', id: 'b', parent: 'b', status: 'pending' }
    ])

    expect(tree.map(([t, d]) => [t.id, d])).toEqual([
      ['a', 0],
      ['b', 0]
    ])
  })

  it('keeps cycle members instead of dropping them', () => {
    const tree = todoTree([
      { content: 'A', id: 'a', parent: 'b', status: 'pending' },
      { content: 'B', id: 'b', parent: 'a', status: 'pending' }
    ])

    expect(tree.map(([t]) => t.id).sort()).toEqual(['a', 'b'])
  })

  it('preserves parent through parseTodos', () => {
    expect(parseTodos([{ content: 'x', id: 'c', parent: 'p', status: 'pending' }])).toEqual([
      { content: 'x', id: 'c', parent: 'p', status: 'pending' }
    ])
  })
})

describe('parseTodos', () => {
  it('parses todo arrays with valid ids, content, and statuses', () => {
    expect(
      parseTodos([
        { content: 'Gather ingredients', id: 'prep', status: 'completed' },
        { content: 'Boil water', id: 'boil', status: 'in_progress' },
        { content: 'Serve', id: 'serve', status: 'pending' }
      ])
    ).toEqual([
      { content: 'Gather ingredients', id: 'prep', status: 'completed' },
      { content: 'Boil water', id: 'boil', status: 'in_progress' },
      { content: 'Serve', id: 'serve', status: 'pending' }
    ])
  })

  it('parses nested todo payloads from wrapped objects and JSON strings', () => {
    expect(parseTodos({ todos: [{ content: 'Plate', id: 'plate', status: 'pending' }] })).toEqual([
      { content: 'Plate', id: 'plate', status: 'pending' }
    ])

    expect(parseTodos('{"todos":[{"id":"plate","content":"Plate","status":"pending"}]}')).toEqual([
      { content: 'Plate', id: 'plate', status: 'pending' }
    ])
  })

  it('returns null for non-todo payloads', () => {
    expect(parseTodos(undefined)).toBeNull()
    expect(parseTodos('not json')).toBeNull()
    expect(parseTodos({ message: 'no todos here' })).toBeNull()
  })
})

describe('parseTodoRevision', () => {
  it('parses direct and wrapped revisions', () => {
    expect(parseTodoRevision({ revision: 3 })).toBe(3)
    expect(parseTodoRevision({ result: '{"revision":4}' })).toBe(4)
  })

  it('rejects invalid revisions', () => {
    expect(parseTodoRevision({ revision: -1 })).toBeNull()
    expect(parseTodoRevision({ revision: 1.5 })).toBeNull()
    expect(parseTodoRevision({ revision: '3' })).toBeNull()
  })
})

describe('latestSessionTodos', () => {
  const todoPart = (todos: unknown, extra: Record<string, unknown> = {}) => ({
    type: 'tool-call',
    toolCallId: 't1',
    toolName: 'todo',
    args: { todos },
    ...extra
  })

  it('returns the last todo list across the transcript (result beats args)', () => {
    const messages = [
      { parts: [todoPart([{ content: 'Old', id: 'a', status: 'pending' }])] },
      { parts: [{ type: 'text', text: 'hi' }] },
      {
        parts: [
          todoPart([{ content: 'Stale', id: 'a', status: 'pending' }], {
            result: { todos: [{ content: 'Fresh', id: 'a', status: 'completed' }] }
          })
        ]
      }
    ]

    expect(latestSessionTodos(messages)).toEqual([{ content: 'Fresh', id: 'a', status: 'completed' }])
  })

  it('prefers the live carried `todos` field over args', () => {
    const messages = [
      {
        parts: [
          todoPart([{ content: 'Args', id: 'a', status: 'pending' }], {
            todos: [{ content: 'Live', id: 'a', status: 'in_progress' }]
          })
        ]
      }
    ]

    expect(latestSessionTodos(messages)).toEqual([{ content: 'Live', id: 'a', status: 'in_progress' }])
  })

  it('returns null when no todo tool calls exist', () => {
    expect(latestSessionTodos([{ parts: [{ type: 'text', text: 'hi' }] }])).toBeNull()
    expect(latestSessionTodos([])).toBeNull()
  })
})

describe('mergeTodoItems', () => {
  const list = [
    { content: 'Fix C', id: 'c', status: 'in_progress' as const },
    { content: 'Fix D', id: 'd', status: 'pending' as const },
    { content: 'Fix A', id: 'a', status: 'pending' as const }
  ]

  it('updates status by id and keeps the rest of the list', () => {
    expect(mergeTodoItems(list, [{ id: 'c', status: 'completed' }])).toEqual([
      { content: 'Fix C', id: 'c', status: 'completed' },
      { content: 'Fix D', id: 'd', status: 'pending' },
      { content: 'Fix A', id: 'a', status: 'pending' }
    ])
  })

  it('appends a new item and fills missing content', () => {
    expect(mergeTodoItems(list, [{ id: 'v', status: 'pending' }])).toEqual([
      ...list,
      { content: '(no description)', id: 'v', status: 'pending' }
    ])
  })
})

describe('nextTodosFromToolEvent', () => {
  const current = [
    { content: 'Fix C', id: 'c', status: 'pending' as const },
    { content: 'Fix D', id: 'd', status: 'pending' as const }
  ]

  it('replaces from the full tool result', () => {
    expect(
      nextTodosFromToolEvent(current, {
        todos: [
          { content: 'Fix C', id: 'c', status: 'completed' },
          { content: 'Fix D', id: 'd', status: 'in_progress' }
        ]
      })
    ).toEqual([
      { content: 'Fix C', id: 'c', status: 'completed' },
      { content: 'Fix D', id: 'd', status: 'in_progress' }
    ])
  })

  it('merges a status-only start payload instead of replacing the list', () => {
    expect(
      nextTodosFromToolEvent(current, {
        args: { merge: true, todos: [{ id: 'c', status: 'completed' }] }
      })
    ).toEqual([
      { content: 'Fix C', id: 'c', status: 'completed' },
      { content: 'Fix D', id: 'd', status: 'pending' }
    ])
  })

  it('does not wipe the list when a merge payload has no usable items', () => {
    expect(nextTodosFromToolEvent(current, { args: { merge: true, todos: [] } })).toBeNull()
  })

  it('still replaces when merge is off', () => {
    expect(
      nextTodosFromToolEvent(current, {
        args: { todos: [{ content: 'Only this', id: 'c', status: 'completed' }] }
      })
    ).toEqual([{ content: 'Only this', id: 'c', status: 'completed' }])
  })
})

describe('parseTodoPatch', () => {
  it('keeps status-only items that parseTodos would drop', () => {
    expect(parseTodos([{ id: 'c', status: 'completed' }])).toEqual([])
    expect(parseTodoPatch([{ id: 'c', status: 'completed' }])).toEqual([{ id: 'c', status: 'completed' }])
  })
})
