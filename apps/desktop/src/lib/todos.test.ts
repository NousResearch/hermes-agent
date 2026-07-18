import { describe, expect, it } from 'vitest'

import { createTodoHistoryMessagesSelector, latestSessionTodos, parseTodos, sessionTodoHistory } from './todos'

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

describe('sessionTodoHistory', () => {
  const list = (status: 'completed' | 'in_progress' | 'pending', content = 'Ship') => [{ content, id: 'ship', status }]

  const todoPart = (todos: unknown) => ({
    args: { todos },
    toolCallId: crypto.randomUUID(),
    toolName: 'todo',
    type: 'tool-call'
  })

  const assistant = (id: string, snapshots: unknown[], pending = false) => ({
    id,
    parts: snapshots.map(todoPart),
    pending,
    role: 'assistant' as const
  })

  it('keeps only the final todo update from each assistant turn', () => {
    const history = sessionTodoHistory([assistant('turn-1', [list('pending'), list('in_progress'), list('completed')])])

    expect(history.snapshots).toEqual([
      expect.objectContaining({ id: 'turn-1', state: 'completed', todos: list('completed') })
    ])
  })

  it('retains distinct snapshots from multiple turns newest first', () => {
    const history = sessionTodoHistory([
      assistant('turn-1', [list('completed', 'Plan')]),
      { id: 'user-2', parts: [], role: 'user' as const },
      assistant('turn-2', [list('completed', 'Build')])
    ])

    expect(history.snapshots.map(snapshot => snapshot.todos[0]?.content)).toEqual(['Build', 'Plan'])
  })

  it('deduplicates identical snapshots across turns', () => {
    const history = sessionTodoHistory([
      assistant('turn-1', [list('completed')]),
      assistant('turn-2', [list('completed')])
    ])

    expect(history.snapshots).toHaveLength(1)
    expect(history.snapshots[0]?.id).toBe('turn-2')
  })

  it('renders stale open snapshots as unfinished history, never running', () => {
    const history = sessionTodoHistory([assistant('turn-1', [list('in_progress')])])

    expect(history.current).toBeNull()
    expect(history.snapshots[0]?.state).toBe('unfinished')
  })

  it('uses the live turn as current and moves it into history when the turn ends', () => {
    const messages = [assistant('turn-1', [list('in_progress')], true)]

    expect(sessionTodoHistory(messages, list('in_progress'), true).current).toEqual(
      expect.objectContaining({ state: 'running', todos: list('in_progress') })
    )
    expect(sessionTodoHistory(messages, null, false)).toEqual(
      expect.objectContaining({ current: null, snapshots: [expect.objectContaining({ state: 'unfinished' })] })
    )
  })

  it('keeps completed transcript history after the live overlay disappears', () => {
    const messages = [assistant('turn-1', [list('completed')])]

    expect(sessionTodoHistory(messages, list('completed'), false).snapshots).toHaveLength(1)
    expect(sessionTodoHistory(messages, null, false).snapshots).toHaveLength(1)
  })

  it('does not promote old history while a new user turn is busy without todos', () => {
    const history = sessionTodoHistory(
      [assistant('turn-1', [list('completed')]), { id: 'user-2', parts: [], role: 'user' }],
      list('completed'),
      true
    )

    expect(history.current).toBeNull()
    expect(history.snapshots).toEqual([expect.objectContaining({ id: 'turn-1', state: 'completed' })])
  })

  it('does not promote old history when the pending assistant has no todo call', () => {
    const history = sessionTodoHistory(
      [
        assistant('turn-1', [list('completed')]),
        { id: 'user-2', parts: [], role: 'user' },
        { id: 'turn-2', parts: [{ text: 'Working', type: 'text' }], pending: true, role: 'assistant' }
      ],
      list('completed'),
      true
    )

    expect(history.current).toBeNull()
    expect(history.snapshots).toEqual([expect.objectContaining({ id: 'turn-1', state: 'completed' })])
  })

  it('treats an explicit empty todo list in the pending turn as no current tasks', () => {
    const history = sessionTodoHistory(
      [assistant('turn-1', [list('completed')]), assistant('turn-2', [[]], true)],
      list('completed'),
      true
    )

    expect(history.current).toBeNull()
    expect(history.snapshots).toEqual([expect.objectContaining({ id: 'turn-1', state: 'completed' })])
  })

  it('ignores the previous live linger when a new pending turn has no todo', () => {
    const history = sessionTodoHistory(
      [assistant('turn-1', [list('completed')]), { id: 'turn-2', parts: [], pending: true, role: 'assistant' }],
      list('completed'),
      true
    )

    expect(history.current).toBeNull()
    expect(history.snapshots[0]?.id).toBe('turn-1')
  })
})

describe('createTodoHistoryMessagesSelector', () => {
  it('preserves reference identity across non-todo streaming deltas', () => {
    const select = createTodoHistoryMessagesSelector()

    const todo = {
      args: { todos: [{ content: 'Ship', id: 'ship', status: 'in_progress' }] },
      toolCallId: 'todo-1',
      toolName: 'todo',
      type: 'tool-call'
    }

    const first = select([
      { id: 'turn-1', parts: [todo, { text: 'A', type: 'text' }], pending: true, role: 'assistant' }
    ])

    const afterTextDelta = select([
      { id: 'turn-1', parts: [todo, { text: 'A longer delta', type: 'text' }], pending: true, role: 'assistant' }
    ])

    expect(afterTextDelta).toBe(first)
  })
})
