import { describe, expect, it } from 'vitest'

import { latestSessionTodos, parseTodos, todoHistoryFromTranscript } from './todos'

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

describe('todoHistoryFromTranscript', () => {
  it('keeps every distinct plan from one assistant turn while collapsing status-only updates', () => {
    const first = [{ content: 'First task', id: 'same', status: 'pending' as const }]
    const completed = [{ content: 'First task', id: 'same', status: 'completed' as const }]
    const replacement = [{ content: 'Replacement task', id: 'same', status: 'in_progress' as const }]

    expect(
      todoHistoryFromTranscript([
        {
          id: 'assistant-shared',
          role: 'assistant',
          timestamp: 10,
          parts: [
            { args: { todos: first }, toolCallId: 'todo-1', toolName: 'todo', type: 'tool-call' },
            { result: { todos: completed }, toolCallId: 'todo-1', toolName: 'todo', type: 'tool-call' },
            { args: { todos: replacement }, toolCallId: 'todo-2', toolName: 'todo', type: 'tool-call' }
          ]
        }
      ])
    ).toEqual([
      { id: 'assistant-shared:todo-2', state: 'unfinished', timestamp: 10, todos: replacement },
      { id: 'assistant-shared:todo-1', state: 'completed', timestamp: 10, todos: completed }
    ])
  })
})
