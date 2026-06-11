import { describe, expect, it } from 'vitest'

import { parseTodos } from './todos'

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
