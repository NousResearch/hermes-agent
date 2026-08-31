import { describe, expect, it } from 'vitest'

import { parseTodoSnapshot, type TodoSnapshot } from './todos'

const snap = (over: Partial<TodoSnapshot> = {}): TodoSnapshot => ({
  generation: 3,
  revision: 2,
  session_id: 'sess-1',
  todos: [
    { content: 'Task one', id: 'a', status: 'in_progress' },
    { content: 'Task two', id: 'b', status: 'pending' }
  ],
  ...over
})

describe('parseTodoSnapshot', () => {
  it('parses a well-formed gateway snapshot', () => {
    const parsed = parseTodoSnapshot(snap())

    expect(parsed).not.toBeNull()
    expect(parsed!.session_id).toBe('sess-1')
    expect(parsed!.revision).toBe(2)
    expect(parsed!.generation).toBe(3)
    expect(parsed!.todos).toHaveLength(2)
    expect(parsed!.todos[0]).toEqual({ content: 'Task one', id: 'a', status: 'in_progress' })
  })

  it('parses a JSON-string payload (defensive, older transports)', () => {
    expect(parseTodoSnapshot(JSON.stringify(snap()))).not.toBeNull()
  })

  it('peeks into {todos: snapshot} wrappers', () => {
    expect(parseTodoSnapshot({ todos: snap() })).not.toBeNull()
  })

  it('is null without a numeric revision (display-only lists never mutate)', () => {
    expect(parseTodoSnapshot({ ...snap(), revision: undefined })).toBeNull()
    expect(parseTodoSnapshot({ ...snap(), revision: '2' })).toBeNull()
    expect(parseTodoSnapshot({ ...snap(), revision: 1.5 })).toBeNull()
    expect(parseTodoSnapshot({ ...snap(), revision: true })).toBeNull()
  })

  it('is null without a numeric generation', () => {
    expect(parseTodoSnapshot({ ...snap(), generation: null })).toBeNull()
    expect(parseTodoSnapshot({ ...snap(), generation: '3' })).toBeNull()
  })

  it('accepts generation 0 (fresh session before any write)', () => {
    const parsed = parseTodoSnapshot({ ...snap(), generation: 0, revision: 0, todos: [] })

    expect(parsed).not.toBeNull()
    expect(parsed!.generation).toBe(0)
  })

  it('is null when todos is missing or not an array', () => {
    expect(parseTodoSnapshot({ ...snap(), todos: undefined })).toBeNull()
    expect(parseTodoSnapshot({ ...snap(), todos: 'nope' })).toBeNull()
  })

  it('is null on non-record input and garbage strings', () => {
    expect(parseTodoSnapshot(null)).toBeNull()
    expect(parseTodoSnapshot(42)).toBeNull()
    expect(parseTodoSnapshot('not json')).toBeNull()
    expect(parseTodoSnapshot([])).toBeNull()
  })

  it('rejects numeric-string revision/generation (typed contract: gateway sends numbers)', () => {
    expect(parseTodoSnapshot({ ...snap(), generation: '3', revision: '2' })).toBeNull()
  })

  it('filters invalid items but keeps the snapshot parseable', () => {
    const parsed = parseTodoSnapshot({
      ...snap(),
      todos: [
        { content: 'ok', id: 'a', status: 'completed' },
        { content: '', id: 'b', status: 'pending' },
        { id: 'c', status: 'pending' },
        'garbage',
        { content: 'also ok', id: 'd', status: 'nope' }
      ]
    })

    expect(parsed).not.toBeNull()
    expect(parsed!.todos).toEqual([{ content: 'ok', id: 'a', status: 'completed' }])
  })
})