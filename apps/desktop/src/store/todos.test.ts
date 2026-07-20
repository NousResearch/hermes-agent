import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TodoItem } from '@/lib/todos'

import {
  $todoHistoryBySession,
  $todosBySession,
  clearActiveSessionTodos,
  clearAllSessionTodoState,
  clearSessionTodos,
  finalizeSessionTodoSnapshot,
  rebuildSessionTodoHistory,
  setSessionTodos,
  todosForHydration
} from './todos'

const todo = (id: string, status: TodoItem['status']): TodoItem => ({ content: `task ${id}`, id, status })

const todoMessage = (id: string, todos: TodoItem[], timestamp: number) => ({
  id,
  role: 'assistant',
  timestamp,
  parts: [{ args: { todos }, toolCallId: `call-${id}`, toolName: 'todo', type: 'tool-call' }]
})

describe('persistent task history hydration', () => {
  afterEach(() => clearAllSessionTodoState())

  it('reconstructs completed and unfinished snapshots once from an authoritative transcript', () => {
    rebuildSessionTodoHistory('s1', [
      todoMessage('turn-1', [todo('a', 'completed')], 10),
      { id: 'user-2', parts: [], role: 'user' },
      todoMessage('turn-2', [todo('b', 'in_progress')], 20)
    ])

    expect($todoHistoryBySession.get().s1).toEqual([
      { id: 'turn-2', state: 'unfinished', timestamp: 20, todos: [todo('b', 'in_progress')] },
      { id: 'turn-1', state: 'completed', timestamp: 10, todos: [todo('a', 'completed')] }
    ])
  })

  it('preserves the history references when reconstruction is semantically unchanged', () => {
    const messages = [todoMessage('turn-1', [todo('a', 'completed')], 10)]

    rebuildSessionTodoHistory('s1', messages)
    const historyMap = $todoHistoryBySession.get()
    const sessionHistory = historyMap.s1

    rebuildSessionTodoHistory('s1', messages)

    expect($todoHistoryBySession.get()).toBe(historyMap)
    expect($todoHistoryBySession.get().s1).toBe(sessionHistory)
  })

  it('finalizes from the live list and replaces a duplicate plan even when status changed', () => {
    rebuildSessionTodoHistory('s1', [todoMessage('old', [todo('a', 'in_progress')], 10)])
    setSessionTodos('s1', [todo('a', 'completed')], 'turn-complete')

    finalizeSessionTodoSnapshot('s1', 'turn-complete', 20)

    expect($todoHistoryBySession.get().s1).toEqual([
      { id: 'turn-complete', state: 'completed', timestamp: 20, todos: [todo('a', 'completed')] }
    ])
  })

  it('supports edit + append without reviving the abandoned tail', () => {
    rebuildSessionTodoHistory('s1', [
      todoMessage('keep', [todo('keep', 'completed')], 10),
      todoMessage('stale-tail', [todo('stale', 'completed')], 20)
    ])

    rebuildSessionTodoHistory('s1', [todoMessage('keep', [todo('keep', 'completed')], 10)])
    setSessionTodos('s1', [todo('appended', 'in_progress')], 'replacement')
    finalizeSessionTodoSnapshot('s1', 'replacement', 40)

    expect($todoHistoryBySession.get().s1.map(snapshot => snapshot.id)).toEqual(['replacement', 'keep'])
  })

  it('supports edit + tail replacement and rewind', () => {
    rebuildSessionTodoHistory('s1', [
      todoMessage('keep', [todo('keep', 'completed')], 10),
      todoMessage('stale-tail', [todo('stale', 'completed')], 20)
    ])
    rebuildSessionTodoHistory('s1', [
      todoMessage('keep', [todo('keep', 'completed')], 10),
      todoMessage('replacement', [todo('replacement', 'in_progress')], 40)
    ])
    expect($todoHistoryBySession.get().s1.map(snapshot => snapshot.id)).toEqual(['replacement', 'keep'])

    rebuildSessionTodoHistory('s1', [todoMessage('keep', [todo('keep', 'completed')], 10)])
    expect($todoHistoryBySession.get().s1.map(snapshot => snapshot.id)).toEqual(['keep'])
  })

  it('isolates runtime sessions even when message ids and message references collide', () => {
    const shared = todoMessage('same-message-id', [todo('same-todo-id', 'completed')], 10)

    rebuildSessionTodoHistory('runtime-a', [shared])
    rebuildSessionTodoHistory('runtime-b', [shared])
    setSessionTodos(
      'runtime-a',
      [{ ...todo('same-todo-id', 'in_progress'), content: 'changed only in A' }],
      'same-message-id'
    )
    finalizeSessionTodoSnapshot('runtime-a', 'same-message-id', 20)

    expect($todoHistoryBySession.get()['runtime-a'][0]?.todos[0]?.content).toBe('changed only in A')
    expect($todoHistoryBySession.get()['runtime-b']).toEqual([
      { id: 'same-message-id', state: 'completed', timestamp: 10, todos: [todo('same-todo-id', 'completed')] }
    ])
  })

  it('clears all live and historical todo state at the session-cache boundary', () => {
    rebuildSessionTodoHistory('s1', [todoMessage('one', [todo('one', 'completed')], 10)])
    rebuildSessionTodoHistory('s2', [todoMessage('two', [todo('two', 'completed')], 20)])
    setSessionTodos('s1', [todo('live', 'in_progress')])

    clearAllSessionTodoState()

    expect($todoHistoryBySession.get()).toEqual({})
    expect($todosBySession.get()).toEqual({})
  })

  it('retires authoritative todo ownership when stop clears the visual list', () => {
    setSessionTodos('s1', [todo('a', 'in_progress')], 'turn-a')

    clearSessionTodos('s1')
    finalizeSessionTodoSnapshot('s1', 'turn-a', 20)

    expect($todoHistoryBySession.get().s1).toBeUndefined()
  })
})

describe('setSessionTodos finished-list auto-clear', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    clearSessionTodos('s1')
    vi.useRealTimers()
  })

  it('keeps an in-flight list indefinitely', () => {
    setSessionTodos('s1', [todo('a', 'completed'), todo('b', 'in_progress')])

    vi.advanceTimersByTime(60_000)

    expect($todosBySession.get().s1).toHaveLength(2)
  })

  it('drops the list shortly after every item completes', () => {
    setSessionTodos('s1', [todo('a', 'completed'), todo('b', 'cancelled')])

    expect($todosBySession.get().s1).toHaveLength(2)

    vi.advanceTimersByTime(5_000)

    expect($todosBySession.get().s1).toBeUndefined()
  })

  it('keeps turn ownership after visual linger so a late completion still finalizes history', () => {
    setSessionTodos('s1', [todo('a', 'completed')], 'turn-a')

    vi.advanceTimersByTime(5_000)
    finalizeSessionTodoSnapshot('s1', 'turn-a', 20)

    expect($todosBySession.get().s1).toBeUndefined()
    expect($todoHistoryBySession.get().s1).toEqual([
      { id: 'turn-a', state: 'completed', timestamp: 20, todos: [todo('a', 'completed')] }
    ])
  })

  it('cancels the pending clear when a new active list arrives', () => {
    setSessionTodos('s1', [todo('a', 'completed')])
    vi.advanceTimersByTime(2_000)

    // The next turn starts a fresh plan before the linger expires.
    setSessionTodos('s1', [todo('a', 'completed'), todo('b', 'pending')])
    vi.advanceTimersByTime(60_000)

    expect($todosBySession.get().s1).toHaveLength(2)
  })
})

describe('clearActiveSessionTodos (turn-end cleanup)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    clearSessionTodos('s1')
    vi.useRealTimers()
  })

  it('drops a still-active list when the turn has ended', () => {
    setSessionTodos('s1', [todo('a', 'completed'), todo('b', 'in_progress')])

    clearActiveSessionTodos('s1')

    expect($todosBySession.get().s1).toBeUndefined()
  })

  it('leaves a finished list to its normal linger instead of clearing immediately', () => {
    setSessionTodos('s1', [todo('a', 'completed')])

    clearActiveSessionTodos('s1')

    expect($todosBySession.get().s1).toHaveLength(1)
    vi.advanceTimersByTime(5_000)
    expect($todosBySession.get().s1).toBeUndefined()
  })

  it('is a no-op when the session has no todos', () => {
    clearActiveSessionTodos('s1')

    expect($todosBySession.get().s1).toBeUndefined()
  })
})

describe('todosForHydration (stale-active guard on restore)', () => {
  it('does not restore an active list (stale after a completed turn)', () => {
    expect(todosForHydration([todo('a', 'completed'), todo('b', 'in_progress')])).toBeNull()
    expect(todosForHydration([todo('a', 'pending')])).toBeNull()
  })

  it('restores a finished list so its linger shows the final checkmarks', () => {
    const finished = [todo('a', 'completed'), todo('b', 'cancelled')]

    expect(todosForHydration(finished)).toEqual(finished)
  })

  it('returns null when there is nothing stored', () => {
    expect(todosForHydration(null)).toBeNull()
  })
})
