import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TodoItem } from '@/lib/todos'

import {
  $todoRevisionsBySession,
  $todosBySession,
  clearActiveSessionTodos,
  clearSessionTodos,
  restoreSessionTodosFromSnapshot,
  setSessionTodos,
  todosForHydration
} from './todos'

const todo = (id: string, status: TodoItem['status']): TodoItem => ({ content: `task ${id}`, id, status })

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

  it('cancels the pending clear when a new active list arrives', () => {
    setSessionTodos('s1', [todo('a', 'completed')])
    vi.advanceTimersByTime(2_000)

    // The next turn starts a fresh plan before the linger expires.
    setSessionTodos('s1', [todo('a', 'completed'), todo('b', 'pending')])
    vi.advanceTimersByTime(60_000)

    expect($todosBySession.get().s1).toHaveLength(2)
  })

  it('does not re-show a finished list when the same revision is read again', () => {
    setSessionTodos('s1', [todo('a', 'completed')], 3)
    vi.advanceTimersByTime(5_000)

    setSessionTodos('s1', [todo('a', 'completed')], 3)

    expect($todosBySession.get().s1).toBeUndefined()
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

  it('does not restore a finished list from an earlier turn', () => {
    const finished = [todo('a', 'completed'), todo('b', 'cancelled')]

    expect(todosForHydration(finished)).toBeNull()
  })

  it('returns null when there is nothing stored', () => {
    expect(todosForHydration(null)).toBeNull()
  })
})

describe('revisioned snapshots', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    clearSessionTodos('s1')
  })

  afterEach(() => {
    clearSessionTodos('s1')
    vi.useRealTimers()
  })

  it('rejects a snapshot older than the latest live update', () => {
    setSessionTodos('s1', [todo('new', 'in_progress')], 5)
    setSessionTodos('s1', [todo('old', 'pending')], 4)

    expect($todosBySession.get().s1?.[0]?.id).toBe('new')
    expect($todoRevisionsBySession.get().s1).toBe(5)
  })

  it('restores an active snapshot only while the session is running', () => {
    const snapshot = { revision: 7, todos: [todo('active', 'in_progress')] }

    restoreSessionTodosFromSnapshot('s1', snapshot, false)
    expect($todosBySession.get().s1).toBeUndefined()

    restoreSessionTodosFromSnapshot('s1', snapshot, true)
    expect($todosBySession.get().s1?.[0]?.id).toBe('active')
  })

  it('applies an unversioned update after a revisioned snapshot (tool.start merge)', () => {
    setSessionTodos('s1', [todo('a', 'pending'), todo('b', 'pending')], 5)
    setSessionTodos('s1', [todo('a', 'completed'), todo('b', 'pending')])

    expect($todosBySession.get().s1?.[0]?.status).toBe('completed')
    expect($todoRevisionsBySession.get().s1).toBe(5)
  })

  it('does not stamp a watermark from an unused empty snapshot', () => {
    restoreSessionTodosFromSnapshot('s1', { revision: 0, todos: [] }, true)

    expect($todosBySession.get().s1).toBeUndefined()
    expect($todoRevisionsBySession.get().s1).toBeUndefined()

    setSessionTodos('s1', [todo('a', 'in_progress')])
    expect($todosBySession.get().s1?.[0]?.id).toBe('a')
  })
})
