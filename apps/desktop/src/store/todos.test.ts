import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TodoItem } from '@/lib/todos'

import {
  $todoContinuationsBySession,
  $todoRevisionsBySession,
  $todosBySession,
  applyTodoContinuationSnapshot,
  clearActiveSessionTodos,
  clearAllTodoContinuations,
  clearSessionTodos,
  clearTodoContinuation,
  resolveTodoPresentation,
  restoreSessionTodosFromSnapshot,
  setSessionTodos,
  type TodoContinuationSnapshot,
  todosForHydration
} from './todos'

const todo = (id: string, status: TodoItem['status']): TodoItem => ({ content: `task ${id}`, id, status })

describe('setSessionTodos finished-list auto-clear', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    clearSessionTodos('s1')
    clearAllTodoContinuations()
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
})

describe('clearActiveSessionTodos (turn-end cleanup)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    clearSessionTodos('s1')
    clearAllTodoContinuations()
    vi.useRealTimers()
  })

  it('drops a still-active list when the turn has ended', () => {
    setSessionTodos('s1', [todo('a', 'completed'), todo('b', 'in_progress')])

    clearActiveSessionTodos('s1')

    expect($todosBySession.get().s1).toBeUndefined()
  })

  it('keeps unfinished rows when authoritative active continuation will render them statically', () => {
    setSessionTodos('s1', [todo('a', 'completed'), todo('b', 'in_progress')])
    $todoContinuationsBySession.set({ s1: { revision: 1, state: 'active' } })

    clearActiveSessionTodos('s1')

    expect($todosBySession.get().s1).toHaveLength(2)
  })

  it('keeps unfinished rows with an authoritative paused stop reason', () => {
    setSessionTodos('s1', [todo('a', 'in_progress')])
    $todoContinuationsBySession.set({
      s1: { revision: 2, state: 'paused', stopReason: 'Goal stopped after an error' }
    })

    clearActiveSessionTodos('s1')

    expect($todosBySession.get().s1).toHaveLength(1)
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

describe('authoritative todo continuation snapshots', () => {
  afterEach(() => {
    clearSessionTodos('s1')
    clearAllTodoContinuations()
  })

  it('retires retained unfinished rows when a newer none snapshot arrives after turn end', () => {
    setSessionTodos('s1', [todo('a', 'in_progress')])
    applyTodoContinuationSnapshot('s1', { revision: 4, state: 'active' })
    clearActiveSessionTodos('s1')

    applyTodoContinuationSnapshot('s1', { revision: 5, state: 'none' })

    expect($todosBySession.get().s1).toBeUndefined()
  })

  it('per-session reset clears its snapshot and revision high-water mark only', () => {
    applyTodoContinuationSnapshot('s1', { revision: 4, state: 'paused' })
    applyTodoContinuationSnapshot('s2', { revision: 8, state: 'active' })

    clearTodoContinuation('s1')
    applyTodoContinuationSnapshot('s1', { revision: 1, state: 'active' })

    expect($todoContinuationsBySession.get()).toEqual({
      s1: { revision: 1, state: 'active' },
      s2: { revision: 8, state: 'active' }
    })
  })

  it('global reset clears snapshots and revision high-water marks', () => {
    applyTodoContinuationSnapshot('s1', { revision: 4, state: 'paused' })

    clearAllTodoContinuations()
    applyTodoContinuationSnapshot('s1', { revision: 1, state: 'active' })

    expect($todoContinuationsBySession.get().s1).toEqual({ revision: 1, state: 'active' })
  })

  it('keeps the newest revision and clears explicit none state', () => {
    applyTodoContinuationSnapshot('s1', { revision: 4, state: 'paused', stopReason: 'Turn budget exhausted' })
    applyTodoContinuationSnapshot('s1', { revision: 3, state: 'active' })

    expect($todoContinuationsBySession.get().s1).toMatchObject({ revision: 4, state: 'paused' })

    applyTodoContinuationSnapshot('s1', { revision: 5, state: 'none' })
    expect($todoContinuationsBySession.get().s1).toBeUndefined()
  })
})

describe('todo presentation state', () => {
  const active = [todo('done', 'completed'), todo('current', 'in_progress'), todo('next', 'pending')]

  const continuation = (state: TodoContinuationSnapshot['state'], stopReason?: string): TodoContinuationSnapshot => ({
    revision: 3,
    state,
    stopReason
  })

  it('marks an authoritative live turn as working and counts only remaining tasks', () => {
    expect(resolveTodoPresentation(active, { turnLive: true })).toEqual({ kind: 'working', remaining: 2 })
  })

  it('hides unfinished rows after completion when there is no authoritative goal', () => {
    expect(resolveTodoPresentation(active, { turnLive: false })).toEqual({ kind: 'hidden', remaining: 2 })
  })

  it('shows unfinished rows as continuing for an authoritative active goal without a spinner', () => {
    expect(resolveTodoPresentation(active, { continuation: continuation('active'), turnLive: false })).toEqual({
      kind: 'continuing',
      remaining: 2
    })
  })

  it('shows paused and error-stopped goals as paused with the backend stop reason', () => {
    expect(
      resolveTodoPresentation(active, {
        continuation: continuation('paused', 'Goal stopped after a provider error'),
        turnLive: false
      })
    ).toEqual({ kind: 'paused', remaining: 2, stopReason: 'Goal stopped after a provider error' })
  })

  it('restores unfinished history only when authoritative continuation state permits it', () => {
    expect(todosForHydration(active, continuation('active'))).toEqual(active)
    expect(todosForHydration(active, continuation('paused', 'Turn budget exhausted'))).toEqual(active)
    expect(todosForHydration(active)).toBeNull()
    expect(todosForHydration(active, continuation('none'))).toBeNull()
  })

  it('classifies a finished list independently of turn or goal state for the existing linger', () => {
    const finished = [todo('done', 'completed'), todo('skipped', 'cancelled')]

    expect(resolveTodoPresentation(finished, { continuation: continuation('active'), turnLive: true })).toEqual({
      kind: 'finished',
      remaining: 0
    })
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
