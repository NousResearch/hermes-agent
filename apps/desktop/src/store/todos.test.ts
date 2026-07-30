import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TodoItem } from '@/lib/todos'

import {
  $todosBySession,
  applyHydratedSessionTodos,
  clearSessionTodos,
  refreshSessionTodos,
  setSessionTodos,
  todosForHydration
} from './todos'

const todo = (id: string, status: TodoItem['status']): TodoItem => ({ content: `task ${id}`, id, status })

describe('setSessionTodos retention', () => {
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

  it('keeps a finished list until it is replaced or explicitly cleared', () => {
    setSessionTodos('s1', [todo('a', 'completed'), todo('b', 'cancelled')])

    expect($todosBySession.get().s1).toHaveLength(2)
    vi.advanceTimersByTime(60_000)

    expect($todosBySession.get().s1).toHaveLength(2)
  })

  it('replaces a finished list when a new active list arrives', () => {
    setSessionTodos('s1', [todo('a', 'completed')])
    setSessionTodos('s1', [todo('b', 'pending')])

    expect($todosBySession.get().s1).toEqual([todo('b', 'pending')])
  })
})

describe('todosForHydration (durable plan restore)', () => {
  it('restores an active list across reload and compaction', () => {
    const active = [todo('a', 'completed'), todo('b', 'in_progress')]
    const pending = [todo('a', 'pending')]

    expect(todosForHydration(active)).toEqual(active)
    expect(todosForHydration(pending)).toEqual(pending)
  })

  it('restores a finished list', () => {
    const finished = [todo('a', 'completed'), todo('b', 'cancelled')]

    expect(todosForHydration(finished)).toEqual(finished)
  })

  it('returns null when there is nothing stored', () => {
    expect(todosForHydration(null)).toBeNull()
  })
})

describe('applyHydratedSessionTodos', () => {
  afterEach(() => clearSessionTodos('s1'))

  it('rejects a stale response when a newer live plan arrived during refresh', () => {
    const before = [todo('before', 'in_progress')]
    const live = [todo('live', 'in_progress')]

    setSessionTodos('s1', before)
    setSessionTodos('s1', live)

    expect(applyHydratedSessionTodos('s1', before, [todo('stale', 'pending')])).toBe('superseded')
    expect($todosBySession.get().s1).toBe(live)
  })
})

describe('refreshSessionTodos', () => {
  afterEach(() => clearSessionTodos('s1'))

  it('restores persisted in-progress state without executing or completing it', async () => {
    const restored = [todo('interrupted', 'in_progress')]

    expect(
      await refreshSessionTodos(
        's1',
        async () => restored,
        () => true
      )
    ).toBe('applied')
    expect($todosBySession.get().s1).toBe(restored)
    expect($todosBySession.get().s1?.[0]?.status).toBe('in_progress')
  })

  it('drops the response when the user switches sessions while it is pending', async () => {
    expect(
      await refreshSessionTodos(
        's1',
        async () => [todo('stale', 'pending')],
        () => false
      )
    ).toBe('superseded')
    expect($todosBySession.get().s1).toBeUndefined()
  })
})
