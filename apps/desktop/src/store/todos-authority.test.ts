import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { TodoSnapshot } from '@/lib/todos'

// Isolate the feature store from the wide session-store graph (which drags in
// @tabler/icons — missing from the canonical node_modules this worktree
// borrows). todos.ts only needs $sessions/lineageAliases/$sessionStates.
vi.mock('@/store/session', () => ({
  $sessions: atom([]),
  lineageAliases: (id: string) => [id]
}))
vi.mock('@/store/session-states', () => ({
  $sessionStates: atom({})
}))

const snap = (over: Partial<TodoSnapshot> = {}): TodoSnapshot => ({
  generation: 1,
  revision: 1,
  session_id: 's1',
  todos: [
    { content: 'Task one', id: 'a', status: 'in_progress' },
    { content: 'Task two', id: 'b', status: 'pending' }
  ],
  ...over
})

const todo = (id: string, status: TodoSnapshot['todos'][number]['status']): TodoSnapshot['todos'][number] => ({
  content: `task ${id}`,
  id,
  status
})

afterEach(async () => {
  const todos = await import('./todos')
  todos.clearSessionTodos('s1')
  todos.clearSessionTodos('s2')
})

describe('setSessionTodoSnapshot (authoritative cache)', () => {
  it('publishes display list and authority together in one batch', async () => {
    const { $sessionTodoSnapshots, $todosBySession, setSessionTodoSnapshot, todoSnapshotAuthority } = await import('./todos')

    setSessionTodoSnapshot(snap({ revision: 5 }))

    expect($todosBySession.get().s1).toHaveLength(2)
    expect(todoSnapshotAuthority('s1')).toEqual({ generation: 1, revision: 5 })
    expect($sessionTodoSnapshots.get().s1).not.toBeNull()
  })

  it('display list derives from the snapshot todos', async () => {
    const { $todosBySession, setSessionTodoSnapshot } = await import('./todos')

    setSessionTodoSnapshot(snap({ todos: [{ content: 'Only', id: 'x', status: 'completed' }] }))

    expect($todosBySession.get().s1).toEqual([{ content: 'Only', id: 'x', status: 'completed' }])
  })

  it('does not auto-clear an authoritative terminal snapshot (finished linger is display-only)', async () => {
    vi.useFakeTimers()
    try {
      const { $todosBySession, clearSessionTodos, setSessionTodoSnapshot } = await import('./todos')

      setSessionTodoSnapshot(snap({ todos: [{ content: 'Done', id: 'a', status: 'completed' }] }))

      vi.advanceTimersByTime(60_000)

      // The authoritative snapshot stays so the user keeps Reopen controls.
      expect($todosBySession.get().s1).toHaveLength(1)
      clearSessionTodos('s1')
    } finally {
      vi.useRealTimers()
    }
  })

  it('higher generation wins; lower generation is ignored', async () => {
    const { setSessionTodoSnapshot, $todosBySession, todoSnapshotAuthority } = await import('./todos')

    setSessionTodoSnapshot(snap({ generation: 10, revision: 9 }))

    setSessionTodoSnapshot(snap({ generation: 5, revision: 99, todos: [{ content: 'Old', id: 'old', status: 'pending' }] }))

    expect(todoSnapshotAuthority('s1')!.generation).toBe(10)
    expect($todosBySession.get().s1!.some(t => t.id === 'old')).toBe(false)

    setSessionTodoSnapshot(snap({ generation: 11, revision: 10, todos: [{ content: 'New', id: 'new', status: 'completed' }] }))

    expect(todoSnapshotAuthority('s1')!.generation).toBe(11)
    expect($todosBySession.get().s1!.some(t => t.id === 'new')).toBe(true)
  })

  it('equal generation republishes the authoritative plan (exact response for optimistic rollback)', async () => {
    const { setSessionTodoSnapshot, $todosBySession, todoSnapshotAuthority } = await import('./todos')

    const plan = [{ content: 'Same plan', id: 'z', status: 'pending' }] as TodoSnapshot['todos']
    setSessionTodoSnapshot(snap({ generation: 7, revision: 4, todos: plan }))

    // An equal-generation response is the same moment of truth (the store
    // bumps generation on every mutation), so it may republish — this is
    // what an optimistic update rolls back/forward onto. It is the response
    // to OUR request, not an arbitrary divergent list, so identity of
    // authority (generation 7, revision 4) is preserved.
    setSessionTodoSnapshot(snap({ generation: 7, revision: 4, todos: plan }))

    expect(todoSnapshotAuthority('s1')!.revision).toBe(4)
    expect($todosBySession.get().s1).toEqual(plan)
  })

  it('without a revision there is no mutation authority (legacy hydration path)', async () => {
    const { setSessionTodos, todoSnapshotAuthority } = await import('./todos')

    setSessionTodos('s1', snap().todos)

    expect(todoSnapshotAuthority('s1')).toBeNull()
  })

  it('clearActiveSessionTodos preserves an authoritative ACTIVE snapshot mid-turn', async () => {
    const { clearActiveSessionTodos, setSessionTodoSnapshot, $todosBySession } = await import('./todos')

    // clearActiveSessionTodos drops active lists only in the legacy path.
    // With authority present, an in-flight turn's plan must survive a
    // turn-end race (the authoritative snapshot outranks the heuristic).
    setSessionTodoSnapshot(snap({ todos: [todo('a', 'in_progress'), todo('b', 'pending')] }))

    clearActiveSessionTodos('s1')

    expect($todosBySession.get().s1).toHaveLength(2)
  })

  it('clearActiveSessionTodos still drops a legacy ACTIVE list without authority', async () => {
    const { clearActiveSessionTodos, setSessionTodos, $todosBySession } = await import('./todos')

    setSessionTodos('s1', [{ content: 'task a', id: 'a', status: 'in_progress' }])

    clearActiveSessionTodos('s1')

    expect($todosBySession.get().s1).toBeUndefined()
  })
})

describe('legacy display-list path still works without authority', () => {
  it('setSessionTodos without revision clears mutation authority but keeps display', async () => {
    const { setSessionTodos, $todosBySession, todoSnapshotAuthority } = await import('./todos')

    setSessionTodos('s1', snap().todos)

    expect($todosBySession.get().s1).toHaveLength(2)
    expect(todoSnapshotAuthority('s1')).toBeNull()
  })

  it('a later authoritative snapshot upgrades authority', async () => {
    const { setSessionTodos, setSessionTodoSnapshot, todoSnapshotAuthority } = await import('./todos')

    setSessionTodos('s1', snap().todos)
    expect(todoSnapshotAuthority('s1')).toBeNull()

    setSessionTodoSnapshot(snap({ revision: 3 }))

    expect(todoSnapshotAuthority('s1')).toEqual({ generation: 1, revision: 3 })
  })
})

describe('apply todo.updated / tool event ordering', () => {
  it('late lower-generation response cannot overwrite a newer snapshot', async () => {
    const { setSessionTodoSnapshot, $todosBySession, todoSnapshotAuthority } = await import('./todos')

    setSessionTodoSnapshot(snap({ generation: 20, revision: 15 }))

    setSessionTodoSnapshot(snap({ generation: 9, revision: 99 }))

    expect(todoSnapshotAuthority('s1')!.generation).toBe(20)
    expect($todosBySession.get().s1!.some(t => t.id === 'old')).toBe(false)
  })

  it('a plain tool list after a snapshot drops authority but retains the visible list', async () => {
    const { setSessionTodoSnapshot, setSessionTodos, $todosBySession, todoSnapshotAuthority } = await import('./todos')

    setSessionTodoSnapshot(snap({ revision: 8 }))

    // A legacy hydration / display-only event after an authoritative one
    // downgrades mutation authority but retains the visible list.
    setSessionTodos('s1', [{ content: 'task a', id: 'a', status: 'completed' }])

    expect(todoSnapshotAuthority('s1')).toBeNull()
    expect($todosBySession.get().s1).toHaveLength(1)
  })
})