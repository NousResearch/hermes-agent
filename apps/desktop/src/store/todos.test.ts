import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TodoItem } from '@/lib/todos'
import { deferred } from '@/test/deferred'

import {
  $todoContinuationsBySession,
  $todoRevisionsBySession,
  $todosBySession,
  applyTodoContinuationSnapshot,
  captureTodoWriteFence,
  clearActiveSessionTodos,
  clearAllSessionTodos,
  clearAllTodoContinuations,
  clearSessionTodos,
  clearTodoContinuation,
  releaseTodoHydrationToken,
  resolveTodoPresentation,
  restoreSessionTodosFromSnapshot,
  setSessionTodos,
  type TodoContinuationSnapshot,
  todosForHydration
} from './todos'

const todo = (id: string, status: TodoItem['status']): TodoItem => ({ content: `task ${id}`, id, status })

const beginDeferredHydration = (sid: string, read: Promise<null | TodoItem[]>) => {
  const token = (captureTodoWriteFence as (runtimeSessionId: string) => ReturnType<typeof captureTodoWriteFence>)(sid)

  return read.then(todos =>
    todos
      ? setSessionTodos(sid, todos, { ifUnchangedSince: token, preserved: true })
      : clearSessionTodos(sid, { ifUnchangedSince: token })
  )
}

interface TodoHydrationAuthorityStats {
  activeSessionCount: number
  activeTokenCount: number
  generation: number
  latestSessionCount: number
}

async function todoHydrationAuthorityStats(): Promise<TodoHydrationAuthorityStats> {
  const todoStore = await import('./todos')
  const inspect = Reflect.get(todoStore, '_todoHydrationAuthorityStatsForTests')

  expect(inspect).toBeTypeOf('function')

  return (inspect as () => TodoHydrationAuthorityStats)()
}

describe('overlapping Todo hydration ordering', () => {
  afterEach(() => {
    clearAllSessionTodos()
  })

  it('lets a later-started read replace an older read that resolves first', async () => {
    const olderRead = deferred<TodoItem[]>()
    const newerRead = deferred<TodoItem[]>()
    const olderHydration = beginDeferredHydration('runtime-overlap', olderRead.promise)
    const newerHydration = beginDeferredHydration('runtime-overlap', newerRead.promise)

    olderRead.resolve([todo('older', 'pending')])
    expect(await olderHydration).toBe(false)

    newerRead.resolve([todo('newer', 'pending')])
    expect(await newerHydration).toBe(true)
    expect($todosBySession.get()['runtime-overlap']).toEqual([todo('newer', 'pending')])
  })

  it('keeps a later-started read when it resolves before the older read', async () => {
    const olderRead = deferred<TodoItem[]>()
    const newerRead = deferred<TodoItem[]>()
    const olderHydration = beginDeferredHydration('runtime-reversed', olderRead.promise)
    const newerHydration = beginDeferredHydration('runtime-reversed', newerRead.promise)

    newerRead.resolve([todo('newer', 'pending')])
    expect(await newerHydration).toBe(true)

    olderRead.resolve([todo('older', 'pending')])
    expect(await olderHydration).toBe(false)
    expect($todosBySession.get()['runtime-reversed']).toEqual([todo('newer', 'pending')])
  })

  it('keeps a later-started authoritative clear after an older list resolves first', async () => {
    const olderRead = deferred<TodoItem[]>()
    const newerRead = deferred<null>()
    const olderHydration = beginDeferredHydration('runtime-clear', olderRead.promise)
    const newerHydration = beginDeferredHydration('runtime-clear', newerRead.promise)

    olderRead.resolve([todo('older', 'pending')])
    expect(await olderHydration).toBe(false)

    newerRead.resolve(null)
    expect(await newerHydration).toBe(true)
    expect($todosBySession.get()['runtime-clear']).toBeUndefined()
  })

  it.each(['clear', 'set'] as const)('invalidates both hydration reads after an intervening live %s', async write => {
    const olderRead = deferred<TodoItem[]>()
    const newerRead = deferred<TodoItem[]>()
    const olderHydration = beginDeferredHydration('runtime-live', olderRead.promise)
    const newerHydration = beginDeferredHydration('runtime-live', newerRead.promise)

    if (write === 'set') {
      setSessionTodos('runtime-live', [todo('live', 'in_progress')])
    } else {
      clearSessionTodos('runtime-live')
    }

    newerRead.resolve([todo('newer persisted', 'pending')])
    olderRead.resolve([todo('older persisted', 'pending')])

    expect(await newerHydration).toBe(false)
    expect(await olderHydration).toBe(false)
    expect($todosBySession.get()['runtime-live']).toEqual(write === 'set' ? [todo('live', 'in_progress')] : undefined)
  })

  it("does not invalidate a runtime's hydration after another runtime receives a live write", async () => {
    const targetRead = deferred<TodoItem[]>()
    const targetHydration = beginDeferredHydration('runtime-target', targetRead.promise)

    setSessionTodos('runtime-other', [todo('other live', 'in_progress')])
    targetRead.resolve([todo('target persisted', 'pending')])

    expect(await targetHydration).toBe(true)
    expect($todosBySession.get()['runtime-target']).toEqual([todo('target persisted', 'pending')])
  })

  it('invalidates every older hydration token after a global clear', async () => {
    const firstRead = deferred<TodoItem[]>()
    const secondRead = deferred<TodoItem[]>()
    const firstHydration = beginDeferredHydration('runtime-global-a', firstRead.promise)
    const secondHydration = beginDeferredHydration('runtime-global-b', secondRead.promise)

    clearAllSessionTodos()
    firstRead.resolve([todo('first persisted', 'pending')])
    secondRead.resolve([todo('second persisted', 'pending')])

    expect(await firstHydration).toBe(false)
    expect(await secondHydration).toBe(false)
    expect($todosBySession.get()['runtime-global-a']).toBeUndefined()
    expect($todosBySession.get()['runtime-global-b']).toBeUndefined()
  })

  it('binds a provisional cold-resume token to only its returned runtime identity', () => {
    const token = captureTodoWriteFence()

    expect(
      setSessionTodos('runtime-cold-returned', [todo('returned', 'pending')], {
        ifUnchangedSince: token,
        preserved: true
      })
    ).toBe(true)
    expect(
      setSessionTodos('runtime-unrelated', [todo('unrelated', 'pending')], {
        ifUnchangedSince: token,
        preserved: true
      })
    ).toBe(false)
    expect($todosBySession.get()['runtime-unrelated']).toBeUndefined()
  })

  it('shares one generation across bounded retries so a newer operation remains authoritative', async () => {
    const failedAttempt = deferred<TodoItem[]>()
    const retryAttempt = deferred<TodoItem[]>()
    const newerRead = deferred<TodoItem[]>()

    const token = (captureTodoWriteFence as (runtimeSessionId: string) => ReturnType<typeof captureTodoWriteFence>)(
      'runtime-retry'
    )

    const retriedHydration = (async () => {
      for (const read of [failedAttempt.promise, retryAttempt.promise]) {
        try {
          return setSessionTodos('runtime-retry', await read, { ifUnchangedSince: token, preserved: true })
        } catch {
          // One logical hydration operation retains its start order across a bounded retry.
        }
      }

      return false
    })()

    const newerHydration = beginDeferredHydration('runtime-retry', newerRead.promise)

    newerRead.resolve([todo('newer', 'pending')])
    expect(await newerHydration).toBe(true)
    failedAttempt.reject(new Error('transient read failure'))
    retryAttempt.resolve([todo('older retry', 'pending')])

    expect(await retriedHydration).toBe(false)
    expect($todosBySession.get()['runtime-retry']).toEqual([todo('newer', 'pending')])
  })

  it.each(['older-first', 'newer-first'] as const)(
    'orders two provisional cold hydrations that bind to the same runtime (%s completion)',
    async completionOrder => {
      const todoStore = await import('./todos')

      const bindTodoHydrationToken = (
        todoStore as typeof todoStore & {
          bindTodoHydrationToken?: (
            token: ReturnType<typeof captureTodoWriteFence>,
            runtimeSessionId: string
          ) => boolean
        }
      ).bindTodoHydrationToken

      if (!bindTodoHydrationToken) {
        expect(bindTodoHydrationToken).toBeTypeOf('function')

        return
      }

      const olderRead = deferred<TodoItem[]>()
      const newerRead = deferred<TodoItem[]>()
      const olderToken = captureTodoWriteFence()
      const newerToken = captureTodoWriteFence()

      expect(bindTodoHydrationToken(olderToken, 'runtime-cold-shared')).toBe(true)
      expect(bindTodoHydrationToken(newerToken, 'runtime-cold-shared')).toBe(true)

      const olderHydration = olderRead.promise.then(todos =>
        setSessionTodos('runtime-cold-shared', todos, { ifUnchangedSince: olderToken, preserved: true })
      )

      const newerHydration = newerRead.promise.then(todos =>
        setSessionTodos('runtime-cold-shared', todos, { ifUnchangedSince: newerToken, preserved: true })
      )

      if (completionOrder === 'older-first') {
        olderRead.resolve([todo('older cold', 'pending')])
        expect(await olderHydration).toBe(false)
        newerRead.resolve([todo('newer cold', 'pending')])
        expect(await newerHydration).toBe(true)
      } else {
        newerRead.resolve([todo('newer cold', 'pending')])
        expect(await newerHydration).toBe(true)
        olderRead.resolve([todo('older cold', 'pending')])
        expect(await olderHydration).toBe(false)
      }

      expect($todosBySession.get()['runtime-cold-shared']).toEqual([todo('newer cold', 'pending')])
    }
  )

  it('retains a newer-started high-water mark until an older overlapping hydration refuses', async () => {
    const olderToken = captureTodoWriteFence('runtime-drain')
    const newerToken = captureTodoWriteFence('runtime-drain')

    expect(await todoHydrationAuthorityStats()).toMatchObject({
      activeSessionCount: 1,
      activeTokenCount: 2,
      latestSessionCount: 1
    })

    releaseTodoHydrationToken(newerToken)
    expect(await todoHydrationAuthorityStats()).toMatchObject({
      activeSessionCount: 1,
      activeTokenCount: 1,
      latestSessionCount: 1
    })

    expect(
      setSessionTodos('runtime-drain', [todo('older drain', 'pending')], {
        ifUnchangedSince: olderToken,
        preserved: true
      })
    ).toBe(false)
    expect($todosBySession.get()['runtime-drain']).toBeUndefined()
    expect(await todoHydrationAuthorityStats()).toMatchObject({
      activeSessionCount: 0,
      activeTokenCount: 0,
      latestSessionCount: 0
    })
  })

  it('retires exact-session hydration authority and rejects its pending token without poisoning a fresh lifecycle', async () => {
    const staleToken = captureTodoWriteFence('runtime-cleanup')

    expect(await todoHydrationAuthorityStats()).toMatchObject({
      activeSessionCount: 1,
      activeTokenCount: 1,
      latestSessionCount: 1
    })

    clearSessionTodos('runtime-cleanup')
    expect(await todoHydrationAuthorityStats()).toMatchObject({
      activeSessionCount: 0,
      activeTokenCount: 0,
      latestSessionCount: 0
    })
    expect(
      setSessionTodos('runtime-cleanup', [todo('stale lifecycle', 'pending')], {
        ifUnchangedSince: staleToken,
        preserved: true
      })
    ).toBe(false)

    const freshToken = captureTodoWriteFence('runtime-cleanup')

    expect(
      setSessionTodos('runtime-cleanup', [todo('fresh lifecycle', 'pending')], {
        ifUnchangedSince: freshToken,
        preserved: true
      })
    ).toBe(true)
    expect($todosBySession.get()['runtime-cleanup']).toEqual([todo('fresh lifecycle', 'pending')])
    expect(await todoHydrationAuthorityStats()).toMatchObject({
      activeSessionCount: 0,
      activeTokenCount: 0,
      latestSessionCount: 0
    })
  })

  it('globally retires bound and provisional authority without retaining runtime identities', async () => {
    const boundTokens = Array.from({ length: 64 }, (_, index) => captureTodoWriteFence(`runtime-global-${index}`))
    const provisionalToken = captureTodoWriteFence()

    expect(await todoHydrationAuthorityStats()).toMatchObject({
      activeSessionCount: 64,
      activeTokenCount: 65,
      latestSessionCount: 64
    })

    clearAllSessionTodos()
    expect(await todoHydrationAuthorityStats()).toEqual({
      activeSessionCount: 0,
      activeTokenCount: 0,
      generation: 0,
      latestSessionCount: 0
    })

    for (const [index, token] of boundTokens.entries()) {
      expect(
        setSessionTodos(`runtime-global-${index}`, [todo(`stale global ${index}`, 'pending')], {
          ifUnchangedSince: token,
          preserved: true
        })
      ).toBe(false)
    }
    expect(
      setSessionTodos('runtime-global-provisional', [todo('stale provisional', 'pending')], {
        ifUnchangedSince: provisionalToken,
        preserved: true
      })
    ).toBe(false)
    expect(await todoHydrationAuthorityStats()).toEqual({
      activeSessionCount: 0,
      activeTokenCount: 0,
      generation: 0,
      latestSessionCount: 0
    })
  })
})

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

  it('accepts a newer hydration after the older finished-list linger fires while its read is pending', async () => {
    setSessionTodos('s1', [todo('finished', 'completed')])
    const read = deferred<TodoItem[]>()
    const hydration = beginDeferredHydration('s1', read.promise)

    vi.advanceTimersByTime(5_000)
    expect($todosBySession.get().s1).toBeUndefined()

    read.resolve([todo('hydrated', 'pending')])

    expect(await hydration).toBe(true)
    expect($todosBySession.get().s1).toEqual([todo('hydrated', 'pending')])
  })

  it('does not let an older finished-list linger remove a newer hydration that publishes first', async () => {
    setSessionTodos('s1', [todo('finished', 'completed')])
    const read = deferred<TodoItem[]>()
    const hydration = beginDeferredHydration('s1', read.promise)

    read.resolve([todo('hydrated', 'pending')])
    expect(await hydration).toBe(true)

    vi.advanceTimersByTime(60_000)

    expect($todosBySession.get().s1).toEqual([todo('hydrated', 'pending')])
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
