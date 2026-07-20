import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { TodoItem } from '@/lib/todos'
import {
  $todoHistoryBySession,
  $todosBySession,
  clearSessionTodoHistory,
  clearSessionTodos,
  setSessionTodos
} from '@/store/todos'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'
const todo = (id: string, status: TodoItem['status']): TodoItem => ({ content: `task ${id}`, id, status })

let handleEvent: ((event: RpcEvent) => void) | null = null
let sessionStates = new Map<string, ClientSessionState>()

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(sessionStates)
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
      const next = updater(current)
      sessionStateByRuntimeIdRef.current.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

const complete = () => act(() => handleEvent!({ payload: { text: 'done' }, session_id: SID, type: 'message.complete' }))

describe('useMessageStream turn-end todo cleanup', () => {
  beforeEach(() => {
    handleEvent = null
    sessionStates = new Map()
    clearSessionTodoHistory(SID)
    clearSessionTodos(SID)
  })

  afterEach(() => {
    cleanup()
    clearSessionTodoHistory(SID)
    clearSessionTodos(SID)
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('drops a still-active task list when the turn completes', async () => {
    await mountStream()
    setSessionTodos(SID, [todo('a', 'completed'), todo('b', 'in_progress')])

    complete()

    expect($todosBySession.get()[SID]).toBeUndefined()
  })

  it('does not touch task history when a text delta streams through a 1000-message session', async () => {
    await mountStream()
    sessionStates.set(SID, {
      ...createClientSessionState(),
      messages: Array.from({ length: 1_000 }, (_, index) => ({
        id: `message-${index}`,
        parts: [{ text: 'unchanged', type: 'text' as const }],
        role: index % 2 === 0 ? ('user' as const) : ('assistant' as const)
      }))
    })
    const history = [{ id: 'old', state: 'completed' as const, todos: [todo('old', 'completed')] }]
    $todoHistoryBySession.set({ [SID]: history })
    const before = $todoHistoryBySession.get()

    act(() => handleEvent!({ payload: { text: 'x' }, session_id: SID, type: 'message.delta' }))

    expect($todoHistoryBySession.get()).toBe(before)
    expect($todoHistoryBySession.get()[SID]).toBe(history)
  })

  it('updates live todos on the todo event and finalizes one snapshot at message.complete', async () => {
    await mountStream()
    const before = $todoHistoryBySession.get()

    act(() =>
      handleEvent!({
        payload: { name: 'todo', todos: [todo('a', 'in_progress')] },
        session_id: SID,
        type: 'tool.progress'
      })
    )

    expect($todosBySession.get()[SID]).toEqual([todo('a', 'in_progress')])
    expect($todoHistoryBySession.get()).toBe(before)

    complete()

    expect($todoHistoryBySession.get()[SID]).toMatchObject([{ state: 'unfinished', todos: [todo('a', 'in_progress')] }])
  })

  it('keeps a finished list on completion so its linger shows the final checkmarks', async () => {
    await mountStream()
    setSessionTodos(SID, [todo('a', 'completed')])

    complete()

    // Not cleared immediately — the finished-list linger still owns it.
    expect($todosBySession.get()[SID]).toHaveLength(1)
  })

  it('does not let a todo-free turn replace the prior turn snapshot during its visual linger', async () => {
    await mountStream()
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'))
    $todoHistoryBySession.set({
      [SID]: [{ id: 'older', state: 'completed', timestamp: 1, todos: [todo('older', 'completed')] }]
    })

    act(() =>
      handleEvent!({
        payload: { name: 'todo', todos: [todo('a', 'completed')] },
        session_id: SID,
        type: 'tool.complete'
      })
    )
    complete()
    const afterTurnA = $todoHistoryBySession.get()[SID]

    expect(afterTurnA.map(snapshot => snapshot.id)).toEqual(['assistant-stream-1767225600000', 'older'])
    expect(afterTurnA[0]?.timestamp).toBe(1_767_225_600)
    expect($todosBySession.get()[SID]).toEqual([todo('a', 'completed')])

    vi.setSystemTime(new Date('2026-01-01T00:00:02Z'))
    act(() => handleEvent!({ payload: {}, session_id: SID, type: 'message.start' }))
    complete()

    expect($todoHistoryBySession.get()[SID]).toEqual(afterTurnA)
    expect($todosBySession.get()[SID]).toEqual([todo('a', 'completed')])
    vi.useRealTimers()
  })

  it('drops a still-active task list when the turn errors out', async () => {
    await mountStream()
    setSessionTodos(SID, [todo('a', 'in_progress')])

    act(() => handleEvent!({ payload: { message: 'boom' }, session_id: SID, type: 'error' }))

    expect($todosBySession.get()[SID]).toBeUndefined()
  })

  it('commits a finished todo turn to history on error while keeping its visual linger', async () => {
    await mountStream()
    act(() =>
      handleEvent!({
        payload: { name: 'todo', todos: [todo('a', 'completed')] },
        session_id: SID,
        type: 'tool.complete'
      })
    )

    act(() => handleEvent!({ payload: { message: 'boom' }, session_id: SID, type: 'error' }))

    // Finished list still lingers visually, but the plan is now in history so it
    // stays reachable and matches what a later transcript rebuild reconstructs.
    expect($todosBySession.get()[SID]).toEqual([todo('a', 'completed')])
    expect($todoHistoryBySession.get()[SID]).toMatchObject([{ state: 'completed', todos: [todo('a', 'completed')] }])

    // The error already consumed turn ownership, so a trailing complete is a no-op.
    complete()

    expect($todoHistoryBySession.get()[SID]).toMatchObject([{ state: 'completed', todos: [todo('a', 'completed')] }])
  })
})
