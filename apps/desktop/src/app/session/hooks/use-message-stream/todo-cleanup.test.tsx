import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { TodoItem } from '@/lib/todos'
import { $todosBySession, clearSessionTodos, setSessionTodos } from '@/store/todos'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'
const todo = (id: string, status: TodoItem['status']): TodoItem => ({ content: `task ${id}`, id, status })

let handleEvent: ((event: RpcEvent) => void) | null = null
let stateByRuntimeId: Map<string, ClientSessionState>

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(stateByRuntimeId)
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
    stateByRuntimeId = new Map()
    clearSessionTodos(SID)
  })

  afterEach(() => {
    cleanup()
    clearSessionTodos(SID)
    vi.restoreAllMocks()
  })

  it('drops a still-active task list when the turn completes', async () => {
    await mountStream()
    setSessionTodos(SID, [todo('a', 'completed'), todo('b', 'in_progress')])

    complete()

    expect($todosBySession.get()[SID]).toBeUndefined()
  })

  it('keeps a finished list on completion so its linger shows the final checkmarks', async () => {
    await mountStream()
    setSessionTodos(SID, [todo('a', 'completed')])

    complete()

    // Not cleared immediately — the finished-list linger still owns it.
    expect($todosBySession.get()[SID]).toHaveLength(1)
  })

  it('drops a still-active task list when the turn errors out', async () => {
    await mountStream()
    setSessionTodos(SID, [todo('a', 'in_progress')])

    act(() => handleEvent!({ payload: { message: 'boom' }, session_id: SID, type: 'error' }))

    expect($todosBySession.get()[SID]).toBeUndefined()
  })

  it('treats message.complete with error status as a failed turn', async () => {
    await mountStream()

    act(() => handleEvent!({ payload: undefined, session_id: SID, type: 'message.start' }))
    act(() =>
      handleEvent!({
        payload: { status: 'error', text: 'Context length exceeded and cannot compress further.' },
        session_id: SID,
        type: 'message.complete'
      })
    )

    const state = stateByRuntimeId.get(SID)

    expect(state?.busy).toBe(false)
    expect(state?.awaitingResponse).toBe(false)
    expect(state?.turnStartedAt).toBeNull()
    expect(state?.messages.at(-1)?.pending).toBe(false)
    expect(state?.messages.at(-1)?.error).toBe('Context length exceeded and cannot compress further.')
  })
})
