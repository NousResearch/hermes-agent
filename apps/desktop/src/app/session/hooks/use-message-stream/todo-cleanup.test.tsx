import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { TodoItem } from '@/lib/todos'
import { $clarifyRequests, clearClarifyRequest } from '@/store/clarify'
import { $todosBySession, clearSessionTodos, setSessionTodos } from '@/store/todos'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const triggerHaptic = vi.hoisted(() => vi.fn())
const playCompletionSound = vi.hoisted(() => vi.fn())
const dispatchNativeNotification = vi.hoisted(() => vi.fn())

vi.mock('@/lib/completion-sound', () => ({ playCompletionSound }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic }))
vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification }))

const SID = 'session-1'
const todo = (id: string, status: TodoItem['status']): TodoItem => ({ content: `task ${id}`, id, status })

let handleEvent: ((event: RpcEvent) => void) | null = null
let latestSessionState: ClientSessionState | undefined

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
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
      latestSessionState = next

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
    latestSessionState = undefined
    triggerHaptic.mockClear()
    playCompletionSound.mockClear()
    dispatchNativeNotification.mockClear()
    clearClarifyRequest()
    clearSessionTodos(SID)
  })

  afterEach(() => {
    cleanup()
    clearClarifyRequest()
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

  it('uses the existing completion haptic only for the active conversation', async () => {
    await mountStream()

    complete()
    expect(playCompletionSound).toHaveBeenCalledOnce()
    expect(triggerHaptic).toHaveBeenCalledWith('streamDone')

    playCompletionSound.mockClear()
    triggerHaptic.mockClear()
    act(() => handleEvent!({ payload: { text: 'background done' }, session_id: 'session-2', type: 'message.complete' }))
    expect(playCompletionSound).toHaveBeenCalledOnce()
    expect(triggerHaptic).not.toHaveBeenCalled()
  })

  it('restores a replayed blocking prompt inline without repeating its native notification', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: {
          _replayed: true,
          choices: ['continue', 'stop'],
          question: 'Keep going?',
          request_id: 'clarify-1'
        },
        session_id: SID,
        type: 'clarify.request'
      })
    )

    expect($clarifyRequests.get()[SID]).toMatchObject({ question: 'Keep going?', requestId: 'clarify-1' })
    expect(latestSessionState?.needsInput).toBe(true)
    expect(dispatchNativeNotification).not.toHaveBeenCalled()
  })

  it('drops a still-active task list when the turn errors out', async () => {
    await mountStream()
    setSessionTodos(SID, [todo('a', 'in_progress')])

    act(() => handleEvent!({ payload: { message: 'boom' }, session_id: SID, type: 'error' }))

    expect($todosBySession.get()[SID]).toBeUndefined()
  })
})
