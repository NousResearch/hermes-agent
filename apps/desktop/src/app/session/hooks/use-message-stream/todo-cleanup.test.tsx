import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { TodoItem } from '@/lib/todos'
import {
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  flushQueuedPromptMutations,
  getQueuedPrompts
} from '@/store/composer-queue'
import { $todosBySession, clearSessionTodos, setSessionTodos } from '@/store/todos'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'
const todo = (id: string, status: TodoItem['status']): TodoItem => ({ content: `task ${id}`, id, status })

let handleEvent: ((event: RpcEvent) => void) | null = null

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
    clearSessionTodos(SID)
    window.localStorage.removeItem('hermes.desktop.composerQueue.v1')
    $queuedPromptsBySession.set({})
  })

  afterEach(async () => {
    cleanup()
    clearSessionTodos(SID)
    await flushQueuedPromptMutations()
    window.localStorage.removeItem('hermes.desktop.composerQueue.v1')
    $queuedPromptsBySession.set({})
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

  it('releases a locally retained queue entry when its backend turn completes', async () => {
    await mountStream()
    const entry = enqueueQueuedPrompt('stored-session-1', {
      text: 'survive until completion',
      attachments: []
    })

    act(() =>
      handleEvent!({
        payload: { text: 'done', submission_id: entry!.id },
        session_id: SID,
        type: 'message.complete'
      })
    )

    expect(getQueuedPrompts('stored-session-1')).toEqual([])
  })

  it('removes only the identified queue entry on a terminal backend refusal', async () => {
    await mountStream()
    const refused = enqueueQueuedPrompt('stored-session-1', { text: 'invalid context', attachments: [] })
    const survivor = enqueueQueuedPrompt('stored-session-1', { text: 'P2', attachments: [] })

    act(() =>
      handleEvent!({
        payload: {
          message: 'Context injection refused.',
          submission_id: refused!.id,
          terminal: true
        },
        session_id: SID,
        type: 'error'
      })
    )
    await flushQueuedPromptMutations()

    expect(getQueuedPrompts('stored-session-1').map(entry => entry.id)).toEqual([survivor!.id])
  })
})
