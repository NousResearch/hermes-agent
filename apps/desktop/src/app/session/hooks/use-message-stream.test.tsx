import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $todosBySession, clearSessionTodos } from '@/store/todos'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './use-message-stream'

let handleEvent: ((event: RpcEvent) => void) | null = null

function MessageStreamHarness({ onEvent }: { onEvent: (handler: (event: RpcEvent) => void) => void }) {
  const activeSessionIdRef = useRef<string | null>('session-1')
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(),
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
    onEvent(stream.handleGatewayEvent)
  }, [onEvent, stream.handleGatewayEvent])

  return null
}

describe('useMessageStream todo status cleanup', () => {
  beforeEach(() => {
    handleEvent = null
    clearSessionTodos('session-1')
  })

  afterEach(() => {
    cleanup()
    clearSessionTodos('session-1')
    vi.restoreAllMocks()
  })

  it('clears stale active todos when the turn completes', async () => {
    render(
      <MessageStreamHarness
        onEvent={handler => {
          handleEvent = handler
        }}
      />
    )

    await waitFor(() => expect(handleEvent).not.toBeNull())

    act(() =>
      handleEvent!({
        payload: {
          name: 'todo',
          todos: [
            { content: 'done', id: 'done', status: 'completed' },
            { content: 'last step', id: 'last', status: 'in_progress' }
          ],
          tool_id: 'todo-live'
        },
        session_id: 'session-1',
        type: 'tool.progress'
      })
    )

    expect($todosBySession.get()['session-1']).toHaveLength(2)

    act(() =>
      handleEvent!({
        payload: { text: 'All done.' },
        session_id: 'session-1',
        type: 'message.complete'
      })
    )

    expect($todosBySession.get()['session-1']).toBeUndefined()
  })
})
