import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'
const hydrateFromStoredSession = vi.fn(async () => undefined)

let handleEvent: ((event: RpcEvent) => void) | null = null
let sessionState: ClientSessionState | undefined

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession,
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
      const next = updater(current)
      sessionStateByRuntimeIdRef.current.set(sessionId, next)
      sessionState = next

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

describe('useMessageStream error completion', () => {
  beforeEach(() => {
    handleEvent = null
    sessionState = undefined
    hydrateFromStoredSession.mockClear()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('preserves a gateway-reported error even when its text is not error-shaped', async () => {
    render(<Harness />)
    await waitFor(() => expect(handleEvent).not.toBeNull())

    const message = 'The requested model is not supported for this Copilot subscription.'
    act(() =>
      handleEvent!({
        payload: { status: 'error', text: message },
        session_id: SID,
        type: 'message.complete'
      })
    )

    expect(sessionState?.messages).toEqual([
      expect.objectContaining({ error: message, parts: [], role: 'assistant' })
    ])
    expect(hydrateFromStoredSession).not.toHaveBeenCalled()
  })
})
