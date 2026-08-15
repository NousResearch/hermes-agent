import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $providerWaitSessions } from '@/store/provider-wait'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'
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

function emit(type: RpcEvent['type'], payload: RpcEvent['payload'] = {}) {
  act(() => handleEvent!({ payload, session_id: SID, type }))
}

describe('provider wait visibility', () => {
  beforeEach(async () => {
    handleEvent = null
    $providerWaitSessions.set({})
    render(<Harness />)
    await waitFor(() => expect(handleEvent).not.toBeNull())
  })

  afterEach(() => {
    cleanup()
    $providerWaitSessions.set({})
    vi.restoreAllMocks()
  })

  it('surfaces explained waits but ignores generic spinner rewrites', () => {
    emit('thinking.delta', { text: '⏳ waiting on local-model — 30s with no output yet' })
    expect($providerWaitSessions.get()).toEqual({
      [SID]: '⏳ waiting on local-model — 30s with no output yet'
    })

    emit('thinking.delta', { text: '◉_◉ cogitating...' })
    expect($providerWaitSessions.get()).toEqual({})
  })

  it.each(['message.delta', 'reasoning.delta', 'tool.start', 'message.complete', 'error'] as const)(
    'clears the wait when %s proves the turn progressed or ended',
    type => {
      emit('thinking.delta', { text: '⚠ no output from provider for 900s — reconnecting...' })
      emit(type, type === 'tool.start' ? { name: 'terminal', tool_id: 'tool-1' } : { text: 'progress' })

      expect($providerWaitSessions.get()).toEqual({})
    }
  )
})
