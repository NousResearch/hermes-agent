import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $currentUsage } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './use-message-stream'

let handleEvent: (event: RpcEvent) => void = () => undefined

function MessageStreamHarness({ activeSessionId = 'session-1' }: { activeSessionId?: string }) {
  const activeSessionIdRef = useRef<string | null>(activeSessionId)
  const queryClientRef = useRef(new QueryClient())
  const statesRef = useRef(new Map<string, ClientSessionState>())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    updateSessionState: (sessionId, updater) => {
      const previous = statesRef.current.get(sessionId) ?? createClientSessionState(null)
      const next = updater(previous)
      statesRef.current.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

describe('useMessageStream token usage events', () => {
  beforeEach(() => {
    handleEvent = () => undefined
    $currentUsage.set({ calls: 0, input: 0, output: 0, total: 0 })
  })

  afterEach(() => {
    cleanup()
    $currentUsage.set({ calls: 0, input: 0, output: 0, total: 0 })
    vi.restoreAllMocks()
  })

  it('updates current usage from token.usage before message.complete', () => {
    render(<MessageStreamHarness />)

    act(() =>
      handleEvent({
        payload: {},
        session_id: 'session-1',
        type: 'message.start'
      } as RpcEvent)
    )

    act(() =>
      handleEvent({
        payload: {
          context_length: 131_072,
          context_pct: 49.9,
          context_tokens: 65_432,
          input_tokens: 1_200,
          output_tokens: 34,
          total_tokens: 1_234
        },
        session_id: 'session-1',
        type: 'token.usage'
      } as RpcEvent)
    )

    expect($currentUsage.get()).toMatchObject({
      context_max: 131_072,
      context_percent: 49.9,
      context_used: 65_432,
      input: 1_200,
      output: 34,
      total: 1_234
    })
  })

  it('ignores token.usage events for inactive sessions', () => {
    $currentUsage.set({ calls: 1, input: 10, output: 5, total: 15 })
    render(<MessageStreamHarness />)

    act(() =>
      handleEvent({
        payload: {
          context_length: 131_072,
          context_tokens: 70_000,
          input_tokens: 70_000,
          total_tokens: 70_000
        },
        session_id: 'session-2',
        type: 'token.usage'
      } as RpcEvent)
    )

    expect($currentUsage.get()).toEqual({ calls: 1, input: 10, output: 5, total: 15 })
  })
})
