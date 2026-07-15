import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'
let handleEvent: ((event: RpcEvent) => void) | null = null
let states = new Map<string, ClientSessionState>()

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(states)
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const current = states.get(sessionId) ?? createClientSessionState()
      const next = updater(current)

      states.set(sessionId, next)

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

describe('live durable history refresh', () => {
  beforeEach(() => {
    handleEvent = null
    states = new Map()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('hydrates an idle session and preserves its local assistant error', async () => {
    states.set(
      SID,
      createClientSessionState(null, [
        { id: 'local-user', parts: [{ text: 'failed prompt', type: 'text' }], role: 'user' },
        { error: 'provider failed', id: 'local-error', parts: [], role: 'assistant' }
      ])
    )
    await mountStream()

    act(() =>
      handleEvent!({
        payload: {
          messages: [
            { content: 'external prompt', role: 'user' },
            { content: 'external reply', role: 'assistant' }
          ]
        },
        session_id: SID,
        type: 'session.history.updated'
      })
    )

    const messages = states.get(SID)!.messages
    expect(messages.slice(0, 2).map(chatMessageText)).toEqual(['external prompt', 'external reply'])
    expect(messages.at(-1)?.error).toBe('provider failed')
  })

  it('does not replace a session while a turn is active', async () => {
    const initial = createClientSessionState(null, [
      { id: 'live', parts: [{ text: 'live turn', type: 'text' }], role: 'user' }
    ])

    initial.busy = true
    states.set(SID, initial)
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { messages: [{ content: 'stale durable row', role: 'user' }] },
        session_id: SID,
        type: 'session.history.updated'
      })
    )

    expect(states.get(SID)!.messages).toBe(initial.messages)
  })
})
