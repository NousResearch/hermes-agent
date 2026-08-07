import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import type { ChatMessagePart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'
let handleEvent: ((event: RpcEvent) => void) | null = null
let sessionStateByRuntimeId: Map<string, ClientSessionState>
let reasoningSnapshots: ChatMessagePart[]

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(sessionStateByRuntimeId)
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

      const message = next.messages.find(item => item.id === next.streamId)
      const reasoning = message?.parts.find(part => part.type === 'reasoning')

      if (reasoning) {
        reasoningSnapshots.push(reasoning)
      }

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

function emit(type: RpcEvent['type'], payload: RpcEvent['payload'] = {}) {
  act(() => handleEvent!({ payload, session_id: SID, type }))
}

function streamedParts(): ChatMessagePart[] {
  const state = sessionStateByRuntimeId.get(SID)
  const message = state?.messages.find(item => item.id === state.streamId)

  return message?.parts ?? []
}

describe('useMessageStream reasoning.available fallback', () => {
  beforeEach(() => {
    handleEvent = null
    sessionStateByRuntimeId = new Map()
    reasoningSnapshots = []
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('preserves streamed reasoning when the late available fallback arrives', async () => {
    await mountStream()
    const streamedReasoning = 'streamed reasoning '.repeat(40)

    emit('reasoning.delta', { text: streamedReasoning })
    emit('reasoning.available', { text: streamedReasoning.slice(0, 500) })

    const reasoningParts = streamedParts().filter(part => part.type === 'reasoning')

    expect(reasoningParts).toHaveLength(1)
    expect(reasoningParts[0]).toMatchObject({ text: streamedReasoning })
    expect(reasoningSnapshots.at(-1)).toBe(reasoningSnapshots.at(-2))
  })

  it('inserts late available reasoning before assistant text when no delta streamed', async () => {
    await mountStream()

    emit('message.delta', { text: 'Final answer.' })
    emit('reasoning.available', { text: 'Late reasoning.' })

    expect(streamedParts()).toMatchObject([
      { text: 'Late reasoning.', type: 'reasoning' },
      { text: 'Final answer.', type: 'text' }
    ])
  })
})
