import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $clarifyRequests } from '@/store/clarify'
import { $activeSessionId } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SESSION_ID = 'runtime-session-1'
let handleEvent: ((event: RpcEvent) => void) | null = null
let states = new Map<string, ClientSessionState>()

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SESSION_ID)
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

function emit(type: RpcEvent['type'], payload: RpcEvent['payload']) {
  act(() => handleEvent!({ payload, session_id: SESSION_ID, type }))
}

function clarifyRequest() {
  emit('clarify.request', {
    choices: ['Use this chat', 'Start a new one'],
    question: 'Where should I continue?',
    request_id: 'clarify-1'
  })
}

describe('useMessageStream clarify events', () => {
  beforeEach(() => {
    handleEvent = null
    states = new Map([
      [SESSION_ID, { ...createClientSessionState(), awaitingResponse: true, busy: true }]
    ])
    $activeSessionId.set(SESSION_ID)
    $clarifyRequests.set({})
  })

  afterEach(() => {
    cleanup()
    $activeSessionId.set(null)
    $clarifyRequests.set({})
    vi.restoreAllMocks()
  })

  it('adds a pending clarify tool row when request arrives without tool.start', async () => {
    await mountStream()

    clarifyRequest()

    const state = states.get(SESSION_ID)
    const message = state?.messages.at(-1)
    const part = message?.parts.at(-1)

    expect(state?.needsInput).toBe(true)
    expect(message).toMatchObject({ pending: true, role: 'assistant' })
    expect(part).toMatchObject({
      args: {
        choices: ['Use this chat', 'Start a new one'],
        question: 'Where should I continue?'
      },
      toolName: 'clarify',
      type: 'tool-call'
    })
    expect(part).not.toHaveProperty('result')
    expect($clarifyRequests.get()[SESSION_ID]).toMatchObject({
      question: 'Where should I continue?',
      requestId: 'clarify-1'
    })
  })

  it('reuses the request row when tool.start arrives afterwards', async () => {
    await mountStream()

    clarifyRequest()
    emit('tool.start', {
      args: {
        choices: ['Use this chat', 'Start a new one'],
        question: 'Where should I continue?'
      },
      name: 'clarify',
      tool_id: 'tool-call-1'
    })

    const clarifyParts = states
      .get(SESSION_ID)
      ?.messages.flatMap(message =>
        message.parts.filter(part => part.type === 'tool-call' && part.toolName === 'clarify')
      )

    expect(clarifyParts).toHaveLength(1)
    expect(clarifyParts?.[0]).toMatchObject({
      toolCallId: 'tool-call-1',
      toolName: 'clarify',
      type: 'tool-call'
    })
  })
})
