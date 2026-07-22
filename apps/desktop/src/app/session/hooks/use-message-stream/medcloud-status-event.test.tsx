import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'runtime-session-1'
let handleEvent: ((event: RpcEvent) => void) | null = null
let sessionStates: Map<string, ClientSessionState>

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(sessionStates)
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

function emit(transitionId: number, state: string, text: string) {
  act(() =>
    handleEvent!({
      payload: {
        delivery_id: transitionId + 100,
        operation_id: 'operation-1',
        state,
        text,
        transition_id: transitionId
      },
      session_id: SID,
      type: 'plugin.medcloud.status'
    })
  )
}

describe('MedCloud durable status events', () => {
  beforeEach(() => {
    handleEvent = null
    sessionStates = new Map()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('upserts one persistent system row per operation across live and replayed transitions', async () => {
    await mountStream()
    emit(10, 'QUEUED', 'MedCloud action operation-1: queued.')
    emit(11, 'VERIFIED', 'MedCloud action operation-1: verified.')
    emit(10, 'RUNNING', 'MedCloud action operation-1: running.')

    const messages = sessionStates.get(SID)?.messages ?? []
    expect(messages).toHaveLength(1)
    expect(messages[0]).toMatchObject({
      id: 'medcloud-status-operation-1',
      role: 'system',
      parts: [{ text: 'MedCloud action operation-1: verified.', type: 'text' }]
    })
  })

  it('ignores malformed transition events', async () => {
    await mountStream()
    emit(0, 'RUNNING', 'forged')
    expect(sessionStates.get(SID)).toBeUndefined()
  })
})
