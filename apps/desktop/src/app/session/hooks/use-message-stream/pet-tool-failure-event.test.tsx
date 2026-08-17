import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $petActivity, $petState, setPetActivity } from '@/store/pet'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'
const OTHER_SID = 'session-2'

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

function emit(type: RpcEvent['type'], payload: RpcEvent['payload'] = {}, sessionId = SID) {
  act(() => handleEvent!({ payload, session_id: sessionId, type }))
}

describe('pet tool-failure reaction', () => {
  beforeEach(() => {
    handleEvent = null
    setPetActivity({
      busy: false,
      awaitingInput: false,
      toolRunning: false,
      reasoning: false,
      error: false,
      justCompleted: false,
      celebrate: false
    })
  })

  afterEach(() => {
    cleanup()
    setPetActivity({ error: false, toolRunning: false })
    vi.restoreAllMocks()
  })

  it('briefly shows failed when the active session has an isolated tool error', async () => {
    await mountStream()

    emit('tool.start', { name: 'terminal', tool_id: 'tool-1' })
    emit('tool.complete', { name: 'terminal', tool_id: 'tool-1', error: 'exit code 1' })

    expect($petActivity.get().error).toBe(true)
    expect($petState.get()).toBe('failed')
  })

  it('does not show failed for a successful tool or a background-session failure', async () => {
    await mountStream()

    emit('tool.complete', { name: 'terminal', tool_id: 'tool-1' })
    expect($petActivity.get().error).toBe(false)

    emit('tool.complete', { name: 'terminal', tool_id: 'tool-2', error: 'exit code 1' }, OTHER_SID)
    expect($petActivity.get().error).toBe(false)
  })
})
