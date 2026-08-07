import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearAllPrompts, sessionApprovalRequest } from '@/store/prompts'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'

const sessionStates = new Map<string, ClientSessionState>()
let handleEvent: ((event: RpcEvent) => void) | null = null
let updateSessionStateCalls: string[] = []

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
      updateSessionStateCalls.push(sessionId)
      const next = updater(sessionStates.get(sessionId) ?? createClientSessionState())
      sessionStates.set(sessionId, next)

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

function emitApprovalRequest() {
  act(() =>
    handleEvent!({
      payload: { command: 'rm -rf /tmp/example', description: 'dangerous command' },
      session_id: SID,
      type: 'approval.request'
    })
  )
}

describe('interrupted stream blocking prompts', () => {
  beforeEach(() => {
    handleEvent = null
    updateSessionStateCalls = []
    sessionStates.clear()
  })

  afterEach(() => {
    cleanup()
    clearAllPrompts()
    sessionStates.clear()
    updateSessionStateCalls = []
    vi.restoreAllMocks()
  })

  it('ignores a late approval.request after the session has been interrupted', async () => {
    sessionStates.set(SID, { ...createClientSessionState(), interrupted: true })
    await mountStream()

    emitApprovalRequest()

    expect(sessionApprovalRequest(SID).get()).toBeNull()
    expect(updateSessionStateCalls).toEqual([])
  })

  it('parks approval.request for a running session', async () => {
    await mountStream()

    emitApprovalRequest()

    expect(sessionApprovalRequest(SID).get()).toMatchObject({ command: 'rm -rf /tmp/example', sessionId: SID })
    expect(updateSessionStateCalls).toEqual([SID])
  })
})
