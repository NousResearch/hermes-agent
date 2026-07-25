import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $notifications, clearNotifications } from '@/store/notifications'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'rt-new-session'

let handleEvent: ((event: RpcEvent) => void) | null = null
let sessionStates: Map<string, ClientSessionState> | null = null

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
    sessionStates = sessionStateByRuntimeIdRef.current
  }, [stream.handleGatewayEvent])

  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

/** Seed the session as it looks right after a first-message submit: the
 *  optimistic user row is present, the turn is awaiting its response. */
function seedOptimisticFirstMessage() {
  sessionStates!.set(SID, {
    ...createClientSessionState('stored-new-session', [
      { id: 'user-123-abc', role: 'user', parts: [textPart('first message of a new chat')] }
    ]),
    busy: true,
    awaitingResponse: true
  })
}

describe('useMessageStream agent-init error surfacing (#63078)', () => {
  beforeEach(() => {
    handleEvent = null
    sessionStates = null
    clearNotifications()
  })

  afterEach(() => {
    cleanup()
    clearNotifications()
    vi.restoreAllMocks()
  })

  it('renders an agent-init failure as a visible in-transcript error and keeps the optimistic first message', async () => {
    await mountStream()
    seedOptimisticFirstMessage()

    act(() =>
      handleEvent!({
        payload: {
          message:
            'agent initialization timed out after 601s — your message was not sent; retry once the session is ready'
        },
        session_id: SID,
        type: 'error'
      })
    )

    const state = sessionStates!.get(SID)!

    // The user's optimistic first message must survive — the failure mode of
    // #63078 was the message silently vanishing into a blank session.
    const userRows = state.messages.filter(m => m.role === 'user')
    expect(userRows).toHaveLength(1)
    expect(userRows[0]!.id).toBe('user-123-abc')

    // The failure is VISIBLE in the session view: an assistant error bubble...
    const errorRows = state.messages.filter(m => m.role === 'assistant' && m.error)
    expect(errorRows).toHaveLength(1)
    expect(errorRows[0]!.error).toContain('your message was not sent')

    // ...and the composer is released (no forever-spinner on a dead turn).
    expect(state.busy).toBe(false)
    expect(state.awaitingResponse).toBe(false)

    // A global toast also fired (turn-ending errors are easy to miss inline).
    expect($notifications.get().some(n => n.kind === 'error' && n.message?.includes('was not sent'))).toBe(true)
  })

  it('renders the pre-ready cancel error event (#65567 server emit) visibly', async () => {
    await mountStream()
    seedOptimisticFirstMessage()

    act(() =>
      handleEvent!({
        payload: { message: 'Turn cancelled before the agent was ready' },
        session_id: SID,
        type: 'error'
      })
    )

    const state = sessionStates!.get(SID)!
    expect(state.messages.some(m => m.role === 'assistant' && m.error?.includes('cancelled'))).toBe(true)
    expect(state.messages.some(m => m.id === 'user-123-abc')).toBe(true)
    expect(state.busy).toBe(false)
  })
})
