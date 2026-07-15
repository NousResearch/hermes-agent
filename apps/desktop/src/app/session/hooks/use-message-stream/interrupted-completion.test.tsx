import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const cues = vi.hoisted(() => ({
  dispatchNativeNotification: vi.fn(),
  flashPetActivity: vi.fn(),
  markPetUnread: vi.fn(),
  playCompletionSound: vi.fn(),
  setPetActivity: vi.fn()
}))

vi.mock('@/lib/completion-sound', () => ({ playCompletionSound: cues.playCompletionSound }))
vi.mock('@/store/native-notifications', () => ({
  dispatchNativeNotification: cues.dispatchNativeNotification
}))
vi.mock('@/store/pet', () => ({
  flashPetActivity: cues.flashPetActivity,
  markPetUnread: cues.markPetUnread,
  setPetActivity: cues.setPetActivity
}))

const SID = 'session-1'
const LEGACY_INTERRUPT_SENTINELS = [
  'Operation interrupted.',
  'Operation interrupted: waiting for model response (0.3s elapsed).',
  'Operation interrupted during retry (upstream gateway timeout (504, 42s), attempt 1/3).',
  'Operation interrupted: handling API error (RateLimitError: Too many requests).',
  'Operation interrupted: retrying API call after error (retry 2/3).'
] as const
const PREFIX_COLLIDING_PARTIAL =
  'Operation interrupted: waiting for model response (this phrase describes the log; the real answer continues here.'

let currentState: ClientSessionState
let handleEvent: ((event: RpcEvent) => void) | null = null

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>([[SID, currentState]]))
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const state = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
      currentState = updater(state)
      sessionStateByRuntimeIdRef.current.set(sessionId, currentState)

      return currentState
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

describe('useMessageStream interrupted completion', () => {
  beforeEach(() => {
    currentState = {
      ...createClientSessionState(),
      awaitingResponse: true,
      busy: true,
      interrupted: false
    }
    handleEvent = null
    vi.clearAllMocks()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it.each(LEGACY_INTERRUPT_SENTINELS)(
    'settles backend interruption without prose or completion cues, then accepts the queued turn start: %s',
    async interruptText => {
      await mountStream()

      act(() =>
        handleEvent!({
          payload: { status: 'interrupted', text: interruptText },
          session_id: SID,
          type: 'message.complete'
        })
      )

      expect(currentState.interrupted).toBe(false)
      expect(currentState.busy).toBe(false)
      expect(currentState.awaitingResponse).toBe(false)
      expect(currentState.messages.map(chatMessageText)).not.toContain(interruptText)
      expect(cues.playCompletionSound).not.toHaveBeenCalled()
      expect(cues.flashPetActivity).not.toHaveBeenCalled()
      expect(cues.dispatchNativeNotification).not.toHaveBeenCalled()

      act(() => handleEvent!({ payload: {}, session_id: SID, type: 'message.start' }))

      expect(currentState.busy).toBe(true)
      expect(currentState.awaitingResponse).toBe(true)
      expect(currentState.interrupted).toBe(false)
    }
  )

  it('keeps real partial assistant text that begins with a legacy prefix', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { status: 'interrupted', text: PREFIX_COLLIDING_PARTIAL },
        session_id: SID,
        type: 'message.complete'
      })
    )

    expect(currentState.messages.map(chatMessageText)).toContain(PREFIX_COLLIDING_PARTIAL)
    expect(cues.playCompletionSound).not.toHaveBeenCalled()
    expect(cues.flashPetActivity).not.toHaveBeenCalled()
    expect(cues.dispatchNativeNotification).not.toHaveBeenCalled()
  })
})
