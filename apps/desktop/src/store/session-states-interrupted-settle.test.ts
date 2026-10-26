import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { errorRecoveryPlan } from '@/lib/error-surface'
import { requestGatewayForAgent } from '@/store/gateway'

import { $activeSessionId, $selectedStoredSessionId, $unreadFinishedSessionIds } from './session'
import {
  $sessionStates,
  $workingSessionIds,
  clearAllSessionStates,
  LIVE_TURN_EVENT_SILENCE_MS,
  noteSessionEvent,
  publishSessionState,
  setLiveTurnBackend
} from './session-states'

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

// Read from the store rather than restating the threshold: this asserts what
// happens when the silence window runs out, not what the window is.
const SILENCE_MS = LIVE_TURN_EVENT_SILENCE_MS

function liveTurn(over: Partial<ClientSessionState> = {}): ClientSessionState {
  return {
    ...createClientSessionState('s1'),
    awaitingResponse: true,
    busy: true,
    turnLive: true,
    turnStartedAt: Date.now(),
    ...over
  }
}

describe('settling a deliberately interrupted live turn', () => {
  let releaseBackend: () => void = () => undefined

  const card = (runtimeId: string) => $sessionStates.get()[runtimeId]?.messages.find(message => message.errorSurface)

  beforeEach(() => {
    vi.useFakeTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  afterEach(() => {
    releaseBackend()
    releaseBackend = () => undefined
    vi.mocked(requestGatewayForAgent).mockReset()
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  it('never paints a retryable no-reply card for an interrupted turn', async () => {
    $activeSessionId.set('rt-stop')

    // A live turn that has produced no reply yet (prefill, a quiet background
    // review) goes silent, and the silence check starts. The user then
    // deliberately ends the turn — Stop, a superseding prompt, or a session
    // delete stamping the doomed runtime — marking interrupted=true while the
    // turn is still settling. When the backend then confirms the turn ended,
    // that confirmation must not repaint the deliberate interruption as a
    // retryable connection failure (#131205).
    publishSessionState('rt-stop', liveTurn({ storedSessionId: 's-stop', messages: [] }))
    noteSessionEvent('rt-stop')
    publishSessionState('rt-stop', {
      ...$sessionStates.get()['rt-stop']!,
      interrupted: true,
      needsInput: false
    })
    releaseBackend = setLiveTurnBackend({ request: vi.fn(async () => ({ sessions: [] })) as never })

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    const settled = $sessionStates.get()['rt-stop']
    expect(settled?.interrupted).toBe(true)
    expect(settled?.busy).toBe(false)
    expect(settled?.turnLive).toBe(false)
    expect(card('rt-stop')).toBeUndefined()
  })

  it('still offers the retry card when an uninterrupted turn ends with no reply', async () => {
    $activeSessionId.set('rt-gone')
    publishSessionState('rt-gone', liveTurn({ storedSessionId: 's-gone', messages: [] }))
    releaseBackend = setLiveTurnBackend({ request: vi.fn(async () => ({ sessions: [] })) as never })
    noteSessionEvent('rt-gone')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    const failed = card('rt-gone')
    expect($workingSessionIds.get()).not.toContain('s-gone')
    expect(failed?.errorSurface?.code).toBe('no_reply')
    expect(errorRecoveryPlan(failed?.errorSurface).retry).toBe(true)
  })
})
