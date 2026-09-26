import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { errorRecoveryPlan } from '@/lib/error-surface'

import { $activeSessionId, $selectedStoredSessionId, $unreadFinishedSessionIds } from './session'
import {
  $sessionStates,
  $stalledSessionIds,
  $workingSessionIds,
  clearAllSessionStates,
  LIVE_TURN_EVENT_SILENCE_MS,
  noteSessionEvent,
  publishSessionState,
  SESSION_WATCHDOG_TIMEOUT_MS
} from './session-states'

// Read from the store rather than restated here: these assert what happens on
// either side of the threshold, not what the threshold is.
const WATCHDOG_MS = SESSION_WATCHDOG_TIMEOUT_MS

function state(over: Partial<ClientSessionState> = {}): ClientSessionState {
  return { ...createClientSessionState(null), storedSessionId: 's1', ...over }
}

describe('session watchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  it('marks a silent session stalled without pretending it finished', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))

    vi.advanceTimersByTime(WATCHDOG_MS)

    expect($workingSessionIds.get()).toContain('s1')
    expect($stalledSessionIds.get()).toContain('s1')
  })

  it('clears stalled on new activity and rearms the watchdog', () => {
    const working = state({ busy: true, storedSessionId: 's2' })
    publishSessionState('rt2', working)
    vi.advanceTimersByTime(WATCHDOG_MS)
    expect($stalledSessionIds.get()).toContain('s2')

    publishSessionState('rt2', { ...working, awaitingResponse: true })
    expect($stalledSessionIds.get()).not.toContain('s2')

    vi.advanceTimersByTime(WATCHDOG_MS - 1)
    expect($stalledSessionIds.get()).not.toContain('s2')
    expect($workingSessionIds.get()).toContain('s2')
  })

  it('clears both running and stalled on an authoritative terminal transition', () => {
    const working = state({ busy: true, storedSessionId: 's3' })
    publishSessionState('rt3', working)
    vi.advanceTimersByTime(WATCHDOG_MS)
    expect($stalledSessionIds.get()).toContain('s3')

    publishSessionState('rt3', { ...working, busy: false })

    expect($workingSessionIds.get()).not.toContain('s3')
    expect($stalledSessionIds.get()).not.toContain('s3')
  })

  it('never marks a session stalled when it settles before the window', () => {
    const working = state({ busy: true, storedSessionId: 's4' })
    publishSessionState('rt4', working)
    publishSessionState('rt4', { ...working, busy: false })
    vi.advanceTimersByTime(WATCHDOG_MS)

    expect($workingSessionIds.get()).not.toContain('s4')
    expect($stalledSessionIds.get()).not.toContain('s4')
  })

  it('clears stalled state and disarms timers on a gateway wipe', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    vi.advanceTimersByTime(WATCHDOG_MS)
    expect($stalledSessionIds.get()).toEqual(['s1'])

    clearAllSessionStates()
    vi.advanceTimersByTime(WATCHDOG_MS)

    expect($workingSessionIds.get()).toEqual([])
    expect($stalledSessionIds.get()).toEqual([])
  })
})

describe('computed $workingSessionIds', () => {
  beforeEach(() => {
    clearAllSessionStates()
  })

  afterEach(() => {
    clearAllSessionStates()
  })

  it('reflects busy sessions under the id their surfaces key on', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    publishSessionState('rt2', state({ busy: false, storedSessionId: 's2' }))
    // Not yet persisted, so the runtime id is the only id it has — and the one
    // the row is keyed by until the backend hands a stored id back.
    publishSessionState('rt3', state({ busy: true, storedSessionId: null }))

    expect($workingSessionIds.get()).toEqual(['s1', 'rt3'])
  })

  it('updates when session state changes', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    expect($workingSessionIds.get()).toEqual(['s1'])

    publishSessionState('rt1', state({ busy: false, storedSessionId: 's1' }))
    expect($workingSessionIds.get()).toEqual([])
  })
})

const SILENCE_MS = LIVE_TURN_EVENT_SILENCE_MS

function partial(text: string, over: Partial<ClientSessionState> = {}): ClientSessionState {
  return state({
    awaitingResponse: true,
    busy: true,
    messages: [
      {
        id: 'a1',
        parts: [{ type: 'text', text }],
        pending: true,
        role: 'assistant'
      }
    ],
    model: 'any-model',
    sawAssistantPayload: true,
    streamId: 'a1',
    turnLive: true,
    turnStartedAt: Date.now(),
    ...over
  })
}

describe('live turn event silence', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  it('force-settles a silent live turn even after a partial payload and offers retry', () => {
    $activeSessionId.set('rt1')
    publishSessionState('rt1', partial('partial answer', { model: 'glm-5.3-flash', storedSessionId: 's1' }))
    noteSessionEvent('rt1')

    vi.advanceTimersByTime(SILENCE_MS)

    const settled = $sessionStates.get().rt1
    expect($workingSessionIds.get()).not.toContain('s1')
    expect(settled?.busy).toBe(false)
    expect(settled?.awaitingResponse).toBe(false)
    expect(settled?.turnLive).toBe(false)
    expect(settled?.messages.every(message => !message.pending)).toBe(true)
    expect(
      settled?.messages.some(message =>
        message.parts.some(part => part.type === 'text' && part.text === 'partial answer')
      )
    ).toBe(true)

    const failed = settled?.messages.find(message => message.errorSurface)
    expect(failed?.errorSurface?.retryable).toBe(true)
    expect(errorRecoveryPlan(failed?.errorSurface).retry).toBe(true)
    expect(failed?.error).not.toMatch(/glm|deepseek|interrupted mid-run/i)
  })

  it('force-settles a silent live turn that never produced a payload', () => {
    $activeSessionId.set('rt-empty')
    publishSessionState(
      'rt-empty',
      state({ awaitingResponse: true, busy: true, model: 'deepseek-chat', storedSessionId: 's-empty', turnLive: true })
    )
    noteSessionEvent('rt-empty')

    vi.advanceTimersByTime(SILENCE_MS)

    const settled = $sessionStates.get()['rt-empty']
    expect($workingSessionIds.get()).not.toContain('s-empty')
    expect(settled?.busy).toBe(false)
    expect(settled?.turnLive).toBe(false)
    const failed = settled?.messages.find(message => message.role === 'assistant' && message.errorSurface)
    expect(failed?.errorSurface?.retryable).toBe(true)
    expect(errorRecoveryPlan(failed?.errorSurface).retry).toBe(true)
  })

  it('does not settle a live turn that keeps producing events', () => {
    publishSessionState('rt-live', partial('still working', { storedSessionId: 's-live' }))
    noteSessionEvent('rt-live')

    vi.advanceTimersByTime(SILENCE_MS - 1)
    noteSessionEvent('rt-live')
    vi.advanceTimersByTime(SILENCE_MS - 1)

    expect($workingSessionIds.get()).toContain('s-live')
    expect($sessionStates.get()['rt-live']?.messages.some(message => message.errorSurface)).toBe(false)
  })

  it('does not settle a turn the user is still answering', () => {
    publishSessionState('rt-ask', partial('need a choice', { needsInput: true, storedSessionId: 's-ask' }))
    noteSessionEvent('rt-ask')

    vi.advanceTimersByTime(SILENCE_MS)

    expect($workingSessionIds.get()).toContain('s-ask')
    expect($sessionStates.get()['rt-ask']?.messages.some(message => message.errorSurface)).toBe(false)
  })

  it('settles only the session that stopped producing events', () => {
    $activeSessionId.set('rt-a')
    publishSessionState('rt-a', partial('a', { storedSessionId: 's-a' }))
    publishSessionState('rt-b', partial('b', { storedSessionId: 's-b' }))
    noteSessionEvent('rt-a')
    noteSessionEvent('rt-b')

    vi.advanceTimersByTime(SILENCE_MS - 1)
    noteSessionEvent('rt-b')
    vi.advanceTimersByTime(1)

    expect($workingSessionIds.get()).not.toContain('s-a')
    expect($workingSessionIds.get()).toContain('s-b')
    expect($sessionStates.get()['rt-a']?.messages.some(message => message.errorSurface?.retryable)).toBe(true)
    expect($sessionStates.get()['rt-b']?.busy).toBe(true)
  })

  it('does not stamp a retry when the turn settles before the silence window', () => {
    const working = partial('done soon', { storedSessionId: 's-done' })
    publishSessionState('rt-done', working)
    noteSessionEvent('rt-done')
    publishSessionState('rt-done', {
      ...working,
      awaitingResponse: false,
      busy: false,
      messages: working.messages.map(message => ({ ...message, pending: false })),
      turnLive: false
    })

    vi.advanceTimersByTime(SILENCE_MS)

    expect($sessionStates.get()['rt-done']?.messages.some(message => message.errorSurface)).toBe(false)
    expect($workingSessionIds.get()).not.toContain('s-done')
  })
})
