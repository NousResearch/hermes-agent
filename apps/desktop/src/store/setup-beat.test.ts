import type { ServerRequest } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setupBeatOutcome } from '@/components/assistant-ui/setup-beat/receipt'

import { $onboardingPhase } from './onboarding-phase'
import { $onboardingSession } from './onboarding-session'
import {
  answerParkedCard,
  parkedCardWaiting,
  sessionOnboardingCardWaiting,
  yieldParkedCardsToMessage
} from './parked-card-answers'
import { rememberServerRequest, resetServerRequestsForTests } from './server-requests'
import { $setupBeatRequests, answerSetupBeat, clearSetupBeat, parkSetupBeat } from './setup-beat'

function parkCard(requestId: string, open = true, beat: 'layout' | 'name' = 'name') {
  const respond = vi.fn()

  const request: ServerRequest = {
    fail: vi.fn(),
    id: requestId,
    method: 'setup_beat',
    params: { session_id: 'rt-1' },
    respond
  }

  if (open) {
    rememberServerRequest(request)
  }

  parkSetupBeat({
    beat,
    card: {},
    question: 'What should I call you?',
    requestId,
    sessionId: 'rt-1',
    toolCallId: `call-${requestId}`
  })

  return respond
}

beforeEach(() => {
  // A run is shown whenever its chat is named (startOnboarding).
  $onboardingPhase.set('shown')
  $onboardingSession.set({ connectionId: null, runId: 'run-1', runtimeId: 'rt-1', storedId: 'stored-1' })
})

afterEach(() => {
  resetServerRequestsForTests()
  $setupBeatRequests.set({})
  $onboardingSession.set(null)
  $onboardingPhase.set('idle')
})

describe('setup_beat card', () => {
  it('takes composer text as the user words and closes, and a stale cancel leaves a newer card', () => {
    parkCard('srq-1')
    clearSetupBeat('srq-old', 'rt-1')
    expect($setupBeatRequests.get()['rt-1']?.requestId).toBe('srq-1')

    const respond = parkCard('srq-2')

    expect(answerParkedCard('rt-1', '  Sid  ')).toBe(true)
    expect(respond).toHaveBeenCalledWith({ said: 'Sid' })
    expect($setupBeatRequests.get()['rt-1']).toBeUndefined()
    expect(answerParkedCard('rt-1', 'again')).toBe(false)
  })

  it('reports a card waiting on the session while a beat is parked, and none once it settles', () => {
    const $waiting = sessionOnboardingCardWaiting('rt-1')
    const seen: boolean[] = []
    const off = $waiting.subscribe(value => seen.push(value))

    expect(parkedCardWaiting('rt-1')).toBe(false)
    parkCard('srq-3')
    expect(parkedCardWaiting('rt-1')).toBe(true)
    expect(parkedCardWaiting('rt-2')).toBe(false)
    expect(answerParkedCard('rt-1', 'Sid')).toBe(true)
    expect(parkedCardWaiting('rt-1')).toBe(false)
    expect(seen).toEqual([false, true, false])
    off()
  })

  it('leaves a message sent past any other card to the composer, and closes the card for it', () => {
    const respond = parkCard('srq-5', true, 'layout')

    expect(answerParkedCard('rt-1', 'can ya look around and take a wild guess?')).toBe(false)
    expect(respond).not.toHaveBeenCalled()
    expect(yieldParkedCardsToMessage('rt-1')).toBe(true)
    expect(respond).toHaveBeenCalledWith({ wrote_instead: true })
    expect($setupBeatRequests.get()['rt-1']).toBeUndefined()
    expect(yieldParkedCardsToMessage('rt-1')).toBe(false)
  })

  it('keeps a card whose answer could not be sent', () => {
    parkCard('srq-4', false)

    expect(answerSetupBeat('rt-1', { name: 'Sid' })).toBe(false)
    expect($setupBeatRequests.get()['rt-1']?.requestId).toBe('srq-4')
  })

  it('answers in the shared per-beat contract, never with a label alone', () => {
    const respond = parkCard('srq-5', true, 'layout')

    expect(answerSetupBeat('rt-1', { layout: 'sidebar-right' })).toBe(true)
    expect(respond).toHaveBeenCalledWith({ answer: { layout: 'sidebar-right' } })

    parkCard('srq-6')
    // @ts-expect-error every answer carries its beat's pick; the label is only what the card showed
    answerSetupBeat('rt-1', { label: 'Elite' })
  })

  it('collapses every way a card can close to a receipt, and draws nothing for a card-less call', () => {
    expect(setupBeatOutcome({ answer: { name: 'Sid' }, beat: 'name' })).toMatchObject({ kind: 'answer' })
    expect(setupBeatOutcome({ beat: 'theme', said: 'make it violet' })).toMatchObject({ kind: 'said' })
    expect(setupBeatOutcome({ answer: { name: 'Ada' }, beat: 'name', said: 'Ada' })).toMatchObject({ kind: 'said' })
    expect(setupBeatOutcome({ beat: 'layout', timed_out: true })).toMatchObject({ kind: 'timed_out' })
    expect(setupBeatOutcome({ beat: 'layout', wrote_instead: true })).toMatchObject({ kind: 'timed_out' })
    expect(setupBeatOutcome({ skipped_all: true })).toMatchObject({ kind: 'skipped_all' })
    expect(setupBeatOutcome({ beat: 'build', error: 'not accepted' })).toMatchObject({ kind: 'refused' })
    expect(setupBeatOutcome({ completed: true })).toBeNull()
  })
})
