import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $onboardingPhase } from './onboarding-phase'
import { $onboardingSession } from './onboarding-session'
import {
  answerParkedCard,
  type ParkedCardAnswer,
  registerParkedCardAnswer,
  sessionOnboardingCardWaiting
} from './parked-card-answers'

const kind = (answer: ParkedCardAnswer) => ({ $requests: atom({}), answer, waiting: () => false })

describe('parked card answers', () => {
  const unregister: Array<() => void> = []

  afterEach(() => {
    unregister.splice(0).forEach(off => off())
    $onboardingSession.set(null)
    $onboardingPhase.set('idle')
  })

  it('stops at the first card that takes the text, and forgets a card once it unregisters', () => {
    const declines = vi.fn<ParkedCardAnswer>(() => null)
    const takes = vi.fn<ParkedCardAnswer>(sessionId => (sessionId === 'session-a' ? Promise.resolve(true) : null))
    const later = vi.fn<ParkedCardAnswer>(() => Promise.resolve(true))

    unregister.push(registerParkedCardAnswer('test-declines', kind(declines)))
    const offTakes = registerParkedCardAnswer('test-takes', kind(takes))
    unregister.push(registerParkedCardAnswer('test-later', kind(later)))

    expect(answerParkedCard('session-a', 'hello')).toBe(true)
    expect(declines).toHaveBeenCalledWith('session-a', 'hello')
    expect(later).not.toHaveBeenCalled()

    offTakes()
    later.mockReturnValue(null)

    expect(answerParkedCard('session-a', 'hello')).toBe(false)
    expect(takes).toHaveBeenCalledTimes(1)
  })

  it('reports an answer that failed to send, so the composer can restore the draft', async () => {
    const onFailed = vi.fn()
    unregister.push(registerParkedCardAnswer('test-fails', kind(() => Promise.resolve(false))))

    expect(answerParkedCard('session-a', 'hello', onFailed)).toBe(true)
    await vi.waitFor(() => expect(onFailed).toHaveBeenCalledTimes(1))
  })

  it('takes typed text for an onboarding-only card in the onboarding chat, and leaves every other chat alone', () => {
    const takes = vi.fn<ParkedCardAnswer>(() => Promise.resolve(true))
    unregister.push(registerParkedCardAnswer('test-onboarding', { ...kind(takes), onboardingOnly: true, waiting: () => true }))
    const $waiting = sessionOnboardingCardWaiting('session-a')

    expect(answerParkedCard('session-a', 'hello')).toBe(false)
    expect($waiting.get()).toBe(false)

    // A run is shown whenever its chat is named (startOnboarding).
    $onboardingPhase.set('shown')
    $onboardingSession.set({ connectionId: null, runId: 'run-1', runtimeId: 'session-a', storedId: 'stored-a' })
    expect($waiting.get()).toBe(true)
    expect(answerParkedCard('session-a', 'hello')).toBe(true)
    expect(answerParkedCard('session-b', 'hello')).toBe(false)
  })
})
