import type { OnboardingEndedPayload } from '@hermes/shared'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $onboardingPhase } from '@/store/onboarding-phase'
import { noteOnboardingEnded, startOnboarding } from '@/store/onboarding-run'
import { isOnboardingChat } from '@/store/onboarding-session'
import { $connection, $selectedStoredSessionId } from '@/store/session'

import { OnboardingSkip } from './skip-setup'

const started = { resumed: false, run_id: 'run-1', session_id: 'rt-1', stored_session_id: 'st-1' }
const ended: OnboardingEndedPayload = { outcome: 'done', run_id: 'run-1', stored_session_id: 'st-1' }
const ensureProfile = async (): Promise<void> => undefined
const resume = async (storedId: string) => $selectedStoredSessionId.set(storedId)

// SAFETY: the fake answers onboarding.start and onboarding.state with the fields startOnboarding reads.
const request = async <T,>(method: string): Promise<T> =>
  (method === 'onboarding.state' ? { eligible: true, run: true, run_id: 'run-1', session_id: 'st-1' } : started) as T

async function onboardingChatOnScreen() {
  await startOnboarding(request, resume, { ensureProfile })
  render(<OnboardingSkip sessionId="rt-1" />)
  expect(screen.getByRole('button', { name: 'Skip setup' })).toBeTruthy()
}

beforeEach(() => {
  // SAFETY: factsSource reads only `mode`.
  $connection.set({ mode: 'local' } as never)
  $onboardingPhase.set('idle')
})

afterEach(cleanup)

describe('Skip setup', () => {
  it('goes away when the backend ends the run', async () => {
    await onboardingChatOnScreen()

    act(() => noteOnboardingEnded(ended))

    expect(screen.queryByRole('button', { name: 'Skip setup' })).toBeNull()
  })

  // Last in the file: a throwing listener leaves nanostores' shared listener queue undrained.
  it('goes away even when a listener of the ending throws', async () => {
    await onboardingChatOnScreen()
    $onboardingPhase.listen(phase => {
      if (phase === 'done') {
        throw new Error('listener failed')
      }
    })

    act(() => expect(() => noteOnboardingEnded(ended)).toThrow('listener failed'))

    expect($onboardingPhase.get()).toBe('done')
    expect(isOnboardingChat('rt-1')).toBe(false)
    expect(screen.queryByRole('button', { name: 'Skip setup' })).toBeNull()
  })
})
