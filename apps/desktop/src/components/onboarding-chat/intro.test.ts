import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { stopTour } from '@/lib/tour'
import { $tourActive } from '@/lib/tour/tour-active'
import { $onboardingGate } from '@/store/onboarding-gate'
import { $selectedStoredSessionId } from '@/store/session'
import { $toursEnabled } from '@/store/tours'

import { $chatOnboardingThreadIds } from './assembly'
import { finishGuidedOnboarding } from './intro'

const SETUP = 'setup-chat'
const TASK = 'task-chat'

describe('finishGuidedOnboarding handoff tour', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    // jsdom lays nothing out; the tour only targets a rail with a size.
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 10, 10))
    document.body.innerHTML = '<div data-tour="profile-rail"></div>'
    $chatOnboardingThreadIds.set([SETUP])
    $onboardingGate.set({ guideKickoff: 'idle', guideQueued: false, phase: 'left' })
    $toursEnabled.set(true)
    $selectedStoredSessionId.set(SETUP)
  })

  afterEach(async () => {
    await stopTour()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('opens the tour over the task chat the watched handoff opened', async () => {
    finishGuidedOnboarding(SETUP, TASK)
    $selectedStoredSessionId.set(TASK)
    await vi.advanceTimersByTimeAsync(10_000)

    expect($tourActive.get()).toBe(true)
  })

  it('opens no tour when the setup chat finished in the background', async () => {
    $selectedStoredSessionId.set('other-chat')

    finishGuidedOnboarding(SETUP, TASK)
    await vi.advanceTimersByTimeAsync(10_000)

    expect($tourActive.get()).toBe(false)
  })

  it('opens no tour when the user moved to another chat before the handoff opened', async () => {
    finishGuidedOnboarding(SETUP, TASK)
    $selectedStoredSessionId.set('other-chat')
    await vi.advanceTimersByTimeAsync(10_000)

    expect($tourActive.get()).toBe(false)
  })
})
