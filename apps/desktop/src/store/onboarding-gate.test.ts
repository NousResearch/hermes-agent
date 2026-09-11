import { act, cleanup, render } from '@testing-library/react'
import { createElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { IntroRevealGate } from '@/components/intro-reveal'

import { $introReveal, finishIntroReveal, hasSeenIntroReveal, leaveIntroReveal } from './intro-reveal'
import { $desktopOnboarding } from './onboarding'
import {
  $onboardingGate,
  beginOnboardingFlow,
  beginOnboardingHandoff,
  completeOnboardingFlow,
  devResetOnboardingFlow,
  type OnboardingPhase,
  queueGuideAfterIntro,
  runGuideKickoff,
  skipGuide
} from './onboarding-gate'

beforeEach(() => {
  window.localStorage.clear()
  devResetOnboardingFlow()
  $introReveal.set({ phase: 'hidden' })
  $desktopOnboarding.set({ ...$desktopOnboarding.get(), firstRunSkipped: false })
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function enableOnboarding() {
  const open = vi.fn().mockResolvedValue({ ok: true })

  vi.stubGlobal('hermesDesktop', {
    guestOnboardingEnabled: true,
    introReveal: {
      open,
      close: vi.fn().mockResolvedValue({ ok: true }),
      onSkip: () => () => undefined,
      onClosed: () => () => undefined
    }
  })

  return open
}

describe('onboarding phase record', () => {
  it('advances from the real intro edge through guided, handoff and accepted completion in order', async () => {
    const open = enableOnboarding()
    vi.stubGlobal('hermesDesktop', { guestOnboardingEnabled: false })
    const phases: OnboardingPhase[] = []

    const stop = $onboardingGate.listen(state => {
      if (phases.at(-1) !== state.phase) {
        phases.push(state.phase)
      }
    })

    const gate = render(createElement(IntroRevealGate, { enabled: true }))
    const kickoff = vi.fn().mockResolvedValue(true)

    beginOnboardingFlow()
    queueGuideAfterIntro()
    expect(await runGuideKickoff(kickoff)).toBe(false)
    expect(kickoff).not.toHaveBeenCalled()
    expect(phases).toEqual([])
    expect(open).not.toHaveBeenCalled()

    enableOnboarding()
    act(() => $desktopOnboarding.set({ ...$desktopOnboarding.get(), firstRunSkipped: true }))
    gate.rerender(createElement(IntroRevealGate, { enabled: false }))
    gate.rerender(createElement(IntroRevealGate, { enabled: true }))
    expect($onboardingGate.get().phase).toBe('idle')
    act(() => $desktopOnboarding.set({ ...$desktopOnboarding.get(), firstRunSkipped: false }))
    expect($onboardingGate.get().phase).toBe('cinematic')
    completeOnboardingFlow()
    beginOnboardingHandoff()
    expect($onboardingGate.get().phase).toBe('cinematic')

    act(() => {
      leaveIntroReveal()
      finishIntroReveal()
    })
    expect($onboardingGate.get()).toEqual({ phase: 'cinematic', guideQueued: true })
    expect(hasSeenIntroReveal()).toBe(true)
    expect(await runGuideKickoff(kickoff)).toBe(true)
    beginOnboardingHandoff()
    completeOnboardingFlow()
    queueGuideAfterIntro()
    beginOnboardingFlow()
    expect(phases).toEqual(['cinematic', 'guided', 'handoff', 'done'])
    expect($onboardingGate.get().guideQueued).toBe(false)
    stop()

    devResetOnboardingFlow()
    expect(hasSeenIntroReveal()).toBe(true)
  })

  it('skips from cinematic to done without letting the intro close requeue the guide', async () => {
    enableOnboarding()
    render(createElement(IntroRevealGate, { enabled: true }))
    expect($onboardingGate.get().phase).toBe('cinematic')
    skipGuide()
    act(finishIntroReveal)
    queueGuideAfterIntro()
    const kickoff = vi.fn().mockResolvedValue(true)
    expect(await runGuideKickoff(kickoff)).toBe(false)
    expect(kickoff).not.toHaveBeenCalled()
    expect($onboardingGate.get()).toEqual({ phase: 'done', guideQueued: false })
  })

  it('shares one pending kickoff and retries failure without latching a guide that never started', async () => {
    enableOnboarding()
    render(createElement(IntroRevealGate, { enabled: true }))
    act(finishIntroReveal)

    const failure = new Error('seed persistence failed')
    const rejected = vi.fn().mockRejectedValue(failure)
    const firstFailure = runGuideKickoff(rejected)
    expect(runGuideKickoff(rejected)).toBe(firstFailure)
    await expect(firstFailure).rejects.toBe(failure)
    expect(rejected).toHaveBeenCalledTimes(1)
    expect($onboardingGate.get().guideQueued).toBe(true)
    expect(await runGuideKickoff(async () => false)).toBe(false)
    expect($onboardingGate.get().phase).toBe('cinematic')

    let finish: (started: boolean) => void = () => undefined

    const pending = new Promise<boolean>(resolve => {
      finish = resolve
    })

    const kickoff = vi.fn(() => pending)
    const first = runGuideKickoff(kickoff)
    const second = runGuideKickoff(kickoff)
    expect(second).toBe(first)
    await Promise.resolve()
    expect(kickoff).toHaveBeenCalledTimes(1)
    expect($onboardingGate.get().phase).toBe('cinematic')
    finish(true)
    expect(await Promise.all([first, second])).toEqual([true, true])
    expect(await runGuideKickoff(kickoff)).toBe(true)
    expect(kickoff).toHaveBeenCalledTimes(1)
    expect($onboardingGate.get().phase).toBe('guided')
  })
})
