import { afterEach, expect, it, vi } from 'vitest'

afterEach(() => vi.unstubAllGlobals())

it('covers the pre-queue frame and a persisted guide until kickoff completes', async () => {
  vi.stubGlobal('hermesDesktop', { guestOnboardingEnabled: true })
  const { $guideOpening, $onboardingGate } = await import('./onboarding-gate')

  for (const phase of ['cinematic', 'guided'] as const) {
    $onboardingGate.set({ phase, guideQueued: false, guideKickoff: 'idle' })
    expect($guideOpening.get()).toBe(true)
    $onboardingGate.set({ phase, guideQueued: true, guideKickoff: 'starting' })
    expect($guideOpening.get()).toBe(true)
    $onboardingGate.set({ phase, guideQueued: false, guideKickoff: 'started' })
    expect($guideOpening.get()).toBe(false)
  }

  $onboardingGate.set({ phase: 'skipped', guideQueued: false, guideKickoff: 'idle' })
  expect($guideOpening.get()).toBe(false)
})

it('keeps the opening visible through a shared kickoff and settles only after the seed is durable', async () => {
  vi.stubGlobal('hermesDesktop', { guestOnboardingEnabled: true })
  const { $guideOpening, $onboardingGate, runGuideKickoff, skipGuide } = await import('./onboarding-gate')
  $onboardingGate.set({ phase: 'cinematic', guideQueued: true, guideKickoff: 'idle' })

  let finishSeed: (ready: boolean) => void = () => {}

  const seed = new Promise<boolean>(resolve => {
    finishSeed = resolve
  })

  const kickoff = vi.fn(() => seed)
  expect($guideOpening.get()).toBe(true)

  const first = runGuideKickoff(kickoff)
  const second = runGuideKickoff(kickoff)
  expect(first).toBe(second)
  expect($onboardingGate.get().guideKickoff).toBe('starting')
  expect($guideOpening.get()).toBe(true)
  finishSeed(true)
  await first
  expect(kickoff).toHaveBeenCalledOnce()
  expect($guideOpening.get()).toBe(false)
  expect($onboardingGate.get().phase).toBe('guided')
  skipGuide()
  expect($guideOpening.get()).toBe(false)
  vi.unstubAllGlobals()
})
