import { expect, it, vi } from 'vitest'

import type * as storageModule from '@/lib/storage'

const storage = vi.hoisted(() => new Map<string, string>())

vi.mock('@/lib/onboarding-enabled', () => ({ isOnboardingEnabled: () => true }))
vi.mock('@/lib/storage', async importOriginal => ({
  ...(await importOriginal<typeof storageModule>()),
  readKey: (key: string) => storage.get(key) ?? null,
  writeKey: (key: string, value: string | null) => {
    if (value === null) {
      storage.delete(key)
    } else {
      storage.set(key, value)
    }
  }
}))

it('restores every persisted onboarding phase and rejects unknown phases', async () => {
  const { ONBOARDING_PHASES } = await import('./onboarding-gate')

  for (const phase of ONBOARDING_PHASES.filter(phase => phase !== 'idle')) {
    storage.clear()
    storage.set('hermes-onboarding-phase-v1', phase)
    vi.resetModules()
    const { $onboardingGate } = await import('./onboarding-gate')

    expect($onboardingGate.get().phase).toBe(phase)
  }

  storage.set('hermes-onboarding-phase-v1', 'unknown-phase')
  vi.resetModules()
  const { $onboardingGate } = await import('./onboarding-gate')

  expect($onboardingGate.get().phase).toBe('idle')
})

it('a skipped intro still queues the guided flow, and never replays the film', async () => {
  const { beginOnboardingFlowWithoutIntro } = await import('./onboarding-gate')
  const { hasSeenIntroReveal } = await import('./intro-reveal')

  storage.clear()
  beginOnboardingFlowWithoutIntro(false)
  const { $onboardingGate } = await import('./onboarding-gate')

  expect($onboardingGate.get()).toEqual({ phase: 'cinematic', guideQueued: true })
  // The film is recorded as watched: a later launch without HERMES_SKIP_INTRO
  // must adopt the persisted guide, not play the film over it.
  expect(hasSeenIntroReveal()).toBe(true)

  // A relaunch boots loadGate() over the same persisted state and re-queues the guide.
  vi.resetModules()
  const { $onboardingGate: relaunched } = await import('./onboarding-gate')

  expect(relaunched.get().guideQueued).toBe(true)
  expect(relaunched.get().phase).toBe('cinematic')
})

it('a skipped intro does nothing when onboarding is off or the first run was skipped', async () => {
  vi.doMock('@/lib/onboarding-enabled', () => ({ isOnboardingEnabled: () => false }))
  vi.resetModules()
  const off = await import('./onboarding-gate')

  off.beginOnboardingFlowWithoutIntro(false)
  expect(off.$onboardingGate.get().phase).toBe('idle')

  vi.doUnmock('@/lib/onboarding-enabled')
  vi.resetModules()
  const on = await import('./onboarding-gate')

  on.beginOnboardingFlowWithoutIntro(true)
  expect(on.$onboardingGate.get().phase).toBe('idle')
})
