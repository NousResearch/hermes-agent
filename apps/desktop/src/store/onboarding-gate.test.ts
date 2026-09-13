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
