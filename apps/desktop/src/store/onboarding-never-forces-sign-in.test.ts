/** The lifetime allowance starts after guided onboarding. Provider prompts,
 * deferred warnings and the free-tier ready screen must yield throughout the
 * cinematic, guided and handoff stages. */
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

async function load(phase: string) {
  storage.clear()
  storage.set('hermes-onboarding-phase-v1', phase)
  vi.resetModules()

  const gate = await import('./onboarding-gate')
  const onboarding = await import('./onboarding')

  return { gate, onboarding }
}

it.each(['cinematic', 'guided', 'handoff'])('the provider picker never opens over the guide (%s)', async phase => {
  const { onboarding } = await load(phase)

  onboarding.requestDesktopOnboarding('No inference provider is configured.')

  expect(onboarding.$desktopOnboarding.get().requested).toBe(false)
})

it.each(['cinematic', 'guided', 'handoff'])(
  'a credential warning during the guide is dropped, not deferred (%s)',
  async phase => {
    const { onboarding } = await load(phase)

    onboarding.requestDesktopOnboardingForCredentialWarning(
      "No API key configured for provider 'nous'. First message will fail."
    )

    expect(onboarding.consumePendingCredentialWarning()).toBeNull()
  }
)

it.each(['idle', 'skipped', 'done'])('outside the guide the picker opens as before (%s)', async phase => {
  const { onboarding } = await load(phase)

  onboarding.requestDesktopOnboarding('No inference provider is configured.')

  expect(onboarding.$desktopOnboarding.get().requested).toBe(true)
})
