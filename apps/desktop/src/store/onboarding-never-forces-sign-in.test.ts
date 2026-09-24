/**
 * The free tier is unmetered and the guided first launch never demands an
 * account. This is the acceptance criterion the guided onboarding was built
 * to, as a test rather than a memory: a user can work through the guide and
 * the first build, tool call after tool call, and the only way a sign-in
 * reaches them is the guide's own ready screen at the moment the guide picks.
 * Every surface that could push a sign-in over the guide — the provider
 * picker, the deferred credential warning, the free-tier ready screen — has
 * to yield while the gate is cinematic, guided or handoff.
 */
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
