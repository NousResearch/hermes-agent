// @vitest-environment jsdom
import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConfigRecord } from '@/types/hermes'

// Keep the profile store's side-effecting imports inert (same pattern as
// store/profile.test.ts): no real gateway socket, no real REST module.
const ensureGatewayForProfile = vi.fn(async () => undefined)
const $gateway = atom<unknown>({ id: 'live-socket' })

vi.mock('@/store/gateway', () => ({ $gateway, ensureGatewayForProfile }))
vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(async () => ({})),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const { HERMES_CONFIG_KEY, hermesConfigKey, invalidateHermesConfig, setHermesConfigCache } =
  await import('./use-config-record')

const { $activeGatewayProfile } = await import('@/store/profile')
const { queryClient } = await import('@/lib/query-client')

describe('profile-scoped hermes-config query keys (F4)', () => {
  beforeEach(() => {
    queryClient.clear()
    $activeGatewayProfile.set('default')
  })

  it('keys each profile into its own cache slot — no cross-profile bleed', () => {
    $activeGatewayProfile.set('alpha')
    setHermesConfigCache({ model: 'alpha-model' } as HermesConfigRecord)

    $activeGatewayProfile.set('beta')
    setHermesConfigCache({ model: 'beta-model' } as HermesConfigRecord)

    // Beta's write must not have touched alpha's slot, and vice versa:
    // hopping back to alpha serves alpha's warm cache untouched.
    expect(queryClient.getQueryData(hermesConfigKey('alpha'))).toEqual({ model: 'alpha-model' })
    expect(queryClient.getQueryData(hermesConfigKey('beta'))).toEqual({ model: 'beta-model' })
  })

  it('normalizes empty/unset profiles to the "default" segment', () => {
    expect(hermesConfigKey('')).toEqual([...HERMES_CONFIG_KEY, 'default'])
    expect(hermesConfigKey(null)).toEqual([...HERMES_CONFIG_KEY, 'default'])
    expect(hermesConfigKey('  ')).toEqual([...HERMES_CONFIG_KEY, 'default'])
    expect(hermesConfigKey('coder')).toEqual([...HERMES_CONFIG_KEY, 'coder'])
  })

  it('invalidateHermesConfig targets only the ACTIVE profile slot', async () => {
    queryClient.setQueryData(hermesConfigKey('alpha'), { a: 1 })
    queryClient.setQueryData(hermesConfigKey('beta'), { b: 2 })
    $activeGatewayProfile.set('alpha')

    await invalidateHermesConfig()

    expect(queryClient.getQueryState(hermesConfigKey('alpha'))?.isInvalidated).toBe(true)
    expect(queryClient.getQueryState(hermesConfigKey('beta'))?.isInvalidated).toBe(false)
  })
})
