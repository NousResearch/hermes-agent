import { afterEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import type { HermesConfigRecord } from '@/types/hermes'

const mocks = vi.hoisted(() => ({ activeProfile: 'default' }))

vi.mock('@/hermes', () => ({
  getApiRequestProfile: () => mocks.activeProfile,
  getHermesConfigRecord: vi.fn(),
  profileScopeKey: (profile: string) => profile
}))

import { hermesConfigCacheWriter, hermesConfigKey } from './use-config-record'

afterEach(() => {
  queryClient.clear()
  mocks.activeProfile = 'default'
})

describe('Hermes config cache scope', () => {
  it('shares the concrete active-profile row between ambient and explicit consumers', () => {
    expect(hermesConfigKey()).toEqual(hermesConfigKey('default'))
    expect(hermesConfigKey('profile-a')).not.toEqual(hermesConfigKey('profile-b'))

    const config = { model: { default: 'hermes-4' } } as unknown as HermesConfigRecord
    hermesConfigCacheWriter()(config)

    expect(queryClient.getQueryData(hermesConfigKey('default'))).toBe(config)
  })

  it('binds an ambient writer to the profile active when work starts', () => {
    mocks.activeProfile = 'profile-a'
    const writeProfileA = hermesConfigCacheWriter()
    mocks.activeProfile = 'profile-b'
    const config = { model: { default: 'model-a' } } as unknown as HermesConfigRecord
    writeProfileA(config)

    expect(queryClient.getQueryData(hermesConfigKey('profile-a'))).toBe(config)
    expect(queryClient.getQueryData(hermesConfigKey('profile-b'))).toBeUndefined()
  })
})