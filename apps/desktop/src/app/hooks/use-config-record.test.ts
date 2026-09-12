import { afterEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import type { HermesConfigRecord } from '@/types/hermes'

const mocks = vi.hoisted(() => ({ activeProfile: 'default' }))

vi.mock('@/hermes', () => ({
  getApiRequestProfile: () => mocks.activeProfile,
  getHermesConfigRecord: vi.fn(),
  profileScopeKey: (profile: string) => profile
}))

import { hermesConfigKey, setHermesConfigCache } from './use-config-record'

afterEach(() => {
  queryClient.clear()
  mocks.activeProfile = 'default'
})

describe('Hermes config cache scope', () => {
  it('shares the concrete active-profile row between ambient and explicit consumers', () => {
    expect(hermesConfigKey()).toEqual(hermesConfigKey('default'))
    expect(hermesConfigKey('profile-a')).not.toEqual(hermesConfigKey('profile-b'))

    const config = { model: { default: 'hermes-4' } } as unknown as HermesConfigRecord
    setHermesConfigCache(config)

    expect(queryClient.getQueryData(hermesConfigKey('default'))).toBe(config)
  })

  it('resolves ambient writes again after the active profile changes', () => {
    mocks.activeProfile = 'profile-a'
    const config = { model: { default: 'model-a' } } as unknown as HermesConfigRecord
    setHermesConfigCache(config)

    expect(queryClient.getQueryData(hermesConfigKey('profile-a'))).toBe(config)
    expect(queryClient.getQueryData(hermesConfigKey('default'))).toBeUndefined()
  })
})