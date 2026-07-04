import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getProfilesMock = vi.hoisted(() => vi.fn(async () => ({ profiles: [] })))
const ensureGatewayForProfile = vi.hoisted(() => vi.fn(async () => undefined))
const $gateway = atom<unknown>({ id: 'live-socket' })

vi.mock('@/store/gateway', () => ({ $gateway, ensureGatewayForProfile }))
vi.mock('@/hermes', () => ({
  getProfiles: getProfilesMock,
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: vi.fn() } }))

const PROFILE_CACHE_STORAGE_KEY = 'hermes.desktop.profiles.cache.v1'

describe('profile cache', () => {
  beforeEach(() => {
    vi.resetModules()
    window.localStorage.clear()
    getProfilesMock.mockClear()
    ensureGatewayForProfile.mockClear()
  })

  afterEach(() => {
    window.localStorage.clear()
    vi.restoreAllMocks()
  })

  it('seeds the profile list synchronously from localStorage before backend refresh', async () => {
    window.localStorage.setItem(
      PROFILE_CACHE_STORAGE_KEY,
      JSON.stringify([
        {
          has_env: true,
          is_default: true,
          model: null,
          name: 'default',
          path: '/home/endlessblink/.hermes',
          provider: null,
          skill_count: 0
        },
        {
          has_env: true,
          is_default: false,
          model: null,
          name: 'office-work',
          path: '/home/endlessblink/.hermes/profiles/office-work',
          provider: null,
          skill_count: 4
        }
      ])
    )

    const { $profiles } = await import('./profile')

    expect(getProfilesMock).not.toHaveBeenCalled()
    expect($profiles.get().map(profile => profile.name)).toEqual(['default', 'office-work'])
  })
})
