import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getProfilesMock = vi.hoisted(() => vi.fn(async () => ({ profiles: [] })))
const ensureGatewayForProfile = vi.hoisted(() => vi.fn(async () => undefined))
const $gateway = atom<unknown>({ id: 'live-socket' })

vi.mock('@/store/gateway', () => ({ $gateway, ensureGatewayForProfile }))
vi.mock('@/hermes', () => ({
  BOOT_AGGREGATE_REQUEST_TIMEOUT_MS: 60_000,
  getProfiles: getProfilesMock,
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: vi.fn() } }))

const PROFILE_CACHE_STORAGE_KEY = 'hermes.desktop.profiles.cache.v1'
const SELECTED_PROFILE_SCOPE_STORAGE_KEY = 'hermes.desktop.selectedProfileScope'
const LEGACY_THEME_PROFILE_STORAGE_KEY = 'hermes-desktop-active-profile-v1'

function setDesktopProfileScopeMock(profile: string | null = null) {
  const getScope = vi.fn(async () => ({ profile }))
  const setScope = vi.fn(async (name: string | null) => ({ profile: name }))
  const set = vi.fn(async (name: string | null) => ({ profile: name }))

  Object.assign(window, {
    hermesDesktop: {
      profile: {
        getScope,
        set,
        setScope
      }
    }
  })

  return { getScope, set, setScope }
}

describe('profile cache', () => {
  beforeEach(() => {
    vi.resetModules()
    window.localStorage.clear()
    Reflect.deleteProperty(window, 'hermesDesktop')
    getProfilesMock.mockClear()
    ensureGatewayForProfile.mockClear()
  })

  afterEach(() => {
    window.localStorage.clear()
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
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

  it('uses the boot timeout for active profile refresh', async () => {
    const api = vi.fn(async () => ({ current: 'default' }))
    Object.assign(window, { hermesDesktop: { api } })

    const { refreshActiveProfile } = await import('./profile')

    await refreshActiveProfile()

    expect(api).toHaveBeenCalledWith({ path: '/api/profiles/active', timeoutMs: 60_000 })
  })

  it('seeds selected sidebar profile scope synchronously from localStorage', async () => {
    window.localStorage.setItem(SELECTED_PROFILE_SCOPE_STORAGE_KEY, 'content-creator')

    const { $profileScope, $selectedProfileScope } = await import('./profile')

    expect($selectedProfileScope.get()).toBe('content-creator')
    expect($profileScope.get()).toBe('content-creator')
  })

  it('migrates selected sidebar profile scope from the legacy theme profile key', async () => {
    window.localStorage.setItem(LEGACY_THEME_PROFILE_STORAGE_KEY, 'bina-meatzevet')

    const { $profileScope, $selectedProfileScope } = await import('./profile')

    expect($selectedProfileScope.get()).toBe('bina-meatzevet')
    expect($profileScope.get()).toBe('bina-meatzevet')
  })

  it('hydrates selected sidebar profile scope from Electron userData when localStorage is empty', async () => {
    setDesktopProfileScopeMock('office-work')

    const { $selectedProfileScope } = await import('./profile')

    await vi.waitFor(() => expect($selectedProfileScope.get()).toBe('office-work'))
    expect(window.localStorage.getItem(SELECTED_PROFILE_SCOPE_STORAGE_KEY)).toBe('office-work')
  })

  it('persists selectProfile as sidebar scope without using the relaunching profile setter', async () => {
    const desktopProfile = setDesktopProfileScopeMock(null)
    const { $profileScope, selectProfile } = await import('./profile')

    selectProfile('content-creator')

    expect($profileScope.get()).toBe('content-creator')
    expect(window.localStorage.getItem(SELECTED_PROFILE_SCOPE_STORAGE_KEY)).toBe('content-creator')
    expect(desktopProfile.setScope).toHaveBeenCalledWith('content-creator')
    expect(desktopProfile.set).not.toHaveBeenCalled()
  })
})
