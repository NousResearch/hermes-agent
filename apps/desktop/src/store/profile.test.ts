import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import type { ProfileInfo } from '@/types/hermes'

// Keep profile.ts's side-effecting imports inert: the gateway socket layer and
// the REST query client must not run for real in a unit test.
const ensureGatewayForProfile = vi.fn(async () => undefined)
const $gateway = atom<unknown>({ id: 'live-socket' })
const resetStarmapGraph = vi.fn()

vi.mock('@/store/gateway', () => ({ $gateway, ensureGatewayForProfile }))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn(),
  STARTUP_REQUEST_TIMEOUT_MS: 60_000
}))
vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: vi.fn() } }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph }))

const {
  $activeGatewayProfile,
  $activeProfile,
  $profiles,
  ensureGatewayProfile,
  refreshActiveProfile,
  refreshProfiles
} = await import('./profile')

const { $connection } = await import('./session')
const { queryClient } = await import('@/lib/query-client')
const { getProfiles } = await import('@/hermes')

const profile = (name: string, isDefault = false): ProfileInfo => ({
  has_env: false,
  is_default: isDefault,
  model: null,
  name,
  path: `/tmp/hermes/${name}`,
  provider: null,
  skill_count: 0
})

const remoteConn = (over: Partial<HermesConnection> = {}): HermesConnection =>
  ({ baseUrl: 'https://hermes-roy.tail.ts.net', mode: 'remote', profile: 'vps-remote', ...over }) as HermesConnection

const localConn = (over: Partial<HermesConnection> = {}): HermesConnection =>
  ({ baseUrl: '', mode: 'local', profile: 'default', ...over }) as HermesConnection

const getConnection = vi.fn<(profile?: string | null) => Promise<HermesConnection>>()
const api = vi.fn()
const getDesktopProfile = vi.fn()
const setDesktopProfile = vi.fn()

beforeEach(() => {
  api.mockReset()
  getDesktopProfile.mockReset()
  setDesktopProfile.mockReset()
  getConnection.mockReset()
  ensureGatewayForProfile.mockClear()
  api.mockResolvedValue({ current: 'default' })
  getDesktopProfile.mockResolvedValue({ profile: 'default' })
  setDesktopProfile.mockResolvedValue({ profile: 'default' })
  $gateway.set({ id: 'live-socket' })
  $activeProfile.set('default')
  $activeGatewayProfile.set('default')
  $connection.set(localConn())
  $profiles.set([])
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      api,
      getConnection,
      profile: {
        get: getDesktopProfile,
        set: setDesktopProfile
      }
    }
  })
  vi.mocked(queryClient.invalidateQueries).mockClear()
  resetStarmapGraph.mockClear()
})

afterEach(() => {
  vi.unstubAllGlobals()
  Reflect.deleteProperty(window, 'hermesDesktop')
  $connection.set(null)
})

describe('ensureGatewayProfile → $connection sync (#46651)', () => {
  it('refreshes $connection to the remote descriptor when activating a remote pool profile', async () => {
    // Regression: the primary window backend is local, so $connection.mode is
    // "local". Activating the remote profile must flip it to "remote" — without
    // this, image attach uses path-based image.attach against the remote
    // gateway ("image not found: C:\\…") instead of image.attach_bytes.
    getConnection.mockResolvedValue(remoteConn())

    await ensureGatewayProfile('vps-remote')

    expect(ensureGatewayForProfile).toHaveBeenCalledWith('vps-remote')
    expect(getConnection).toHaveBeenCalledWith('vps-remote')
    expect($connection.get()?.mode).toBe('remote')
    expect($connection.get()?.profile).toBe('vps-remote')
  })

  it('resyncs $connection back to local when returning to the default profile', async () => {
    $activeGatewayProfile.set('vps-remote')
    $connection.set(remoteConn())
    getConnection.mockResolvedValue(localConn())

    await ensureGatewayProfile('default')

    expect(getConnection).toHaveBeenCalledWith('default')
    expect($connection.get()?.mode).toBe('local')
  })

  it('leaves the prior connection intact when the descriptor fetch fails', async () => {
    getConnection.mockRejectedValue(new Error('backend unreachable'))

    await ensureGatewayProfile('vps-remote')

    // Best-effort: boot/reconnect resyncs later; we must not null it out here.
    expect($connection.get()?.mode).toBe('local')
  })

  it('does not churn $connection when the target is already the active profile', async () => {
    $activeGatewayProfile.set('vps-remote')
    $connection.set(remoteConn())

    await ensureGatewayProfile('vps-remote')

    expect(getConnection).not.toHaveBeenCalled()
    expect(ensureGatewayForProfile).not.toHaveBeenCalled()
    expect($connection.get()?.mode).toBe('remote')
  })
})

describe('profile-scoped cache invalidation', () => {
  it('drops the memory graph cache when the active gateway profile changes', () => {
    $activeGatewayProfile.set('coder')

    expect(queryClient.invalidateQueries).toHaveBeenCalled()
    expect(resetStarmapGraph).toHaveBeenCalledTimes(1)
  })
})

describe('refreshActiveProfile persistence (#57283)', () => {
  it('persists the backend active profile when the desktop preference is unset', async () => {
    api
      .mockResolvedValueOnce({ current: 'keo' })
      .mockResolvedValueOnce({ profiles: [profile('default', true), profile('keo')] })
    getDesktopProfile.mockResolvedValueOnce({ profile: null })
    setDesktopProfile.mockResolvedValueOnce({ profile: 'keo' })

    await refreshActiveProfile()

    expect($activeProfile.get()).toBe('keo')
    expect(getDesktopProfile).toHaveBeenCalled()
    expect(setDesktopProfile).toHaveBeenCalledWith('keo')
  })

  it('does not rewrite an existing desktop profile preference', async () => {
    api
      .mockResolvedValueOnce({ current: 'keo' })
      .mockResolvedValueOnce({ profiles: [profile('default', true), profile('keo')] })
    getDesktopProfile.mockResolvedValueOnce({ profile: 'keo' })

    await refreshActiveProfile()

    expect($activeProfile.get()).toBe('keo')
    expect(setDesktopProfile).not.toHaveBeenCalled()
  })
})

describe('refreshProfiles shared rail list (#49289)', () => {
  it('removes a deleted profile from the shared $profiles cache after Manage Profiles refreshes', async () => {
    $profiles.set([profile('default', true), profile('test1')])
    vi.mocked(getProfiles).mockResolvedValueOnce({ profiles: [profile('default', true)] })

    await refreshProfiles()

    expect($profiles.get().map(profile => profile.name)).toEqual(['default'])
  })

  it('leaves the shared $profiles cache intact when the refresh fails', async () => {
    $profiles.set([profile('default', true), profile('test1')])
    vi.mocked(getProfiles).mockRejectedValueOnce(new Error('backend unavailable'))

    await expect(refreshProfiles()).rejects.toThrow('backend unavailable')

    expect($profiles.get().map(profile => profile.name)).toEqual(['default', 'test1'])
  })
})
