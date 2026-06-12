import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

// Keep profile.ts's side-effecting imports inert: the gateway socket layer,
// storage persistence and REST query client must not run for real in unit tests.
const ensureGatewayForProfile = vi.fn(async () => undefined)
const $gateway = atom<unknown>({ id: 'live-socket' })

vi.mock('@/store/gateway', () => ({ $gateway, ensureGatewayForProfile }))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: vi.fn() } }))
vi.mock('@/lib/storage', () => ({
  arraysEqual: (a: unknown[], b: unknown[]) => a.length === b.length && a.every((value, index) => value === b[index]),
  persistBoolean: vi.fn(),
  persistStringArray: vi.fn(),
  persistStringRecord: vi.fn(),
  storedBoolean: vi.fn(() => false),
  storedStringArray: vi.fn(() => []),
  storedStringRecord: vi.fn(() => ({}))
}))

const {
  $activeGatewayProfile,
  $newChatProfile,
  $profileScope,
  $selectedProfileScope,
  $showAllProfiles,
  ensureGatewayProfile,
  selectProfile
} = await import('./profile')
const { $connection } = await import('./session')

const remoteConn = (over: Partial<HermesConnection> = {}): HermesConnection =>
  ({ baseUrl: 'https://hermes-roy.tail.ts.net', mode: 'remote', profile: 'vps-remote', ...over }) as HermesConnection

const localConn = (over: Partial<HermesConnection> = {}): HermesConnection =>
  ({ baseUrl: '', mode: 'local', profile: 'default', ...over }) as HermesConnection

const getConnection = vi.fn<(profile?: string | null) => Promise<HermesConnection>>()

beforeEach(() => {
  getConnection.mockReset()
  ensureGatewayForProfile.mockClear()
  $gateway.set({ id: 'live-socket' })
  $activeGatewayProfile.set('default')
  $selectedProfileScope.set('default')
  $showAllProfiles.set(false)
  $newChatProfile.set(null)
  $connection.set(localConn())
  vi.stubGlobal('window', { hermesDesktop: { getConnection } })
})

afterEach(() => {
  vi.unstubAllGlobals()
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

describe('profile sidebar scope', () => {
  it('shows the selected profile history immediately without waiting for gateway connection', () => {
    $activeGatewayProfile.set('default')
    $selectedProfileScope.set('default')
    $showAllProfiles.set(false)

    selectProfile('google_search_agent')

    expect($profileScope.get()).toBe('google_search_agent')
    expect($newChatProfile.get()).toBe('google_search_agent')
    expect(ensureGatewayForProfile).toHaveBeenCalledWith('google_search_agent')
  })
})
