import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

// Keep profile.ts's side-effecting imports inert: the gateway socket layer and
// the REST query client must not run for real in a unit test.
const ensureGatewayForProfile = vi.fn(async () => undefined)
const $gateway = atom<unknown>({ id: 'live-socket' })

vi.mock('@/store/gateway', () => ({ $gateway, ensureGatewayForProfile }))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: vi.fn() } }))

const { $activeGatewayProfile, ensureGatewayProfile, selectProfile } = await import('./profile')
const { $activeSessionId, $connection, $currentBranch, $currentCwd } = await import('./session')

const remoteConn = (over: Partial<HermesConnection> = {}): HermesConnection =>
  ({ baseUrl: 'https://hermes-roy.tail.ts.net', mode: 'remote', profile: 'vps-remote', ...over }) as HermesConnection

const localConn = (over: Partial<HermesConnection> = {}): HermesConnection =>
  ({ baseUrl: '', mode: 'local', profile: 'default', ...over }) as HermesConnection

const getConnection = vi.fn<(profile?: string | null) => Promise<HermesConnection>>()
const api = vi.fn()

beforeEach(() => {
  api.mockReset()
  getConnection.mockReset()
  ensureGatewayForProfile.mockClear()
  $gateway.set({ id: 'live-socket' })
  $activeGatewayProfile.set('default')
  $activeSessionId.set(null)
  $connection.set(localConn())
  $currentBranch.set('')
  $currentCwd.set('')
  vi.stubGlobal('window', { hermesDesktop: { api, getConnection } })
})

afterEach(() => {
  vi.unstubAllGlobals()
  $activeSessionId.set(null)
  $connection.set(null)
  $currentBranch.set('')
  $currentCwd.set('')
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

  it('seeds an idle local profile workspace from that backend default cwd', async () => {
    getConnection.mockResolvedValue(localConn({ baseUrl: 'http://127.0.0.1:54929', profile: 'utils' }))
    api.mockResolvedValue({ branch: 'main', cwd: 'D:\\project-new\\utils' })

    await ensureGatewayProfile('utils')

    expect(api).toHaveBeenCalledWith({ path: '/api/fs/default-cwd', profile: 'utils' })
    expect($currentCwd.get()).toBe('D:\\project-new\\utils')
    expect($currentBranch.get()).toBe('main')
  })

  it('keeps an active session workspace instead of replacing it with the profile default', async () => {
    $activeSessionId.set('session-1')
    $currentBranch.set('feature/session')
    $currentCwd.set('D:\\project-new\\session-worktree')
    getConnection.mockResolvedValue(localConn({ baseUrl: 'http://127.0.0.1:54929', profile: 'utils' }))
    api.mockResolvedValue({ branch: 'main', cwd: 'D:\\project-new\\utils' })

    await ensureGatewayProfile('utils')

    expect(api).not.toHaveBeenCalled()
    expect($currentCwd.get()).toBe('D:\\project-new\\session-worktree')
    expect($currentBranch.get()).toBe('feature/session')
  })

  it('updates the workspace when the user explicitly switches profiles from an active session', async () => {
    $activeSessionId.set('session-1')
    $currentBranch.set('feature/session')
    $currentCwd.set('D:\\project-new\\session-worktree')
    getConnection.mockResolvedValue(localConn({ baseUrl: 'http://127.0.0.1:54929', profile: 'utils' }))
    api.mockResolvedValue({ branch: 'main', cwd: 'D:\\project-new\\utils' })

    await selectProfile('utils')

    expect(api).toHaveBeenCalledWith({ path: '/api/fs/default-cwd', profile: 'utils' })
    expect($currentCwd.get()).toBe('D:\\project-new\\utils')
    expect($currentBranch.get()).toBe('main')
  })
})
