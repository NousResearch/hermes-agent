import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import type { ProfileInfo, SessionInfo } from '@/types/hermes'

// Keep profile.ts's side-effecting imports inert: the gateway socket layer and
// the REST query client must not run for real in a unit test.
const ensureGatewayForProfile = vi.fn(async () => undefined)
const $gateway = atom<unknown>({ id: 'live-socket' })
const resetStarmapGraph = vi.fn()

vi.mock('@/store/gateway', () => ({ $gateway, ensureGatewayForProfile }))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: vi.fn() } }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph }))

const {
  $activeGatewayProfile,
  $profileLastSessions,
  $profileRestoreRequest,
  $profiles,
  $showAllProfiles,
  deriveProfileActivity,
  ensureGatewayProfile,
  forgetProfileSession,
  forgetSessionMemo,
  isSessionListed,
  neediestSessionId,
  refreshProfiles,
  rememberedProfileSession,
  rememberProfileSession,
  selectProfile,
  selectProfileSession
} = await import('./profile')

const { $connection, $sessions } = await import('./session')
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

const session = (id: string, profileName?: string, over: Partial<SessionInfo> = {}): SessionInfo => ({
  ended_at: null,
  id,
  input_tokens: 0,
  is_active: false,
  last_active: 1000,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  ...(profileName ? { profile: profileName } : {}),
  source: null,
  started_at: 1000,
  title: null,
  tool_call_count: 0,
  ...over
})

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
  $connection.set(localConn())
  $profiles.set([])
  $sessions.set([])
  $profileLastSessions.set({})
  $profileRestoreRequest.set(null)
  $showAllProfiles.set(false)
  vi.stubGlobal('window', { hermesDesktop: { getConnection } })
  vi.mocked(queryClient.invalidateQueries).mockClear()
  resetStarmapGraph.mockClear()
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

describe('profile-scoped cache handling on gateway swap', () => {
  it('resets the memory graph WITHOUT a blanket react-query invalidation (keys are profile-scoped)', () => {
    $activeGatewayProfile.set('coder')

    // Profile-scoped query keys carry the profile segment, so a swap re-keys
    // instead of nuking the cache: hopping back must hit warm caches.
    expect(queryClient.invalidateQueries).not.toHaveBeenCalled()
    expect(resetStarmapGraph).toHaveBeenCalledTimes(1)
  })
})

describe('per-profile last-session memory (F1)', () => {
  it('attributes the open session to its owning profile and restores each side of an A↔B hop', () => {
    $sessions.set([session('s-default', 'default'), session('s-coder', 'coder')])

    rememberProfileSession('s-default')
    rememberProfileSession('s-coder')

    expect(rememberedProfileSession('default')).toBe('s-default')
    expect(rememberedProfileSession('coder')).toBe('s-coder')
    // Remembering one side never clobbers the other.
    rememberProfileSession('s-coder')
    expect(rememberedProfileSession('default')).toBe('s-default')
  })

  it('falls back to the active gateway profile for a brand-new session the aggregator has not listed yet', () => {
    $sessions.set([session('other', 'default')])
    $activeGatewayProfile.set('coder')

    rememberProfileSession('s-new')

    expect(rememberedProfileSession('coder')).toBe('s-new')
  })

  it('never guess-attributes before the session list has loaded', () => {
    $sessions.set([])
    $activeGatewayProfile.set('coder')

    rememberProfileSession('s-unknown')

    expect($profileLastSessions.get()).toEqual({})
  })

  it('matches a compression-rotated session by lineage root', () => {
    $sessions.set([session('tip', 'coder', { _lineage_root_id: 'root' })])

    rememberProfileSession('root')

    expect(rememberedProfileSession('coder')).toBe('root')
  })

  it('evicts a single profile memo and a dead id from every memo', () => {
    $profileLastSessions.set({ coder: 'dead', default: 'dead', writer: 'alive' })

    forgetProfileSession('coder')
    expect(rememberedProfileSession('coder')).toBeNull()

    forgetSessionMemo('dead')
    expect($profileLastSessions.get()).toEqual({ writer: 'alive' })
  })

  it('isSessionListed sees live and lineage-rooted rows but never archived ones', () => {
    const sessions = [
      session('live', 'coder'),
      session('tip', 'coder', { _lineage_root_id: 'rotated' }),
      session('gone', 'coder', { archived: true })
    ]

    expect(isSessionListed(sessions, 'live')).toBe(true)
    expect(isSessionListed(sessions, 'rotated')).toBe(true)
    expect(isSessionListed(sessions, 'gone')).toBe(false)
    expect(isSessionListed(sessions, 'missing')).toBe(false)
  })
})

describe('profile switch → session restore request', () => {
  it('fires a memo-driven restore request when switching to another profile', () => {
    selectProfile('coder')

    const request = $profileRestoreRequest.get()
    expect(request?.profile).toBe('coder')
    expect(request?.sessionId).toBeNull()
    expect(ensureGatewayForProfile).toHaveBeenCalledWith('coder')
  })

  it('re-tapping the active profile leaves the open session alone', () => {
    $activeGatewayProfile.set('coder')

    selectProfile('coder')

    expect($profileRestoreRequest.get()).toBeNull()
  })

  it('leaving the all-profiles view restores even for the same gateway profile', () => {
    $activeGatewayProfile.set('coder')
    $showAllProfiles.set(true)

    selectProfile('coder')

    expect($profileRestoreRequest.get()?.profile).toBe('coder')
  })

  it('selectProfileSession targets the explicit session (rail badge click)', () => {
    selectProfileSession('coder', 's-urgent')

    const request = $profileRestoreRequest.get()
    expect(request?.profile).toBe('coder')
    expect(request?.sessionId).toBe('s-urgent')
    expect($showAllProfiles.get()).toBe(false)
    expect(ensureGatewayForProfile).toHaveBeenCalledWith('coder')
  })

  it('mints a fresh token per switch so a rapid second switch supersedes the first', () => {
    selectProfile('coder')
    const first = $profileRestoreRequest.get()?.token
    $activeGatewayProfile.set('coder')

    selectProfile('writer')

    expect($profileRestoreRequest.get()?.token).not.toBe(first)
  })
})

describe('per-profile activity derivation (F2)', () => {
  it('buckets working/attention sessions by owning profile, attention winning over working', () => {
    const sessions = [
      session('a1', 'coder', { last_active: 10, title: 'build' }),
      session('a2', 'coder', { last_active: 30, title: 'blocked' }),
      session('b1', 'writer', { last_active: 20, title: 'draft' }),
      session('idle', 'coder', { last_active: 99 })
    ]

    const activity = deriveProfileActivity(sessions, ['a1', 'a2', 'b1'], ['a2'])

    expect(activity.coder.working.map(row => row.id)).toEqual(['a1'])
    expect(activity.coder.attention.map(row => row.id)).toEqual(['a2'])
    expect(activity.writer.working.map(row => row.id)).toEqual(['b1'])
    expect(activity.writer.attention).toEqual([])
    // Idle sessions never produce a bucket entry.
    expect(Object.values(activity).flatMap(entry => [...entry.working, ...entry.attention])).toHaveLength(3)
  })

  it('keys default-profile rows as "default" and sorts by recency', () => {
    const sessions = [
      session('old', 'main-home', { is_default_profile: true, last_active: 1 }),
      session('new', 'main-home', { is_default_profile: true, last_active: 2 })
    ]

    const activity = deriveProfileActivity(sessions, ['old', 'new'], [])

    expect(activity.default.working.map(row => row.id)).toEqual(['new', 'old'])
  })

  it('neediestSessionId prefers attention, then the most recent working session', () => {
    const sessions = [
      session('w-old', 'coder', { last_active: 1 }),
      session('w-new', 'coder', { last_active: 5 }),
      session('blocked', 'coder', { last_active: 2 })
    ]

    const withAttention = deriveProfileActivity(sessions, ['w-old', 'w-new', 'blocked'], ['blocked'])
    expect(neediestSessionId(withAttention.coder)).toBe('blocked')

    const workingOnly = deriveProfileActivity(sessions, ['w-old', 'w-new'], [])
    expect(neediestSessionId(workingOnly.coder)).toBe('w-new')

    expect(neediestSessionId(undefined)).toBeNull()
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
