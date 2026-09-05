import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import type { ProfileInfo } from '@/types/hermes'

// Keep profile.ts's side-effecting imports inert: the gateway socket layer and
// the REST query client must not run for real in a unit test.
const ensureGatewayForProfile = vi.fn(async () => undefined)
const ensureGatewayForAgent = vi.fn(async () => undefined)
const openGatewayForProfile = vi.fn(async (_profile: string) => undefined)
const openSecondaryCount = vi.fn(() => 0)
const $gateway = atom<unknown>({ id: 'live-socket', connectionState: 'open' })
const resetStarmapGraph = vi.fn()

vi.mock('@/store/gateway', () => ({
  $gateway,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  openGatewayForProfile,
  openSecondaryCount
}))
// The pool-limits atom is profile.ts's live saturation signal — keep the real
// one so tests can move the cap via the store, but stub its IPC bridge.
vi.mock('@/store/pool-limits', async () => {
  const { atom } = await import('nanostores')

  return { $poolLimits: atom({ idleMs: 600_000, maxBackends: 3 }) }
})
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph }))

const {
  $activeGatewayProfile,
  $profiles,
  ensureGatewayProfile,
  invalidateProfileListFetches,
  prewarmProfileBackend,
  refreshProfiles
} = await import('./profile')

const { $poolLimits } = await import('@/store/pool-limits')

const { $connection } = await import('./session')
const { invalidateProfileScopedQueries } = await import('@/lib/query-client')
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

beforeEach(() => {
  getConnection.mockReset()
  ensureGatewayForProfile.mockClear()
  openGatewayForProfile.mockClear()
  openSecondaryCount.mockReturnValue(0)
  $gateway.set({ id: 'live-socket', connectionState: 'open' })
  $activeGatewayProfile.set('default')
  $connection.set(localConn())
  $profiles.set([])
  vi.stubGlobal('window', { hermesDesktop: { getConnection } })
  vi.mocked(invalidateProfileScopedQueries).mockClear()
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

  it('reconnects when the target profile is active but its gateway socket is closed', async () => {
    $activeGatewayProfile.set('vps-remote')
    $connection.set(remoteConn())
    $gateway.set({ connectionState: 'closed' })
    getConnection.mockResolvedValue(remoteConn())

    await ensureGatewayProfile('vps-remote')

    expect(ensureGatewayForProfile).toHaveBeenCalledWith('vps-remote')
  })
})

describe('profile-scoped cache invalidation', () => {
  it('drops the memory graph cache when the active gateway profile changes', () => {
    $activeGatewayProfile.set('coder')

    expect(invalidateProfileScopedQueries).toHaveBeenCalled()
    expect(resetStarmapGraph).toHaveBeenCalledTimes(1)
  })
})

describe('prewarmProfileBackend (hover-intent pool spawn)', () => {
  it('opens the gateway (spawn + connect, no activation) for a non-active profile', () => {
    prewarmProfileBackend('warm-basic')

    expect(openGatewayForProfile).toHaveBeenCalledWith('warm-basic')
    // Pre-warm must never activate — that's the click's job.
    expect(ensureGatewayForProfile).not.toHaveBeenCalled()
  })

  it('skips the profile the gateway is already on', () => {
    $activeGatewayProfile.set('warm-active')

    prewarmProfileBackend('warm-active')

    expect(openGatewayForProfile).not.toHaveBeenCalled()
  })

  it('throttles repeat pre-warms for the same profile within the interval', () => {
    prewarmProfileBackend('warm-throttle-a')
    prewarmProfileBackend('warm-throttle-a')
    prewarmProfileBackend('warm-throttle-b')

    const calls = openGatewayForProfile.mock.calls.map(([name]) => name)
    expect(calls.filter(name => name === 'warm-throttle-a')).toHaveLength(1)
    expect(calls.filter(name => name === 'warm-throttle-b')).toHaveLength(1)
  })

  it('swallows spawn failures — error UX belongs to the real switch', () => {
    openGatewayForProfile.mockRejectedValueOnce(new Error('spawn failed'))

    expect(() => prewarmProfileBackend('warm-failing')).not.toThrow()
  })

  it('skips pre-warm when the pool is saturated (#91545 evict/respawn cascade)', () => {
    // Every pool slot occupied: a speculative spawn would LRU-evict a warm
    // backend — often the one the user is about to click. Default limit 3,
    // 3 open secondaries → the next spawn would exceed the cap.
    openSecondaryCount.mockReturnValue(3)

    prewarmProfileBackend('warm-saturated')

    expect(openGatewayForProfile).not.toHaveBeenCalled()
  })

  it('pre-warms while pool slots are free', () => {
    openSecondaryCount.mockReturnValue(1)

    prewarmProfileBackend('warm-slot-free')

    expect(openGatewayForProfile).toHaveBeenCalledWith('warm-slot-free')
  })

  it('follows the live pool-limit atom, not a hard-coded cap', () => {
    // User raises Warm Bot Backends to 8 in Settings: prewarming must keep
    // working well past the old default of 3.
    openSecondaryCount.mockReturnValue(5)
    $poolLimits.set({ idleMs: 600_000, maxBackends: 8 })

    prewarmProfileBackend('warm-raised-cap')

    expect(openGatewayForProfile).toHaveBeenCalledWith('warm-raised-cap')

    // And lowering the cap re-engages the guard at the new boundary.
    $poolLimits.set({ idleMs: 600_000, maxBackends: 2 })

    prewarmProfileBackend('warm-lowered-cap')

    expect(openGatewayForProfile).not.toHaveBeenCalledWith('warm-lowered-cap')
  })

  it('counts in-flight spawns against the cap so a hover sweep cannot overshoot (#91545 race)', async () => {
    // The race: openSecondaryCount() only sees backends whose socket is
    // already OPEN. During a fast hover sweep the sockets have not opened yet,
    // so a naive guard reads a stale low count for every prewarm and spawns
    // past maxBackends — the exact eviction cascade the guard should prevent.
    // Hold every spawn in flight (socket never opens) and confirm the guard
    // still stops at the cap. Default cap 3, no open secondaries yet.
    $poolLimits.set({ idleMs: 600_000, maxBackends: 3 })
    openSecondaryCount.mockReturnValue(0)
    // Capture each spawn's resolver so we can drain them at the end and leave
    // the module-level in-flight set clean for later tests.
    const resolvers: Array<() => void> = []
    openGatewayForProfile.mockImplementation(
      () =>
        new Promise<undefined>(resolve => {
          resolvers.push(() => resolve(undefined)) // held until drain
        })
    )

    // Sweep across five distinct profiles faster than any socket can open.
    prewarmProfileBackend('sweep-a')
    prewarmProfileBackend('sweep-b')
    prewarmProfileBackend('sweep-c')
    prewarmProfileBackend('sweep-d')
    prewarmProfileBackend('sweep-e')

    // Only maxBackends (3) spawns may be launched; the rest are refused while
    // their predecessors are still in flight.
    expect(openGatewayForProfile).toHaveBeenCalledTimes(3)
    const swept = openGatewayForProfile.mock.calls.map(([name]) => name)
    expect(swept).toEqual(['sweep-a', 'sweep-b', 'sweep-c'])
    expect(resolvers).toHaveLength(3)

    // Drain the held spawns so the in-flight reservations release and no
    // promise dangles into the next test.
    resolvers.forEach(release => release())
    await new Promise(resolve => setTimeout(resolve, 0))
  })

  it('frees the in-flight reservation once a spawn settles, letting later prewarms proceed', async () => {
    $poolLimits.set({ idleMs: 600_000, maxBackends: 3 })
    openSecondaryCount.mockReturnValue(0)
    // Two profiles fill 2 of 3 slots and stay open; a third spawn is held in
    // flight so all three slots are reserved.
    let resolveHeld: (() => void) | undefined
    openGatewayForProfile
      .mockResolvedValueOnce(undefined)
      .mockResolvedValueOnce(undefined)
      .mockImplementationOnce(
        () =>
          new Promise<undefined>(resolve => {
            resolveHeld = () => resolve(undefined)
          })
      )
      .mockResolvedValue(undefined)

    prewarmProfileBackend('fill-1')
    prewarmProfileBackend('fill-2')
    prewarmProfileBackend('fill-3-held')
    // Flush all microtasks so the two resolved spawns run their
    // .catch().finally() chain and release their in-flight reservation.
    await new Promise(resolve => setTimeout(resolve, 0))

    // fill-1 and fill-2 have released; fill-3 is still in flight (1 slot used),
    // so a fourth prewarm has room and proceeds.
    prewarmProfileBackend('fill-4')
    expect(openGatewayForProfile).toHaveBeenCalledWith('fill-4')

    // Drain the held spawn so no promise dangles.
    resolveHeld?.()
    await new Promise(resolve => setTimeout(resolve, 0))
  })
})

describe('refreshProfiles shared rail list (#49289)', () => {
  beforeEach(() => {
    vi.mocked(getProfiles).mockReset()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
  })

  it('removes a deleted profile from the shared $profiles cache after Manage Profiles refreshes', async () => {
    $profiles.set([profile('default', true), profile('test1')])
    vi.mocked(getProfiles).mockResolvedValueOnce({ profiles: [profile('default', true)] })

    await refreshProfiles()

    expect($profiles.get().map(profile => profile.name)).toEqual(['default'])
  })

  it('recovers from transient failures and writes the returned profile list (#70679)', async () => {
    // Global remote mode: the refresh fires while the remote HTTP proxy is still
    // routing, so the first attempts fail and a later one succeeds. The retry
    // backoff is 500ms then 1000ms (refreshProfiles retries twice on failure).
    $profiles.set([])
    vi.mocked(getProfiles)
      .mockRejectedValueOnce(new Error('backend unavailable'))
      .mockRejectedValueOnce(new Error('backend unavailable'))
      .mockResolvedValueOnce({ profiles: [profile('default', true), profile('healthops')] })

    const refresh = refreshProfiles()
    await vi.advanceTimersByTimeAsync(500)
    await vi.advanceTimersByTimeAsync(1000)
    await expect(refresh).resolves.toHaveLength(2)

    expect(vi.mocked(getProfiles)).toHaveBeenCalledTimes(3)
    expect($profiles.get().map(profile => profile.name)).toEqual(['default', 'healthops'])
  })

  it('shares one retry chain across concurrent callers (single-flight)', async () => {
    // Gateway open fires both useBackgroundSync and the activeGatewayProfile
    // effect at once; both callers must ride the same chain, not double it.
    $profiles.set([])
    vi.mocked(getProfiles)
      .mockRejectedValueOnce(new Error('backend unavailable'))
      .mockResolvedValueOnce({ profiles: [profile('default', true), profile('healthops')] })

    const first = refreshProfiles()
    const second = refreshProfiles()
    await vi.advanceTimersByTimeAsync(500)
    await expect(first).resolves.toHaveLength(2)
    await expect(second).resolves.toHaveLength(2)

    expect(vi.mocked(getProfiles)).toHaveBeenCalledTimes(2)
    expect($profiles.get().map(profile => profile.name)).toEqual(['default', 'healthops'])
  })

  it('leaves the shared $profiles cache intact when every retry fails', async () => {
    $profiles.set([profile('default', true), profile('test1')])
    vi.mocked(getProfiles).mockRejectedValue(new Error('backend unavailable'))

    const refresh = refreshProfiles()
    const rejection = expect(refresh).rejects.toThrow('backend unavailable')
    await vi.advanceTimersByTimeAsync(500)
    await vi.advanceTimersByTimeAsync(1000)
    await rejection

    expect(vi.mocked(getProfiles)).toHaveBeenCalledTimes(3)
    expect($profiles.get().map(profile => profile.name)).toEqual(['default', 'test1'])
  })
})

describe('stale profile-list fetches across a backend switch (#85731)', () => {
  it('a late response from the previous backend cannot clobber the new backend list', async () => {
    // The disappearing-rail mechanism: /api/profiles is in flight against
    // backend A when the user applies a different remote/Cloud connection.
    // The soft re-home fetches backend B's list, then A's late (often empty /
    // default-only) response lands LAST and collapses the rail.
    let resolveOld: (value: { profiles: ProfileInfo[] }) => void = () => undefined
    vi.mocked(getProfiles).mockImplementationOnce(() => new Promise(resolve => (resolveOld = resolve)))

    const oldFetch = refreshProfiles() // in flight against backend A

    // Connection apply → soft re-home strands in-flight fetches...
    invalidateProfileListFetches()

    // ...and the new backend's list arrives.
    vi.mocked(getProfiles).mockResolvedValueOnce({
      profiles: [profile('default', true), profile('eric'), profile('coder')]
    })
    await refreshProfiles()
    expect($profiles.get().map(profile => profile.name)).toEqual(['default', 'eric', 'coder'])

    // Backend A's stale response finally lands: it must NOT overwrite $profiles.
    resolveOld({ profiles: [profile('default', true)] })
    await oldFetch

    expect($profiles.get().map(profile => profile.name)).toEqual(['default', 'eric', 'coder'])
  })

  it('a normal refresh still writes the cache after prior invalidations', async () => {
    invalidateProfileListFetches()
    vi.mocked(getProfiles).mockResolvedValueOnce({ profiles: [profile('default', true), profile('solo')] })

    await refreshProfiles()

    expect($profiles.get().map(profile => profile.name)).toEqual(['default', 'solo'])
  })

  it('strands in-flight fetches when the active gateway profile swaps backends', async () => {
    // Sibling site of the same class: a live profile swap moves /api/profiles
    // routing to another backend mid-fetch. The $activeGatewayProfile
    // subscriber must bump the epoch exactly like the connection-apply wipe.
    let resolveOld: (value: { profiles: ProfileInfo[] }) => void = () => undefined
    vi.mocked(getProfiles).mockImplementationOnce(() => new Promise(resolve => (resolveOld = resolve)))

    const oldFetch = refreshProfiles() // in flight against the old profile's backend

    $activeGatewayProfile.set('coder') // swap → subscriber invalidates

    vi.mocked(getProfiles).mockResolvedValueOnce({
      profiles: [profile('default', true), profile('coder')]
    })
    await refreshProfiles()

    resolveOld({ profiles: [] })
    await oldFetch

    expect($profiles.get().map(profile => profile.name)).toEqual(['default', 'coder'])
  })
})
