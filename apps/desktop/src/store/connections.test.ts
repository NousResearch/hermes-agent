import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type { DesktopConnectionsRegistry } from '@/global'
import { BACKEND_BOOT_WAIT_TIMEOUT_MS, SOURCE_SWITCH_DIAL_TIMEOUT_MS } from '@/lib/with-timeout'

import { deferred } from '../test/deferred'

const $activeGatewayProfile = atom('default')
const $newChatProfile = atom<null | string>(null)
const $freshSessionRequest = atom(0)
const $showAllProfiles = atom(false)

const $connection = atom<null | {
  connectionId?: string
  mode?: 'local' | 'remote'
  profile?: string
  registryScoped?: boolean
}>(null)

// The runtime session id minted by the CURRENT backend — the binding a pick
// must drop before the next gateway is published (#93937).
const $activeSessionId = atom<null | string>(null)

interface ActivationOptions {
  beforeActivate?: () => boolean
  signal?: AbortSignal
}

const ensureGatewayAgent = vi.fn(
  async (_connectionId: null | string, _profile: string, _options?: ActivationOptions): Promise<void> => undefined
)

const openGatewayAgent = vi.fn(async (_connectionId: string, _profile: string): Promise<void> => undefined)
const refreshActiveProfile = vi.fn(async () => undefined)
const requestFreshSession = vi.fn()
const exitProjectScope = vi.fn()

vi.mock('@/store/session', () => ({
  $connection,
  $activeSessionId,
  $selectedStoredSessionId: atom(null),
  setActiveSessionId: (id: null | string) => $activeSessionId.set(id),
  setSelectedStoredSessionId: vi.fn()
}))
vi.mock('@/store/project-scope', () => ({ exitProjectScope }))
vi.mock('@/store/gateway-switch', () => ({ supersedeGatewaySwitch: vi.fn() }))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile,
  $freshSessionRequest,
  $newChatProfile,
  $newChatRoute: atom(null),
  $showAllProfiles,
  captureNewChatSource: vi.fn(),
  currentNewChatIntent: () => 0,
  ensureGatewayAgent,
  normalizeProfileKey: (name: null | string | undefined) => (name ?? '').trim() || 'default',
  openGatewayAgent,
  refreshActiveProfile,
  requestFreshSession
}))

const {
  $activeConnectionId,
  $connectionsRegistry,
  $pendingConnectionId,
  initializeConnectionsRegistry,
  refreshConnectionsRegistry,
  _resetConnectionsForTests,
  selectConnection,
  setConnectionsRegistry
} = await import('./connections')

const registry: DesktopConnectionsRegistry = {
  connections: [
    { id: 'local', kind: 'local', label: 'This device', tokenPreview: null, tokenSet: false },
    { id: 'homelab', kind: 'remote', label: 'Homelab', tokenPreview: '...abc', tokenSet: true },
    { id: 'work-vps', kind: 'remote', label: 'Work VPS', tokenPreview: '...xyz', tokenSet: true }
  ],
  primary: 'local',
  secureTokenStorage: true,
  version: 2
}

const list = vi.fn(async () => registry)
const api = vi.fn(async () => ({ profiles: [] }))
const setLastUsed = vi.fn(async (id: string) => ({ ok: true, registry: { ...registry, lastUsed: id } }))

beforeEach(() => {
  localStorage.clear()
  _resetConnectionsForTests()
  $connectionsRegistry.set(null)
  $connection.set(null)
  $activeGatewayProfile.set('default')
  $newChatProfile.set(null)
  $showAllProfiles.set(false)
  ensureGatewayAgent.mockReset()
  // Mirrors the real door: beforeActivate runs right before the activation
  // publishes, and a declined hook publishes nothing.
  ensureGatewayAgent.mockImplementation(async (connectionId, profile, options) => {
    if (options?.beforeActivate && !options.beforeActivate()) {
      return
    }

    $connection.set({
      connectionId: connectionId ?? undefined,
      mode: connectionId === 'local' ? 'local' : 'remote',
      // The primary local descriptor from startHermes() historically carried
      // no profile key at all (see the switch-back regression test at the
      // bottom of this file); every other route publishes its profile.
      ...(connectionId === 'local' ? {} : { profile }),
      registryScoped: true
    })
  })
  openGatewayAgent.mockReset()
  openGatewayAgent.mockResolvedValue(undefined)
  refreshActiveProfile.mockClear()
  requestFreshSession.mockClear()
  exitProjectScope.mockClear()
  $activeSessionId.set(null)
  list.mockClear()
  setLastUsed.mockClear()
  api.mockReset()
  api.mockResolvedValue({ profiles: [] })
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.stubGlobal('window', {
    hermesDesktop: { api, connections: { list, setLastUsed } },
    localStorage,
    location: window.location
  })
})

afterEach(() => vi.unstubAllGlobals())

describe('connection registry cache', () => {
  it('loads only Electron local registry state', async () => {
    await refreshConnectionsRegistry()

    expect(list).toHaveBeenCalledTimes(1)
    expect($connectionsRegistry.get()).toEqual(registry)
    expect($activeConnectionId.get()).toBeNull()

    // A stuck IPC read must release the lifecycle's retry loop and preserve
    // the last good cache, even if the timed-out snapshot eventually arrives.
    vi.useFakeTimers()
    const stuck = deferred<DesktopConnectionsRegistry>()
    list.mockImplementationOnce(() => stuck.promise)

    try {
      const refresh = refreshConnectionsRegistry()
      const rejected = expect(refresh).rejects.toThrow('Timed out reading the connection registry')
      await vi.advanceTimersByTimeAsync(5_000)
      await rejected
      stuck.resolve({ ...registry, connections: [] })
      await Promise.resolve()
      expect($connectionsRegistry.get()).toEqual(registry)
    } finally {
      vi.useRealTimers()
    }
  })

  it('restores the last-used source once when that launch mode is enabled', async () => {
    list.mockResolvedValueOnce({ ...registry, lastUsed: 'homelab', launchMode: 'last-used' })
    $connection.set({ connectionId: 'local', mode: 'local' })

    await initializeConnectionsRegistry()
    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).toHaveBeenCalledTimes(1)
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
    expect(setLastUsed).toHaveBeenCalledWith('homelab')
  })

  it('preserves the established Primary-source launch behavior by default', async () => {
    list.mockResolvedValueOnce({ ...registry, lastUsed: 'homelab', launchMode: 'primary' })
    $connection.set({ connectionId: 'local', mode: 'local' })

    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).not.toHaveBeenCalled()
  })

  it('selects the registry primary on boot when an update left an unqualified local descriptor', async () => {
    // Post-update boot can publish a local descriptor with no registry id
    // while connections.json still says launchMode=primary and the primary is
    // SSH. That live local must not block selecting the registered primary.
    list.mockResolvedValueOnce({
      ...registry,
      lastUsed: 'local',
      launchMode: 'primary',
      primary: 'homelab'
    })
    $connection.set({ mode: 'local' })

    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).toHaveBeenCalledTimes(1)
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
  })

  it('restores a remote registry primary through its exact connection id', async () => {
    list.mockResolvedValueOnce({ ...registry, primary: 'homelab', launchMode: 'primary' })
    $connection.set({ connectionId: 'local', mode: 'local' })

    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).toHaveBeenCalledTimes(1)
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
    expect(setLastUsed).toHaveBeenCalledWith('homelab')
  })

  it('boot restore yields to a source the user already picked while boot was settling', async () => {
    // Primary is local, launch mode is primary. The user clicks a fleet-rail
    // square on the homelab gateway before the boot-time restore runs. The
    // restore must not "return" the window to the primary over that choice.
    list.mockResolvedValue({ ...registry, primary: 'local', launchMode: 'primary' })
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })

    await selectConnection('homelab', { profile: 'omer' })
    expect($activeConnectionId.get()).toBe('homelab')
    ensureGatewayAgent.mockClear()

    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect($activeConnectionId.get()).toBe('homelab')
  })

  it('uses only the resolved descriptor identity for the active gateway', () => {
    setConnectionsRegistry({ ...registry, primary: 'homelab' })
    $connection.set({ connectionId: 'work-vps', mode: 'remote' })
    expect($activeConnectionId.get()).toBe('work-vps')

    $connection.set({ mode: 'remote' })
    expect($activeConnectionId.get()).toBeNull()

    $connection.set({ connectionId: 'work-vps', mode: 'remote' })
    expect($activeConnectionId.get()).toBe('work-vps')
  })
})

describe('selectConnection', () => {
  it('activates the picked gateway in place and starts a fresh draft there, with no wipe', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })

    await selectConnection('homelab')

    expect(ensureGatewayAgent).toHaveBeenCalledTimes(1)
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
    expect(api).not.toHaveBeenCalled()
    expect(requestFreshSession).toHaveBeenCalledTimes(1)
    expect(exitProjectScope).toHaveBeenCalledTimes(1)
    expect($newChatProfile.get()).toBe('default')
    expect(refreshActiveProfile).toHaveBeenCalledTimes(1)
    expect(setLastUsed).toHaveBeenCalledWith('homelab')
  })

  it('does not reset or dial when the active source/profile is selected again', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })

    await selectConnection('local')

    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect(requestFreshSession).not.toHaveBeenCalled()
  })

  it('uses an explicit local id when This device is not primary', async () => {
    setConnectionsRegistry({ ...registry, primary: 'homelab' })
    $connection.set({ connectionId: 'homelab', mode: 'remote' })

    await selectConnection('local')

    expect(ensureGatewayAgent).toHaveBeenCalledWith('local', 'default', expect.anything())
    expect(api).not.toHaveBeenCalled()
  })

  it('lands on an explicit profile instead of the one last used on that source', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })

    // Remember "scout" on homelab by activating it once…
    await selectConnection('homelab', { profile: 'scout' })
    expect(ensureGatewayAgent).toHaveBeenLastCalledWith('homelab', 'scout', expect.anything())

    // …then pick a different square on the same source: the click wins over
    // whatever was last used there, and the fresh draft targets that profile.
    await selectConnection('local')
    await selectConnection('homelab', { profile: 'omer' })
    expect(ensureGatewayAgent).toHaveBeenLastCalledWith('homelab', 'omer', expect.anything())
    expect($newChatProfile.get()).toBe('omer')
  })

  it('restores the last profile used on each source', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local', profile: 'research', registryScoped: true })
    $activeGatewayProfile.set('research')
    $connection.set({ connectionId: 'homelab', mode: 'remote', registryScoped: true })
    $activeGatewayProfile.set('default')

    await selectConnection('local')

    expect(ensureGatewayAgent).toHaveBeenCalledWith('local', 'research', expect.anything())
  })

  it('does not remember a migrated v1 routing alias as a backend profile', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'homelab', mode: 'remote' })
    $activeGatewayProfile.set('legacy-homelab-alias')
    $connection.set({ connectionId: 'local', mode: 'local', registryScoped: true })

    await selectConnection('homelab')

    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
  })

  it('does not remember a stale startup profile under the resolved source', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local', profile: 'default', registryScoped: true })
    $activeGatewayProfile.set('work-agent')
    $connection.set({ connectionId: 'homelab', mode: 'remote', profile: 'default', registryScoped: true })
    $activeGatewayProfile.set('default')

    await selectConnection('local')

    expect(ensureGatewayAgent).toHaveBeenCalledWith('local', 'default', expect.anything())
  })

  it('a gateway that is down fails at the open and touches nothing in front', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    $activeSessionId.set('a93bb39d')
    openGatewayAgent.mockRejectedValueOnce(new Error('offline'))

    await expect(selectConnection('homelab')).rejects.toThrow('offline')

    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect($activeSessionId.get()).toBe('a93bb39d')
    expect(requestFreshSession).not.toHaveBeenCalled()
    expect($newChatProfile.get()).toBeNull()
    expect($pendingConnectionId.get()).toBeNull()
    expect(setLastUsed).not.toHaveBeenCalled()
    expect($connection.get()?.connectionId).toBe('local')
  })

  it.each(['remote', 'cloud'] as const)(
    'requires protected REST auth on a warm %s socket before replacing the current chat',
    async kind => {
      const oauthRegistry: DesktopConnectionsRegistry = {
        ...registry,
        connections: registry.connections.map(connection =>
          connection.id === 'homelab' ? { ...connection, kind, authMode: 'oauth' } : connection
        )
      }

      setConnectionsRegistry(oauthRegistry)
      const source = { connectionId: 'work-vps', mode: 'remote' as const, profile: 'research', registryScoped: true }
      $connection.set(source)
      $activeGatewayProfile.set('research')
      $activeSessionId.set('source-runtime')
      $newChatProfile.set('research')
      setApiRequestConnection('work-vps')
      setApiRequestProfile('research')
      setLastUsed.mockImplementationOnce(async id => ({ ok: true, registry: { ...oauthRegistry, lastUsed: id } }))

      // A kept socket already passed its handshake; opening it succeeds even
      // though fresh REST requests no longer authenticate.
      const expired = new Error(kind === 'remote' ? '401 Unauthorized: no_cookie' : '403 Forbidden: session expired')
      api.mockRejectedValueOnce(expired)

      await expect(selectConnection('homelab', { profile: 'scout' })).rejects.toBe(expired)
      expect($connection.get()).toBe(source)
      expect($activeSessionId.get()).toBe('source-runtime')
      expect($newChatProfile.get()).toBe('research')
      expect(ensureGatewayAgent).not.toHaveBeenCalled()
      expect(requestFreshSession).not.toHaveBeenCalled()
      expect(exitProjectScope).not.toHaveBeenCalled()
      expect(setLastUsed).not.toHaveBeenCalled()

      // Auth refreshed externally: the next click proves the read again and lands.
      await selectConnection('homelab', { profile: 'scout' })
      expect(api).toHaveBeenCalledTimes(2)
      expect($activeConnectionId.get()).toBe('homelab')
      expect($activeSessionId.get()).toBeNull()
    }
  )

  it("leaves the old gateway's project before the fresh draft is requested", async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })

    await selectConnection('homelab')

    expect(exitProjectScope.mock.invocationCallOrder[0]).toBeLessThan(requestFreshSession.mock.invocationCallOrder[0])
  })

  it("#93937: the previous gateway's runtime is unbound BEFORE the picked gateway is published", async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    // Runtime id minted by the local backend; only local has ever heard of it.
    $activeSessionId.set('a93bb39d')

    const published: Array<{ activeSessionId: null | string; connectionId?: string; drafted: number }> = []

    const off = $connection.listen(next => {
      published.push({
        activeSessionId: $activeSessionId.get(),
        connectionId: next?.connectionId,
        drafted: requestFreshSession.mock.calls.length
      })
    })

    await selectConnection('homelab')
    off()

    expect(published).toEqual([{ activeSessionId: null, connectionId: 'homelab', drafted: 1 }])
  })

  it('a failed activation leaves the gateway in front usable and points the draft back at it', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    // The pick reached its turn (draft rehomed), then the dial failed.
    ensureGatewayAgent.mockImplementationOnce(async (_connectionId, _profile, options) => {
      options?.beforeActivate?.()

      throw new Error('offline')
    })

    await expect(selectConnection('homelab')).rejects.toThrow('offline')

    expect($connection.get()?.connectionId).toBe('local')
    expect($newChatProfile.get()).toBe('default')
    expect(requestFreshSession).toHaveBeenCalledTimes(2)
    expect(setLastUsed).not.toHaveBeenCalled()
    expect($pendingConnectionId.get()).toBeNull()
  })

  it('a pick superseded while queued neither drafts nor activates', async () => {
    // Both activations sit behind an in-flight profile/agent activation (the
    // profile store's mutex); their beforeActivate hooks only run once it settles.
    const mutex = deferred()

    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    $activeSessionId.set('a93bb39d')
    ensureGatewayAgent.mockImplementation(async (connectionId, _profile, options) => {
      await mutex.promise

      if (options?.beforeActivate && !options.beforeActivate()) {
        return
      }

      $connection.set({
        connectionId: connectionId ?? undefined,
        mode: 'remote',
        profile: 'default',
        registryScoped: true
      })
    })

    const published: string[] = []
    const off = $connection.listen(next => published.push(`${next?.connectionId}:active=${$activeSessionId.get()}`))

    const first = selectConnection('homelab')
    await vi.waitFor(() => expect(ensureGatewayAgent).toHaveBeenCalledTimes(1))
    const second = selectConnection('work-vps')
    await vi.waitFor(() => expect(ensureGatewayAgent).toHaveBeenCalledTimes(2))

    // Nothing is touched while both are still queued.
    expect($activeSessionId.get()).toBe('a93bb39d')
    expect(requestFreshSession).not.toHaveBeenCalled()

    mutex.resolve()
    await Promise.all([first, second])
    off()

    // homelab stepped aside at its turn: no draft, no publication, no error.
    expect(published).toEqual(['work-vps:active=null'])
    expect(requestFreshSession).toHaveBeenCalledTimes(1)
    expect(setLastUsed).toHaveBeenCalledTimes(1)
    expect(setLastUsed).toHaveBeenCalledWith('work-vps')
    expect($pendingConnectionId.get()).toBeNull()
  })

  it('a slow-but-healthy cold dial is not a failed pick: it caps only past the ssh layer budget', async () => {
    vi.useFakeTimers()

    try {
      const dial = deferred<void>()

      setConnectionsRegistry(registry)
      $connection.set({ connectionId: 'local', mode: 'local' })
      ensureGatewayAgent.mockImplementationOnce(async (connectionId, profile) => {
        await dial.promise
        $connection.set({ connectionId: connectionId ?? undefined, mode: 'remote', profile, registryScoped: true })
      })

      let outcome = 'pending'

      const attempt = selectConnection('homelab').then(
        () => {
          outcome = 'resolved'
        },
        (error: Error) => {
          outcome = error.message
        }
      )

      // A cold remote dial (ssh connect, locate/version execs, the spawned
      // backend's ready sentinel, the forward) routinely runs 37-64 s.
      await vi.advanceTimersByTimeAsync(60_000)
      expect(outcome).toBe('pending')
      expect($pendingConnectionId.get()).toBe('homelab')

      dial.resolve()
      await attempt

      expect(outcome).toBe('resolved')
      expect($connection.get()?.connectionId).toBe('homelab')
    } finally {
      vi.useRealTimers()
    }
  })

  it('a dial that never answers times out visibly, cannot publish later, and the gateway can be retried', async () => {
    vi.useFakeTimers()

    try {
      const activation = deferred()

      setConnectionsRegistry(registry)
      $connection.set({ connectionId: 'local', mode: 'local' })
      ensureGatewayAgent.mockImplementationOnce(async (connectionId, _profile, options) => {
        await activation.promise

        // Mirrors the real door: ownership is checked right before publishing.
        if (options?.signal?.aborted) {
          return
        }

        $connection.set({
          connectionId: connectionId ?? undefined,
          mode: 'remote',
          profile: 'default',
          registryScoped: true
        })
      })

      const outcome = selectConnection('homelab').then(
        () => 'resolved',
        (error: Error) => error.message
      )

      await vi.advanceTimersByTimeAsync(SOURCE_SWITCH_DIAL_TIMEOUT_MS)

      expect(await outcome).toMatch(/Timed out connecting to "Homelab"/)
      expect($pendingConnectionId.get()).toBeNull()

      activation.resolve()
      await vi.advanceTimersByTimeAsync(0)
      expect($connection.get()?.connectionId).toBe('local')

      await selectConnection('homelab')
      expect($connection.get()?.connectionId).toBe('homelab')
    } finally {
      vi.useRealTimers()
    }
  })

  it('an activation that published but never settled counts as landed', async () => {
    vi.useFakeTimers()

    try {
      setConnectionsRegistry(registry)
      $connection.set({ connectionId: 'local', mode: 'local' })
      // The socket activates and publishes; only a trailing descriptor resync stalls.
      ensureGatewayAgent.mockImplementationOnce((connectionId, _profile, options) => {
        options?.beforeActivate?.()
        $connection.set({
          connectionId: connectionId ?? undefined,
          mode: 'remote',
          profile: 'default',
          registryScoped: true
        })

        return new Promise<void>(() => undefined)
      })

      const attempt = selectConnection('homelab')
      await vi.advanceTimersByTimeAsync(SOURCE_SWITCH_DIAL_TIMEOUT_MS)
      await attempt

      expect($connection.get()?.connectionId).toBe('homelab')
      expect(setLastUsed).toHaveBeenCalledWith('homelab')
      expect(requestFreshSession).toHaveBeenCalledTimes(1)
      expect($pendingConnectionId.get()).toBeNull()
    } finally {
      vi.useRealTimers()
    }
  })

  it('waits for the primary descriptor before restoring a source at boot', async () => {
    // The sidebar mounts while the primary gateway is still booting. Dialing
    // the preferred source before its descriptor is published can create a
    // second SSH backend for the exact same registered connection.
    list.mockResolvedValueOnce({ ...registry, lastUsed: 'homelab', launchMode: 'last-used' })
    $showAllProfiles.set(true)

    const restoring = initializeConnectionsRegistry()

    await new Promise(resolve => setTimeout(resolve, 0))
    expect(ensureGatewayAgent).not.toHaveBeenCalled()

    $connection.set({ connectionId: 'homelab', mode: 'remote' })
    await restoring

    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect(setLastUsed).toHaveBeenCalledWith('homelab')
    expect($showAllProfiles.get()).toBe(true)
  })

  it('boot restore proceeds after the descriptor wait deadline (bounded wait)', { timeout: 30_000 }, async () => {
    // A primary that never publishes (spawn failure, dead SSH target) must
    // not strand the registry restore forever: after the deadline the restore
    // runs exactly as it did before the wait existed.
    vi.useFakeTimers()

    try {
      list.mockResolvedValueOnce({ ...registry, lastUsed: 'homelab', launchMode: 'last-used' })

      const restoring = initializeConnectionsRegistry()

      await vi.advanceTimersByTimeAsync(1_000)
      expect(ensureGatewayAgent).not.toHaveBeenCalled()

      // Descriptor never arrives; deadline elapses.
      await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS + 15_000)
      await restoring

      expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
    } finally {
      vi.useRealTimers()
    }
  })

  it('a user-initiated source switch still collapses "All profiles"', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    $showAllProfiles.set(true)

    await selectConnection('homelab')

    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
    expect($showAllProfiles.get()).toBe(false)
  })

  it('a boot-time restore of the last-used source leaves "All profiles" on', async () => {
    list.mockResolvedValueOnce({ ...registry, lastUsed: 'homelab', launchMode: 'last-used' })
    // Main opens the local backend before the registry settles, so a source is
    // already published when the restore runs. The old null-id inference read
    // that as a user-initiated switch and collapsed the browse view on restart.
    $connection.set({ connectionId: 'local', mode: 'local' })
    $showAllProfiles.set(true)

    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
    expect($showAllProfiles.get()).toBe(true)
  })

  it('restores the last-used profile on switch-back even when the primary local descriptor is profile-less', async () => {
    // The pair is remembered while the primary local descriptor is honest
    // about its profile (boot publication).
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local', profile: 'mac', registryScoped: true })
    $activeGatewayProfile.set('mac')

    expect(JSON.parse(localStorage.getItem('hermes.desktop.lastProfileByConnection') || '{}')).toEqual({
      local: 'mac'
    })

    // A later resync republishes a profile-less primary descriptor (the
    // startHermes shape). The remembered pair is the authority for "what was
    // last used here" — picking away and back must still restore 'mac', and
    // the pick must not die in targetIsActive() on the descriptor gap.
    await selectConnection('homelab')
    expect(ensureGatewayAgent).toHaveBeenLastCalledWith('homelab', 'default', expect.anything())

    await selectConnection('local')

    expect(ensureGatewayAgent).toHaveBeenLastCalledWith('local', 'mac', expect.anything())
    expect($newChatProfile.get()).toBe('mac')
  })

  it('never re-homes a live connection the registry cannot name', async () => {
    // A window connected through the legacy v1 route carries an unqualified
    // descriptor, so resolvedConnectionId — and therefore $activeConnectionId
    // — is null. Restoring the registry primary over it would re-home a
    // working remote onto local a few seconds after boot.
    list.mockResolvedValueOnce({ ...registry, launchMode: 'primary', primary: 'local' })
    $connection.set({ mode: 'remote', profile: 'default', registryScoped: false })

    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect($connection.get()?.mode).toBe('remote')
  })
})
