import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'

import { deferred } from '../test/deferred'

const $activeGatewayProfile = atom('default')
const $newChatProfile = atom<null | string>(null)
const $showAllProfiles = atom(false)

const $connection = atom<null | {
  connectionId?: string
  mode?: 'local' | 'remote'
  profile?: string
  registryScoped?: boolean
}>(null)

// The runtime session id minted by the CURRENT backend — the binding a switch
// must sever before the next source is published (#93937).
const $activeSessionId = atom<null | string>(null)
const $gatewaySwitching = atom(false)

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
const beforeConnectionSwitch = vi.fn()
const wipeSessionListsForGatewaySwitch = vi.fn(() => $activeSessionId.set(null))

// Test double for the store's commit point with the real one's contract
// (barrier → machine-context reset → wipe, synchronously; the barrier is
// owned by the latest token); the real implementation is covered by
// gateway-switch.test.ts.
let latestSwitchToken = 0

const beginGatewaySwitch = vi.fn(() => {
  $gatewaySwitching.set(true)
  beforeConnectionSwitch()
  wipeSessionListsForGatewaySwitch()

  return ++latestSwitchToken
})

const endGatewaySwitch = vi.fn((token?: number) => {
  if (token === undefined || token === latestSwitchToken) {
    $gatewaySwitching.set(false)
  }
})

const recoverActiveSourceAfterFailedGatewaySwitch = vi.fn()

vi.mock('@/store/session', () => ({ $connection }))
vi.mock('@/store/gateway-switch', () => ({
  $gatewaySwitching,
  beginGatewaySwitch,
  endGatewaySwitch,
  recoverActiveSourceAfterFailedGatewaySwitch,
  wipeSessionListsForGatewaySwitch
}))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile,
  $newChatProfile,
  $showAllProfiles,
  captureNewChatSource: vi.fn(),
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
  // Mirrors the real door: the commit hook runs right before the activation
  // publishes, and a declined hook publishes nothing.
  ensureGatewayAgent.mockImplementation(async (connectionId, profile, options) => {
    if (options?.beforeActivate && !options.beforeActivate()) {
      return
    }

    $connection.set({
      connectionId: connectionId ?? undefined,
      mode: connectionId === 'local' ? 'local' : 'remote',
      profile,
      registryScoped: true
    })
  })
  openGatewayAgent.mockReset()
  openGatewayAgent.mockResolvedValue(undefined)
  refreshActiveProfile.mockClear()
  requestFreshSession.mockClear()
  beforeConnectionSwitch.mockClear()
  beginGatewaySwitch.mockClear()
  endGatewaySwitch.mockClear()
  recoverActiveSourceAfterFailedGatewaySwitch.mockClear()
  wipeSessionListsForGatewaySwitch.mockClear()
  $activeSessionId.set(null)
  $gatewaySwitching.set(false)
  list.mockClear()
  setLastUsed.mockClear()
  vi.stubGlobal('window', { hermesDesktop: { connections: { list, setLastUsed } }, localStorage })
})

afterEach(() => vi.unstubAllGlobals())

describe('connection registry cache', () => {
  it('loads only Electron local registry state', async () => {
    await refreshConnectionsRegistry()

    expect(list).toHaveBeenCalledTimes(1)
    expect($connectionsRegistry.get()).toEqual(registry)
    expect($activeConnectionId.get()).toBeNull()
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

  it('restores a remote registry primary through its exact connection id', async () => {
    list.mockResolvedValueOnce({ ...registry, primary: 'homelab', launchMode: 'primary' })
    $connection.set({ connectionId: 'local', mode: 'local' })

    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).toHaveBeenCalledTimes(1)
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
    expect(setLastUsed).toHaveBeenCalledWith('homelab')
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
  it('dials a secondary source and starts a fresh source-scoped draft', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })

    await selectConnection('homelab')

    expect(openGatewayAgent).toHaveBeenCalledWith('homelab', 'default')
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
    expect(beforeConnectionSwitch).toHaveBeenCalledTimes(1)
    expect(requestFreshSession).toHaveBeenCalledTimes(1)
    expect(wipeSessionListsForGatewaySwitch).toHaveBeenCalledTimes(1)
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
  })

  it('lets a later source choice win while an earlier dial is still pending', async () => {
    let releaseDials!: () => void

    const dialGate = new Promise<void>(resolve => {
      releaseDials = resolve
    })

    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    openGatewayAgent.mockImplementation(async () => {
      await dialGate
    })

    const openHomelab = selectConnection('homelab')
    await Promise.resolve()
    const stayLocal = selectConnection('local')

    releaseDials()
    await Promise.all([openHomelab, stayLocal])

    expect(openGatewayAgent.mock.calls).toEqual([
      ['homelab', 'default'],
      ['local', 'default']
    ])
    // The superseded dial never activates: the user doesn't flip through
    // homelab on the way back to local, and only the winner commits.
    expect(ensureGatewayAgent.mock.calls.map(call => [call[0], call[1]])).toEqual([['local', 'default']])
    expect(beginGatewaySwitch).toHaveBeenCalledTimes(1)
    expect(wipeSessionListsForGatewaySwitch).toHaveBeenCalledTimes(1)
    // Only the latest intent repaints the profile list.
    expect(refreshActiveProfile).toHaveBeenCalledTimes(1)
    expect($connection.get()?.connectionId).toBe('local')
    expect($gatewaySwitching.get()).toBe(false)
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

  it('keeps the current source usable when a dial fails: nothing is severed before the target is reachable', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    $activeSessionId.set('a93bb39d')
    openGatewayAgent.mockRejectedValueOnce(new Error('offline'))

    await expect(selectConnection('homelab')).rejects.toThrow('offline')

    // The dial failed in phase 1 — the switch never committed, so the open
    // transcript, its runtime binding and the session lists are all intact.
    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect(beginGatewaySwitch).not.toHaveBeenCalled()
    expect(beforeConnectionSwitch).not.toHaveBeenCalled()
    expect(wipeSessionListsForGatewaySwitch).not.toHaveBeenCalled()
    expect($activeSessionId.get()).toBe('a93bb39d')
    expect($gatewaySwitching.get()).toBe(false)
    expect(requestFreshSession).not.toHaveBeenCalled()
    expect($newChatProfile.get()).toBeNull()
    expect($pendingConnectionId.get()).toBeNull()
    expect(setLastUsed).not.toHaveBeenCalled()
    expect($connection.get()?.connectionId).toBe('local')
  })

  it('an activation that does not land after the wipe lowers the barrier and repaints the still-active source', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    // The dial opened the socket but the activation was declined (source
    // edited/removed mid-switch): the commit hook ran (wipe done), yet
    // $connection never moves to homelab.
    ensureGatewayAgent.mockImplementationOnce(async (_connectionId, _profile, options) => {
      options?.beforeActivate?.()
    })

    await expect(selectConnection('homelab')).rejects.toThrow('did not become active')

    expect(beginGatewaySwitch).toHaveBeenCalledTimes(1)
    expect(endGatewaySwitch).toHaveBeenCalledTimes(1)
    expect($gatewaySwitching.get()).toBe(false)
    // The lists were wiped for a commit that never happened; the source that
    // is still active gets repainted and the user lands on a fresh draft there.
    expect(recoverActiveSourceAfterFailedGatewaySwitch).toHaveBeenCalledTimes(1)
    expect(requestFreshSession).toHaveBeenCalledTimes(1)
    expect(setLastUsed).not.toHaveBeenCalled()
    expect($newChatProfile.get()).toBeNull()
    expect($pendingConnectionId.get()).toBeNull()
    expect($connection.get()?.connectionId).toBe('local')
  })

  it("#93937: severs the previous source's runtime session binding BEFORE the new source is published, behind the barrier", async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    // Runtime id minted by the local backend; only local has ever heard of it.
    $activeSessionId.set('a93bb39d')

    // Phase 1 (the dial) must not touch the current workspace at all.
    openGatewayAgent.mockImplementationOnce(async () => {
      expect($activeSessionId.get()).toBe('a93bb39d')
      expect($gatewaySwitching.get()).toBe(false)
      expect(wipeSessionListsForGatewaySwitch).not.toHaveBeenCalled()
    })

    // Every publication of the new source, with what a session-scoped effect
    // would read at that instant.
    const published: Array<{ activeSessionId: null | string; connectionId?: string; switching: boolean }> = []

    const off = $connection.listen(next => {
      published.push({
        activeSessionId: $activeSessionId.get(),
        connectionId: next?.connectionId,
        switching: $gatewaySwitching.get()
      })
    })

    await selectConnection('homelab')
    off()

    // The old runtime id was already gone when homelab became visible, and
    // the barrier was up — nothing could pair 'a93bb39d' with the new backend.
    expect(published).toEqual([{ activeSessionId: null, connectionId: 'homelab', switching: true }])
    // dial → commit (barrier + reset + wipe, inside the activation's commit
    // hook) → publish, in that order.
    expect(openGatewayAgent).toHaveBeenCalledWith('homelab', 'default')
    expect(openGatewayAgent.mock.invocationCallOrder[0]).toBeLessThan(ensureGatewayAgent.mock.invocationCallOrder[0])
    expect(beginGatewaySwitch).toHaveBeenCalledTimes(1)
    expect(beforeConnectionSwitch).toHaveBeenCalledTimes(1)
    expect(endGatewaySwitch).toHaveBeenCalledTimes(1)
    expect($gatewaySwitching.get()).toBe(false)
    expect($activeSessionId.get()).toBeNull()
  })

  it('a click that supersedes a QUEUED commit wins: the superseded switch neither wipes nor activates', async () => {
    // Both activations sit behind an in-flight profile/agent switch (the
    // profile store's mutex); their commit hooks only run once it settles.
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

    // Nothing is severed while both are still queued.
    expect(beginGatewaySwitch).not.toHaveBeenCalled()
    expect($activeSessionId.get()).toBe('a93bb39d')

    mutex.resolve()
    await Promise.all([first, second])
    off()

    // homelab stepped aside at its turn: no wipe, no publication, no error
    // UI; work-vps wiped once and is the only source ever published.
    expect(ensureGatewayAgent.mock.calls.map(call => call[0])).toEqual(['homelab', 'work-vps'])
    expect(beginGatewaySwitch).toHaveBeenCalledTimes(1)
    expect(published).toEqual(['work-vps:active=null'])
    expect($connection.get()?.connectionId).toBe('work-vps')
    expect(recoverActiveSourceAfterFailedGatewaySwitch).not.toHaveBeenCalled()
    expect(requestFreshSession).toHaveBeenCalledTimes(1)
    expect(setLastUsed).toHaveBeenCalledTimes(1)
    expect(setLastUsed).toHaveBeenCalledWith('work-vps')
    expect($gatewaySwitching.get()).toBe(false)
    expect($pendingConnectionId.get()).toBeNull()
  })

  it('does not spend the activation timeout while waiting for the serialized commit turn', async () => {
    vi.useFakeTimers()

    try {
      const mutex = deferred()

      setConnectionsRegistry(registry)
      $connection.set({ connectionId: 'local', mode: 'local' })
      ensureGatewayAgent.mockImplementationOnce(async (connectionId, _profile, options) => {
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

      const attempt = selectConnection('homelab')
      await vi.waitFor(() => expect(ensureGatewayAgent).toHaveBeenCalledTimes(1))

      // Queue ownership belongs to the shared mutex, not the actual activation
      // attempt, so waiting here must not consume its 20-second commit budget.
      await vi.advanceTimersByTimeAsync(20_000)
      expect(beginGatewaySwitch).not.toHaveBeenCalled()

      mutex.resolve()
      await attempt

      expect($connection.get()?.connectionId).toBe('homelab')
      expect(beginGatewaySwitch).toHaveBeenCalledTimes(1)
      expect($gatewaySwitching.get()).toBe(false)
    } finally {
      vi.useRealTimers()
    }
  })

  it('a dial that never answers times out: nothing severed, the click fails visibly, and the source can be retried', async () => {
    vi.useFakeTimers()

    try {
      setConnectionsRegistry(registry)
      $connection.set({ connectionId: 'local', mode: 'local' })
      $activeSessionId.set('a93bb39d')
      openGatewayAgent.mockImplementationOnce(() => new Promise<void>(() => undefined))

      const outcome = selectConnection('homelab').then(
        () => 'resolved',
        (error: Error) => error.message
      )

      await vi.advanceTimersByTimeAsync(20_000)

      expect(await outcome).toMatch(/Timed out connecting to "Homelab"/)
      expect(ensureGatewayAgent).not.toHaveBeenCalled()
      expect(beginGatewaySwitch).not.toHaveBeenCalled()
      expect($activeSessionId.get()).toBe('a93bb39d')
      expect($gatewaySwitching.get()).toBe(false)
      expect($pendingConnectionId.get()).toBeNull()

      // The stalled click does not poison the source: a retry is a real switch,
      // not a duplicate of the pending one.
      await selectConnection('homelab')

      expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
      expect($connection.get()?.connectionId).toBe('homelab')
    } finally {
      vi.useRealTimers()
    }
  })

  it('an activation that published the new source but never settled counts as committed', async () => {
    vi.useFakeTimers()

    try {
      setConnectionsRegistry(registry)
      $connection.set({ connectionId: 'local', mode: 'local' })
      // The socket activates and publishes synchronously; only the trailing
      // descriptor resync (an IPC) stalls.
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
      await vi.advanceTimersByTimeAsync(20_000)
      await attempt

      expect($connection.get()?.connectionId).toBe('homelab')
      expect($gatewaySwitching.get()).toBe(false)
      expect(setLastUsed).toHaveBeenCalledWith('homelab')
      expect(requestFreshSession).toHaveBeenCalledTimes(1)
      expect(recoverActiveSourceAfterFailedGatewaySwitch).not.toHaveBeenCalled()
      expect($pendingConnectionId.get()).toBeNull()
    } finally {
      vi.useRealTimers()
    }
  })

  it('a timed-out activation that already published cannot republish after a newer source wins', async () => {
    vi.useFakeTimers()

    try {
      let releaseDescriptor: () => void = () => undefined

      setConnectionsRegistry(registry)
      // Seed A's remembered profile through the real source/profile observer,
      // then restore the currently active local source.
      $activeGatewayProfile.set('research')
      $connection.set({ connectionId: 'homelab', mode: 'remote', profile: 'research', registryScoped: true })
      $activeGatewayProfile.set('default')
      $connection.set({ connectionId: 'local', mode: 'local', profile: 'default', registryScoped: true })

      ensureGatewayAgent
        .mockImplementationOnce((connectionId, profile, options) => {
          options?.beforeActivate?.()

          // Low-level activation publishes synchronously. The trailing descriptor
          // promise remains alive beyond selectConnection's commit timeout.
          $activeGatewayProfile.set(profile)
          $connection.set({
            connectionId: connectionId ?? undefined,
            mode: 'remote',
            profile,
            registryScoped: true
          })

          return new Promise<void>(resolve => {
            releaseDescriptor = () => {
              // Mirrors ensureGatewayAgent's publication seam: a revoked owner
              // observes its signal and must not publish its late descriptor.
              if (!options?.signal?.aborted) {
                $activeGatewayProfile.set(profile)
                $connection.set({
                  connectionId: connectionId ?? undefined,
                  mode: 'remote',
                  profile,
                  registryScoped: true
                })
              }

              resolve()
            }
          })
        })
        .mockImplementationOnce(async (connectionId, profile, options) => {
          if (options?.beforeActivate && !options.beforeActivate()) {
            return
          }

          $activeGatewayProfile.set(profile)
          $connection.set({
            connectionId: connectionId ?? undefined,
            mode: 'remote',
            profile,
            registryScoped: true
          })
        })

      const timedOutOwner = selectConnection('homelab')
      await vi.advanceTimersByTimeAsync(20_000)
      await timedOutOwner

      // Fail open: A really did become active before its trailing descriptor
      // work timed out, so the commit remains successful.
      expect($activeGatewayProfile.get()).toBe('research')
      expect($connection.get()?.connectionId).toBe('homelab')

      await selectConnection('work-vps')
      expect($activeGatewayProfile.get()).toBe('default')
      expect($connection.get()?.connectionId).toBe('work-vps')

      releaseDescriptor()
      await vi.advanceTimersByTimeAsync(0)

      expect($activeGatewayProfile.get()).toBe('default')
      expect($connection.get()?.connectionId).toBe('work-vps')
    } finally {
      vi.useRealTimers()
    }
  })

  it('an activation that stalls AFTER the wipe times out: barrier down, still-active source repainted', async () => {
    vi.useFakeTimers()

    try {
      setConnectionsRegistry(registry)
      $connection.set({ connectionId: 'local', mode: 'local' })
      ensureGatewayAgent.mockImplementationOnce((_connectionId, _profile, options) => {
        options?.beforeActivate?.()

        return new Promise<void>(() => undefined)
      })

      const outcome = selectConnection('homelab').then(
        () => 'resolved',
        (error: Error) => error.message
      )

      await vi.advanceTimersByTimeAsync(20_000)

      expect(await outcome).toMatch(/Timed out activating "Homelab"/)
      expect(beginGatewaySwitch).toHaveBeenCalledTimes(1)
      expect($gatewaySwitching.get()).toBe(false)
      expect(recoverActiveSourceAfterFailedGatewaySwitch).toHaveBeenCalledTimes(1)
      expect(requestFreshSession).toHaveBeenCalledTimes(1)
      expect(setLastUsed).not.toHaveBeenCalled()
      expect($connection.get()?.connectionId).toBe('local')
      expect($pendingConnectionId.get()).toBeNull()
    } finally {
      vi.useRealTimers()
    }
  })

  it('a timed-out activation cannot publish the target after it eventually settles', async () => {
    vi.useFakeTimers()

    try {
      const activation = deferred()

      setConnectionsRegistry(registry)
      $connection.set({ connectionId: 'local', mode: 'local' })
      ensureGatewayAgent.mockImplementationOnce(async (connectionId, _profile, options) => {
        options?.beforeActivate?.()
        await activation.promise

        // Mirrors the real activation door: cancellation ownership is checked
        // immediately before publishing after the async activation work.
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

      await vi.advanceTimersByTimeAsync(20_000)

      expect(await outcome).toMatch(/Timed out activating "Homelab"/)
      expect($connection.get()?.connectionId).toBe('local')
      expect($gatewaySwitching.get()).toBe(false)

      activation.resolve()
      await vi.advanceTimersByTimeAsync(0)

      expect($connection.get()?.connectionId).toBe('local')
      expect(beginGatewaySwitch).toHaveBeenCalledTimes(1)
      expect(endGatewaySwitch).toHaveBeenCalledTimes(1)
      expect($gatewaySwitching.get()).toBe(false)
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
    expect(openGatewayAgent).not.toHaveBeenCalled()
    expect(ensureGatewayAgent).not.toHaveBeenCalled()

    $connection.set({ connectionId: 'homelab', mode: 'remote' })
    await restoring

    expect(openGatewayAgent).not.toHaveBeenCalled()
    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect(setLastUsed).toHaveBeenCalledWith('homelab')
    expect($showAllProfiles.get()).toBe(true)
  })

  it('a user-initiated source switch still collapses "All profiles"', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    $showAllProfiles.set(true)

    await selectConnection('homelab')

    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
    expect($showAllProfiles.get()).toBe(false)
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
    expect(wipeSessionListsForGatewaySwitch).not.toHaveBeenCalled()
    expect($connection.get()?.mode).toBe('remote')
  })
})
