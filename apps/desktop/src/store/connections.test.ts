import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'

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

const ensureGatewayAgent = vi.fn(async (_connectionId: null | string, _profile: string): Promise<void> => undefined)
const openGatewayAgent = vi.fn(async (_connectionId: string, _profile: string): Promise<void> => undefined)
const refreshActiveProfile = vi.fn(async () => undefined)
const requestFreshSession = vi.fn()
const beforeConnectionSwitch = vi.fn()
const wipeSessionListsForGatewaySwitch = vi.fn(() => $activeSessionId.set(null))

// Test double for the store's commit point with the real one's contract
// (barrier → machine-context reset → wipe, synchronously); the real
// implementation is covered by gateway-switch.test.ts.
const beginGatewaySwitch = vi.fn(() => {
  $gatewaySwitching.set(true)
  beforeConnectionSwitch()
  wipeSessionListsForGatewaySwitch()
})

const endGatewaySwitch = vi.fn(() => $gatewaySwitching.set(false))
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
  ensureGatewayAgent.mockImplementation(async connectionId => {
    $connection.set({
      connectionId: connectionId ?? undefined,
      mode: connectionId === 'local' ? 'local' : 'remote',
      profile: 'default',
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
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default')
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
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default')
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
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default')
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

    expect(ensureGatewayAgent).toHaveBeenCalledWith('local', 'default')
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
    expect(ensureGatewayAgent.mock.calls).toEqual([['local', 'default']])
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

    expect(ensureGatewayAgent).toHaveBeenCalledWith('local', 'research')
  })

  it('does not remember a migrated v1 routing alias as a backend profile', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'homelab', mode: 'remote' })
    $activeGatewayProfile.set('legacy-homelab-alias')
    $connection.set({ connectionId: 'local', mode: 'local', registryScoped: true })

    await selectConnection('homelab')

    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default')
  })

  it('does not remember a stale startup profile under the resolved source', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local', profile: 'default', registryScoped: true })
    $activeGatewayProfile.set('work-agent')
    $connection.set({ connectionId: 'homelab', mode: 'remote', profile: 'default', registryScoped: true })
    $activeGatewayProfile.set('default')

    await selectConnection('local')

    expect(ensureGatewayAgent).toHaveBeenCalledWith('local', 'default')
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
    // edited/removed mid-switch): $connection never moves to homelab.
    ensureGatewayAgent.mockImplementationOnce(async () => undefined)

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
    // dial → commit (barrier + reset + wipe) → activate, in that order.
    expect(openGatewayAgent).toHaveBeenCalledWith('homelab', 'default')
    expect(openGatewayAgent.mock.invocationCallOrder[0]).toBeLessThan(beginGatewaySwitch.mock.invocationCallOrder[0])
    expect(beginGatewaySwitch.mock.invocationCallOrder[0]).toBeLessThan(ensureGatewayAgent.mock.invocationCallOrder[0])
    expect(beforeConnectionSwitch).toHaveBeenCalledTimes(1)
    expect(endGatewaySwitch).toHaveBeenCalledTimes(1)
    expect($gatewaySwitching.get()).toBe(false)
    expect($activeSessionId.get()).toBeNull()
  })

  it('boot-time restore leaves "All profiles" browse mode on (#93197)', async () => {
    // Fresh boot: nothing active yet, registry restores last-used. The
    // persisted showAllProfiles=true must survive the silent restore.
    list.mockResolvedValueOnce({ ...registry, lastUsed: 'homelab', launchMode: 'last-used' })
    $showAllProfiles.set(true)

    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default')
    expect($showAllProfiles.get()).toBe(true)
  })

  it('a user-initiated source switch still collapses "All profiles"', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    $showAllProfiles.set(true)

    await selectConnection('homelab')

    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default')
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
