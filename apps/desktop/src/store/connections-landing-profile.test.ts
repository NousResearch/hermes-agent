// A pick that names no profile, on a host that can read which profile the picked gateway runs as
// (the hosted dashboard): it lands on that profile, not on a `default` the gateway may not serve.
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type { DesktopConnectionsRegistry } from '@/global'

import { deferred } from '../test/deferred'

const $activeGatewayProfile = atom('default')
const $newChatProfile = atom<null | string>(null)
const $newChatRoute = atom<null | { connectionId: string; profile: string }>(null)
const $newChatConnectionId = atom<null | string>(null)
// Every new-chat owner intent bumps this, like the real profile store.
let newChatIntent = 0

const captureNewChatSource = vi.fn((connectionId: null | string = null) => {
  newChatIntent += 1
  $newChatConnectionId.set(connectionId)
})

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
const isLegacyNewChatProfile = vi.fn((_profile: string) => false)
const pinLegacyNewChatProfile = vi.fn()

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
  $newChatConnectionId,
  $newChatProfile,
  $newChatRoute,
  $showAllProfiles,
  captureNewChatSource,
  currentNewChatIntent: () => newChatIntent,
  ensureGatewayAgent,
  isLegacyNewChatProfile,
  normalizeProfileKey: (name: null | string | undefined) => (name ?? '').trim() || 'default',
  openGatewayAgent,
  pinLegacyNewChatProfile,
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
// The profile each listed agent's gateway runs as, as a host with no agent of its own reads it.
const launchProfile = vi.fn(async (_connectionId: string): Promise<string | undefined> => 'research')
const setLastUsed = vi.fn(async (id: string) => ({ ok: true, registry: { ...registry, lastUsed: id } }))

beforeEach(() => {
  localStorage.clear()
  _resetConnectionsForTests()
  $connectionsRegistry.set(null)
  $connection.set(null)
  $activeGatewayProfile.set('default')
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  $newChatConnectionId.set(null)
  newChatIntent = 0
  $showAllProfiles.set(false)
  isLegacyNewChatProfile.mockReset()
  isLegacyNewChatProfile.mockReturnValue(false)
  pinLegacyNewChatProfile.mockClear()
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
  launchProfile.mockReset()
  launchProfile.mockResolvedValue('research')
  api.mockReset()
  api.mockResolvedValue({ profiles: [] })
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.stubGlobal('window', {
    hermesDesktop: { api, connections: { launchProfile, list, setLastUsed } },
    localStorage,
    location: window.location
  })
})

afterEach(() => vi.unstubAllGlobals())


describe('a pick that names no profile', () => {
  it('lands on the profile the picked gateway runs as', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })

    await selectConnection('homelab')

    expect(launchProfile).toHaveBeenCalledWith('homelab')
    expect(openGatewayAgent).toHaveBeenCalledWith('homelab', 'research')
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'research', expect.anything())
    expect(ensureGatewayAgent).not.toHaveBeenCalledWith('homelab', 'default', expect.anything())
    expect($newChatProfile.get()).toBe('research')
  })

  it('keeps an explicit or remembered profile without reading', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })

    await selectConnection('work-vps', { profile: 'ops' })
    expect(ensureGatewayAgent).toHaveBeenLastCalledWith('work-vps', 'ops', expect.anything())

    // Remember "scout" on homelab, as a landed activation does.
    $activeGatewayProfile.set('scout')
    $connection.set({ connectionId: 'homelab', mode: 'remote', profile: 'scout', registryScoped: true })
    $activeGatewayProfile.set('default')
    $connection.set({ connectionId: 'local', mode: 'local' })
    await selectConnection('homelab')

    expect(launchProfile).not.toHaveBeenCalled()
    expect(ensureGatewayAgent).toHaveBeenLastCalledWith('homelab', 'scout', expect.anything())
  })

  it('restores the last-used gateway on its own profile', async () => {
    list.mockResolvedValueOnce({ ...registry, lastUsed: 'homelab', launchMode: 'last-used' })
    $connection.set({ connectionId: 'local', mode: 'local' })

    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'research', expect.anything())
    expect(setLastUsed).toHaveBeenCalledWith('homelab')
  })

  it('fails without dialing when the profile cannot be read', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    launchProfile.mockRejectedValueOnce(new Error('relay down'))

    await expect(selectConnection('homelab')).rejects.toThrow('relay down')

    expect(openGatewayAgent).not.toHaveBeenCalled()
    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect($activeConnectionId.get()).toBe('local')
  })

  it('a gateway that names no profile lands on default', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    launchProfile.mockResolvedValueOnce(undefined)

    await selectConnection('homelab')

    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
  })

  // Boot's last-used restore runs once the list is read; a pick the user made before that, still
  // reading its profile, has said where the window goes.
  it("is not overtaken by boot's restore while its profile is read", async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    list.mockResolvedValueOnce({ ...registry, lastUsed: 'work-vps', launchMode: 'last-used' })
    const read = deferred<string | undefined>()
    launchProfile.mockImplementationOnce(() => read.promise)

    const pick = selectConnection('homelab')
    await initializeConnectionsRegistry()
    read.resolve('research')
    await pick

    expect(ensureGatewayAgent).toHaveBeenCalledTimes(1)
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'research', expect.anything())
    expect($activeConnectionId.get()).toBe('homelab')
  })

  // The newer pick owns the switch from the click: an older one still connecting must not land
  // while the newer one reads its profile, nor stay once that read fails.
  it('stops an older pick still connecting as soon as a newer one starts reading', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    const opening = deferred<void>()
    openGatewayAgent.mockImplementationOnce(() => opening.promise)
    launchProfile.mockRejectedValueOnce(new Error('relay down'))

    const older = selectConnection('work-vps', { profile: 'ops' })
    await expect(selectConnection('homelab')).rejects.toThrow('relay down')
    opening.resolve()
    await older

    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect($activeConnectionId.get()).toBe('local')
    expect($pendingConnectionId.get()).toBeNull()
  })

  it('leaves a draft the user started while the profile was read to the user', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    const read = deferred<string | undefined>()
    launchProfile.mockImplementationOnce(() => read.promise)

    const pick = selectConnection('homelab')
    captureNewChatSource('local')
    read.resolve('research')
    await pick

    expect($activeConnectionId.get()).toBe('homelab')
    expect($newChatConnectionId.get()).toBe('local')
    expect($newChatProfile.get()).toBeNull()
  })

  it('a newer pick made while the profile is read wins', async () => {
    setConnectionsRegistry(registry)
    $connection.set({ connectionId: 'local', mode: 'local' })
    const read = deferred<string | undefined>()
    launchProfile.mockImplementationOnce(() => read.promise)

    const first = selectConnection('homelab')
    await selectConnection('work-vps', { profile: 'ops' })
    read.resolve('research')
    await first

    expect(ensureGatewayAgent).toHaveBeenCalledTimes(1)
    expect(ensureGatewayAgent).toHaveBeenCalledWith('work-vps', 'ops', expect.anything())
    expect($activeConnectionId.get()).toBe('work-vps')
  })
})
