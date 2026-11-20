import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopAgentRoster, DesktopConnectionsRegistry } from '@/global'

import { deferred } from '../test/deferred'

// A launch restore whose target is down falls back to the source already in
// front. The target recovering later is something that happened in the
// background, so it must not move the window.

const $activeGatewayProfile = atom('default')
const $freshSessionRequest = atom(0)
let newChatIntent = 0

const $connection = atom<null | {
  connectionId?: string
  mode?: 'local' | 'remote'
  profile?: string
  registryScoped?: boolean
}>(null)

const ensureGatewayAgent = vi.fn(
  async (connectionId: null | string, profile: string, options?: { beforeActivate?: () => boolean }) => {
    if (options?.beforeActivate && !options.beforeActivate()) {
      return
    }

    $connection.set({ connectionId: connectionId ?? undefined, mode: 'remote', profile, registryScoped: true })
  }
)

const openGatewayAgent = vi.fn(async (_connectionId: string, _profile: string): Promise<void> => undefined)

vi.mock('@/store/session', () => ({
  $connection,
  $activeSessionId: atom(null),
  $selectedStoredSessionId: atom(null),
  setActiveSessionId: vi.fn(),
  setSelectedStoredSessionId: vi.fn()
}))
vi.mock('@/store/project-scope', () => ({ exitProjectScope: vi.fn() }))
vi.mock('@/store/gateway-switch', () => ({ supersedeGatewaySwitch: vi.fn() }))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile,
  $freshSessionRequest,
  $newChatConnectionId: atom(null),
  $newChatProfile: atom(null),
  $newChatRoute: atom(null),
  $showAllProfiles: atom(false),
  captureNewChatSource: vi.fn(() => { newChatIntent += 1 }),
  currentNewChatIntent: () => newChatIntent,
  ensureGatewayAgent,
  isLegacyNewChatProfile: () => false,
  normalizeProfileKey: (name: null | string | undefined) => (name ?? '').trim() || 'default',
  openGatewayAgent,
  pinLegacyNewChatProfile: vi.fn(),
  refreshActiveProfile: vi.fn(async () => undefined),
  requestFreshSession: vi.fn()
}))

const { $pendingConnectionId, _resetConnectionsForTests, initializeConnectionsRegistry } = await import('./connections')
const { _resetFleetRosterForTests } = await import('./fleet-roster')

const registry: DesktopConnectionsRegistry = {
  connections: [
    { id: 'local', kind: 'local', label: 'This device', tokenPreview: null, tokenSet: false },
    { id: 'homelab', kind: 'remote', label: 'Homelab', tokenPreview: '...abc', tokenSet: true }
  ],
  lastUsed: 'homelab',
  launchMode: 'last-used',
  primary: 'local',
  secureTokenStorage: true,
  version: 2
}

function roster(homelab: { reachable: boolean; error?: string }): DesktopAgentRoster {
  return {
    agents: [],
    sources: [
      { connectionId: 'local', kind: 'local', label: 'This device', reachable: true },
      { connectionId: 'homelab', kind: 'remote', label: 'Homelab', ...homelab }
    ]
  }
}

const setLastUsed = vi.fn(async (id: string) => ({ ok: true, registry: { ...registry, lastUsed: id } }))

beforeEach(() => {
  localStorage.clear()
  _resetConnectionsForTests()
  _resetFleetRosterForTests()
  newChatIntent = 0
  $connection.set({ connectionId: 'local', mode: 'local', registryScoped: true })
  $activeGatewayProfile.set('default')
  ensureGatewayAgent.mockClear()
  openGatewayAgent.mockReset()
  setLastUsed.mockClear()
})

afterEach(() => vi.unstubAllGlobals())

function stubBridge(
  getAgentRoster: () => Promise<DesktopAgentRoster>,
  {
    api = vi.fn(async () => ({ profiles: [] })),
    list = registry
  }: { api?: (request: { path: string }) => Promise<unknown>; list?: DesktopConnectionsRegistry } = {}
) {
  vi.stubGlobal('window', {
    hermesDesktop: {
      api,
      connections: { list: vi.fn(async () => list), setLastUsed },
      getAgentRoster
    },
    localStorage,
    location: window.location
  })
}

describe('launch restore onto the last-used gateway', () => {
  it('a target reported down at launch never re-homes the window when it recovers', async () => {
    const dial = deferred<void>()
    openGatewayAgent.mockReturnValueOnce(dial.promise)
    stubBridge(async () => roster({ reachable: false, error: 'roster enumeration timed out' }))

    const boot = initializeConnectionsRegistry()
    await vi.waitFor(() => expect(openGatewayAgent).toHaveBeenCalledWith('homelab', 'default'))
    await vi.waitFor(() => expect($pendingConnectionId.get()).toBeNull())

    // Homelab comes back: the kept dial lands, but the window stays put.
    dial.resolve()
    await boot

    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect($connection.get()?.connectionId).toBe('local')
    expect(setLastUsed).not.toHaveBeenCalled()
  })

  it('a reachable but slow target still lands the launch on it', async () => {
    const dial = deferred<void>()
    openGatewayAgent.mockReturnValueOnce(dial.promise)
    // Connect-on-demand (an undialed SSH host) is reachable, not down.
    stubBridge(async () => roster({ reachable: true, error: 'connect-on-demand' }))

    const boot = initializeConnectionsRegistry()
    await vi.waitFor(() => expect(openGatewayAgent).toHaveBeenCalledWith('homelab', 'default'))
    dial.resolve()
    await boot

    expect($connection.get()?.connectionId).toBe('homelab')
    expect(setLastUsed).toHaveBeenCalledWith('homelab')
  })

  it('a target reported down after activation began is cancelled, not published', async () => {
    const report = deferred<DesktopAgentRoster>()
    const dial = deferred<void>()
    stubBridge(() => report.promise)
    ensureGatewayAgent.mockImplementationOnce(async (connectionId, profile, options) => {
      if (options?.beforeActivate && !options.beforeActivate()) {
        return
      }

      await dial.promise
      const { signal } = options as { signal?: AbortSignal }

      if (!signal?.aborted) {
        $connection.set({ connectionId: connectionId ?? undefined, mode: 'remote', profile, registryScoped: true })
      }
    })

    const boot = initializeConnectionsRegistry()
    await vi.waitFor(() => expect(ensureGatewayAgent).toHaveBeenCalled())
    report.resolve(roster({ reachable: false, error: 'roster enumeration timed out' }))
    await vi.waitFor(() => expect($pendingConnectionId.get()).toBeNull())
    dial.resolve()
    await boot

    expect($connection.get()?.connectionId).toBe('local')
    expect(setLastUsed).not.toHaveBeenCalled()
  })

  it('does not roll a newer profile draft back when a launch restore is abandoned', async () => {
    const profile = await import('./profile')
    const report = deferred<DesktopAgentRoster>()
    const dial = deferred<void>()
    stubBridge(() => report.promise)
    ensureGatewayAgent.mockImplementationOnce(async (_connectionId, _profile, options) => {
      options?.beforeActivate?.()
      await dial.promise
    })
    const boot = initializeConnectionsRegistry()
    await vi.waitFor(() => expect(ensureGatewayAgent).toHaveBeenCalled())
    profile.$newChatProfile.set('beta')
    profile.captureNewChatSource('local')
    report.resolve(roster({ reachable: false, error: 'offline' }))
    await vi.waitFor(() => expect($pendingConnectionId.get()).toBeNull())
    dial.resolve()
    await boot
    expect(profile.$newChatProfile.get()).toBe('beta')
    expect($connection.get()?.connectionId).toBe('local')
  })

  it('a cached down row never cancels a profile repair on the gateway already in front', async () => {
    // The window is already on homelab with a profile it does not serve; the
    // fresh profile reads just succeeded there, so an older failed roster row
    // is not evidence the window has fallen back anywhere.
    const { refreshFleetRoster } = await import('./fleet-roster')
    $connection.set({ connectionId: 'homelab', mode: 'remote', profile: 'windows-only', registryScoped: true })
    $activeGatewayProfile.set('windows-only')
    stubBridge(async () => roster({ reachable: false, error: 'roster enumeration timed out' }), {
      api: async request =>
        request.path === '/api/status' ? { gateway_shared_with: ['default'] } : { profiles: [{ name: 'default' }] },
      list: { ...registry, launchMode: 'primary', primary: 'homelab' }
    })
    await refreshFleetRoster()

    await initializeConnectionsRegistry()

    expect($connection.get()).toMatchObject({ connectionId: 'homelab', profile: 'default' })
  })
})
