import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry, DesktopProfileRoute, HermesConnection } from '@/global'
import { deferred } from '@/test/deferred'

import { _resetConnectionsForTests, initializeConnectionsRegistry } from './connections'
import { $defaultProfileRoute } from './default-profile'
import { ensureGatewayForAgent, openGatewayForAgent } from './gateway'
import { $activeGatewayProfile, pinNewChatProfile, requestFreshSession } from './profile'
import { $connection, setActiveSessionId, setConnection } from './session'

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayForAgent: vi.fn(async () => ({})),
  openGatewayForAgent: vi.fn(async () => ({}))
}))

const registry: DesktopConnectionsRegistry = {
  connections: [
    { id: 'local', kind: 'local', label: 'This device', tokenPreview: null, tokenSet: false },
    { id: 'lab', kind: 'remote', label: 'Lab', tokenPreview: null, tokenSet: false }
  ],
  primary: 'lab',
  lastUsed: 'lab',
  launchMode: 'last-used',
  secureTokenStorage: true,
  version: 2
}

function descriptor(connectionId: string, profile: string): HermesConnection {
  return {
    baseUrl: 'http://localhost:7070',
    connectionId,
    isFullscreen: false,
    logs: [],
    mode: connectionId === 'local' ? 'local' : 'remote',
    nativeOverlayWidth: 0,
    profile,
    registryScoped: true,
    token: '',
    windowButtonPosition: null,
    wsUrl: 'ws://localhost:7070'
  }
}

beforeEach(() => {
  _resetConnectionsForTests()
  $defaultProfileRoute.set(null)
  window.history.replaceState(null, '', '/')
  $activeGatewayProfile.set('personal')
  setConnection(descriptor('local', 'personal'))
  window.hermesDesktop = {
    connections: {
      list: async () => registry,
      setLastUsed: async (id: string) => ({ ok: true, registry: { ...registry, lastUsed: id } })
    },
    getConnectionFor: async ({ connectionId, profile }: DesktopProfileRoute) =>
      descriptor(connectionId ?? 'local', profile),
    profile: { getDefault: async () => null },
    api: async ({ path }: { path: string }) =>
      path.includes('active')
        ? { active: $activeGatewayProfile.get(), current: $activeGatewayProfile.get() }
        : { profiles: [] }
  } as never
})

afterEach(() => {
  window.history.replaceState(null, '', '/')
  setConnection(null)
  setActiveSessionId(null)
  $defaultProfileRoute.set(null)
  vi.clearAllMocks()
})

describe('startup default route', () => {
  it('does not replace the main-process route after a transient preference read failure', async () => {
    window.hermesDesktop.profile.getDefault = async () => {
      throw new Error('IPC interrupted')
    }
    await initializeConnectionsRegistry()
    expect($connection.get()).toMatchObject({ connectionId: 'local', profile: 'personal' })
    expect(openGatewayForAgent).not.toHaveBeenCalled()
  })

  it('does not override a profile picked while the preference read was pending', async () => {
    const reading = deferred<DesktopProfileRoute>()
    window.hermesDesktop.profile.getDefault = () => reading.promise
    const initializing = initializeConnectionsRegistry()
    pinNewChatProfile('chosen')
    requestFreshSession()
    reading.resolve({ connectionId: 'lab', profile: 'research' })
    await initializing
    expect(openGatewayForAgent).not.toHaveBeenCalled()
    expect($connection.get()).toMatchObject({ connectionId: 'local', profile: 'personal' })
  })

  it('does not re-home an already selected session when the preference loads', async () => {
    setActiveSessionId('session-already-open')
    window.hermesDesktop.profile.getDefault = async () => ({ connectionId: 'lab', profile: 'research' })
    await initializeConnectionsRegistry()
    expect(openGatewayForAgent).not.toHaveBeenCalled()
  })

  it('keeps the native legacy default and its per-profile remote override out of registry restoration', async () => {
    setConnection({ ...descriptor('lab', 'personal'), registryScoped: false })
    window.hermesDesktop.profile.getDefault = async () => ({ connectionId: null, profile: 'personal' })
    await initializeConnectionsRegistry()
    expect($connection.get()).toMatchObject({ connectionId: 'lab', profile: 'personal', registryScoped: false })
    expect(openGatewayForAgent).not.toHaveBeenCalled()
    expect(ensureGatewayForAgent).not.toHaveBeenCalled()
  })

  it('restores the saved profile, not last-used source/profile, across local → remote → local', async () => {
    for (const route of [
      { connectionId: 'local', profile: 'personal' },
      { connectionId: 'lab', profile: 'research' },
      { connectionId: 'local', profile: 'personal' }
    ]) {
      _resetConnectionsForTests()
      window.hermesDesktop.profile.getDefault = async () => route
      await initializeConnectionsRegistry()
      expect($connection.get()).toMatchObject({ connectionId: route.connectionId ?? 'local', profile: route.profile })
      expect($activeGatewayProfile.get()).toBe(route.profile)
    }
    expect(ensureGatewayForAgent).toHaveBeenCalledWith('lab', 'research', expect.anything())
    expect(ensureGatewayForAgent).toHaveBeenCalledWith('local', 'personal', expect.anything())
  })

  it('leaves profile peer windows on their requested source rather than restoring the application default', async () => {
    window.history.replaceState(null, '', '/?peer=1&profile=personal&connectionId=local')
    window.hermesDesktop.profile.getDefault = async () => ({ connectionId: 'lab', profile: 'research' })
    await initializeConnectionsRegistry()
    expect($connection.get()).toMatchObject({ connectionId: 'local', profile: 'personal' })
    expect(openGatewayForAgent).not.toHaveBeenCalled()
  })
})
