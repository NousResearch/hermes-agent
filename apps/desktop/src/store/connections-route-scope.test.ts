import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry, HermesConnection } from '@/global'
import type * as Hermes from '@/hermes'

vi.mock('@/app/contrib/hooks/use-background-sync', () => ({ resetLiveRuntimeTracking: vi.fn() }))
vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<typeof Hermes>()

  return {
    ...actual,
    getProfiles: vi.fn(async () => ({ profiles: [{ name: 'default' }] })),
    getStarmapGraph: vi.fn(async () => ({ nodes: [], edges: [] })),
    hermesApi: vi.fn(async () => ({ current: 'default', profiles: [{ name: 'default' }] })),
    HermesGateway: class {
      connectionState = 'closed'
      wsUrl = ''
      connect = async (url: string) => {
        this.wsUrl = url
        this.connectionState = 'open'
      }
      close = () => {
        this.connectionState = 'closed'
      }
      onEvent = () => () => undefined
      onState = () => () => undefined
      request = async () => ({})
    }
  }
})

import {
  _resetConnectionsForTests,
  selectConnection,
  setConnectionsRegistry
} from '@/store/connections'
import {
  closeSecondaryGateways,
  configureGatewayRegistry,
  setPrimaryGateway,
  setPrimaryGatewayConnection
} from '@/store/gateway'
import {
  $activeGatewayProfile,
  $newChatRoute,
  $showAllProfiles,
  ensureGatewayAgent,
  newSessionInAgent
} from '@/store/profile'
import { $projectScope, ALL_PROJECTS } from '@/store/project-scope'
import { $projectTree, resolveNewSessionCwd } from '@/store/projects'
import { setConnection } from '@/store/session'
import { $starmapGraph, loadStarmapGraph } from '@/store/starmap'

const registry: DesktopConnectionsRegistry = {
  version: 2,
  primary: 'homelab',
  secureTokenStorage: true,
  connections: [
    { id: 'homelab', kind: 'remote', label: 'Home', tokenSet: true, tokenPreview: null },
    { id: 'local', kind: 'local', label: 'This device', tokenSet: false, tokenPreview: null }
  ]
}

function descriptor(connectionId: string, profile = 'default'): HermesConnection {
  return {
    connectionId,
    profile,
    registryScoped: true,
    isFullscreen: false,
    nativeOverlayWidth: 0,
    logs: [],
    windowButtonPosition: null,
    mode: connectionId === 'local' ? 'local' : 'remote',
    authMode: 'token',
    token: 'test',
    wsUrl: `ws://${connectionId}.invalid/ws`,
    baseUrl: `http://${connectionId}.invalid`
  }
}

const getConnectionFor = vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
  descriptor(connectionId, profile)
)

beforeEach(async () => {
  _resetConnectionsForTests()
  getConnectionFor.mockReset()
  getConnectionFor.mockImplementation(async ({ connectionId, profile }) => descriptor(connectionId, profile))
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      getConnectionFor,
      getGatewayWsUrlFor: vi.fn(async ({ connectionId }) => ({ ok: true, wsUrl: `ws://${connectionId}.invalid/ws` })),
      api: vi.fn(async () => ({ current: 'default', profiles: [{ name: 'default' }] }))
    }
  })
  setConnectionsRegistry(registry)
  setPrimaryGateway({ connectionState: 'open', request: vi.fn(async () => ({})) } as never, 'default')
  setPrimaryGatewayConnection({ connectionId: 'homelab', mode: 'remote' })
  setConnection(descriptor('homelab'))
  configureGatewayRegistry({ onEvent: vi.fn(), onActiveRouteChanged: profile => $activeGatewayProfile.set(profile) })
  $showAllProfiles.set(false)
  newSessionInAgent({ connectionId: 'homelab', profile: 'default' })
  // Drain the real activation mutex used by the fire-and-forget agent action.
  await ensureGatewayAgent('homelab', 'default')
})

afterEach(() => {
  closeSecondaryGateways()
  $newChatRoute.set(null)
})

it("picking another profile on the same gateway leaves the old profile's project before drafting", async () => {
  $projectScope.set('old-profile-project')
  $projectTree.set([{ id: 'old-profile-project', path: '/profile-a/repo', name: 'old', repos: [] }] as never)

  await selectConnection('homelab', { profile: 'research' })

  expect($projectScope.get()).toBe(ALL_PROJECTS)
  expect(resolveNewSessionCwd()).not.toBe('/profile-a/repo')
})

it('picking another gateway with the same profile name rescans the star map', async () => {
  const oldGraph = { nodes: [{ id: 'skill:old-host-only' }], edges: [] }
  $starmapGraph.set(oldGraph as never)

  await selectConnection('local')
  await loadStarmapGraph() // the lazy read a mounting Star Map view makes

  expect($starmapGraph.get()).not.toBe(oldGraph)
})
