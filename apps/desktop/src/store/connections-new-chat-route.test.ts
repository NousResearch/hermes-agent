import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry, HermesConnection } from '@/global'
import type * as Hermes from '@/hermes'

import { deferred } from '../test/deferred'

vi.mock('@/app/contrib/hooks/use-background-sync', () => ({ resetLiveRuntimeTracking: vi.fn() }))
vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<typeof Hermes>()

  return {
    ...actual,
    getProfiles: vi.fn(async () => ({ profiles: [{ name: 'default' }] })),
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

import { $activeConnectionId, _resetConnectionsForTests, selectConnection, setConnectionsRegistry } from './connections'
import {
  activeGatewayConnectionId,
  closeSecondaryGateways,
  configureGatewayRegistry,
  setPrimaryGateway,
  setPrimaryGatewayConnection
} from './gateway'
import { $gatewaySwitching } from './gateway-switch'
import {
  $activeGatewayProfile,
  $newChatRoute,
  $showAllProfiles,
  ensureGatewayAgent,
  newSessionInAgent,
  resolveNewChatOwnerRoute
} from './profile'
import { setConnection } from './session'

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

it('re-homes an explicit draft with the selected source, including the return trip', async () => {
  for (const connectionId of ['local', 'homelab']) {
    await selectConnection(connectionId)
    expect($activeConnectionId.get()).toBe(connectionId)
    expect(activeGatewayConnectionId()).toBe(connectionId)
    expect(resolveNewChatOwnerRoute()).toEqual({ connectionId, profile: 'default' })
  }
})

it('re-homes a fresh draft when leaving All profiles on the already active source', async () => {
  $newChatRoute.set({ connectionId: 'local', profile: 'default' })
  $showAllProfiles.set(true)
  await selectConnection('homelab')
  expect(resolveNewChatOwnerRoute()).toEqual({ connectionId: 'homelab', profile: 'default' })
})

it('keeps a draft the user pinned while the silent boot restore was in flight', async () => {
  // No active source yet: selectConnection runs as the boot-time restore.
  setConnection(null)
  expect($activeConnectionId.get()).toBeNull()
  const route = $newChatRoute.get()
  await selectConnection('local')
  expect($newChatRoute.get()).toEqual(route)
  expect(resolveNewChatOwnerRoute()).toEqual(route)
})

it('preserves the explicit draft when the target dial fails', async () => {
  const route = $newChatRoute.get()
  getConnectionFor.mockRejectedValueOnce(new Error('offline'))
  await expect(selectConnection('local')).rejects.toThrow('offline')
  expect($newChatRoute.get()).toEqual(route)
  expect(resolveNewChatOwnerRoute()).toEqual(route)
})

it('re-homes the draft before remembering the source, so a Send during that IPC names the shown profile', async () => {
  await selectConnection('homelab', { profile: 'gaming' })
  expect(resolveNewChatOwnerRoute()).toEqual({ connectionId: 'homelab', profile: 'gaming' })
  const remembered = deferred<{ ok: boolean; registry: DesktopConnectionsRegistry }>()
  Object.assign(window.hermesDesktop!, {
    connections: { setLastUsed: vi.fn().mockReturnValueOnce(remembered.promise) }
  })
  const switching = selectConnection('homelab', { profile: 'default' })
  await vi.waitFor(() => expect(window.hermesDesktop!.connections!.setLastUsed).toHaveBeenCalled())
  // The UI already shows default with the switch barrier down: a Send now
  // resolves its owner from the draft, which must not still name gaming.
  const foreground = { connectionId: activeGatewayConnectionId(), profile: $activeGatewayProfile.get() }
  const switchingUi = $gatewaySwitching.get()
  const owner = resolveNewChatOwnerRoute()
  remembered.resolve({ ok: true, registry })
  await switching
  expect(foreground).toEqual({ connectionId: 'homelab', profile: 'default' })
  expect(switchingUi).toBe(false)
  expect(owner).toEqual({ connectionId: 'homelab', profile: 'default' })
})

it('does not overwrite a newer draft when remembering a committed switch settles late', async () => {
  const remembered = deferred<{ ok: boolean; registry: DesktopConnectionsRegistry }>()
  Object.assign(window.hermesDesktop!, {
    connections: {
      setLastUsed: vi.fn().mockReturnValueOnce(remembered.promise).mockResolvedValue({ ok: true, registry })
    }
  })
  const first = selectConnection('local')
  await vi.waitFor(() => expect(window.hermesDesktop!.connections!.setLastUsed).toHaveBeenCalled())
  await selectConnection('homelab')
  const route = { connectionId: 'homelab', profile: 'default' }
  $newChatRoute.set(route)
  remembered.resolve({ ok: true, registry })
  await first
  expect($newChatRoute.get()).toEqual(route)
  expect(resolveNewChatOwnerRoute()).toEqual(route)
})

it('keeps a newer agent draft started while an ordinary switch is still bookkeeping', async () => {
  const withOther: DesktopConnectionsRegistry = {
    ...registry,
    connections: [
      ...registry.connections,
      { id: 'other', kind: 'remote', label: 'Other', tokenSet: true, tokenPreview: null }
    ]
  }

  setConnectionsRegistry(withOther)
  const remembered = deferred<{ ok: boolean; registry: DesktopConnectionsRegistry }>()
  Object.assign(window.hermesDesktop!, {
    connections: {
      setLastUsed: vi.fn().mockReturnValueOnce(remembered.promise).mockResolvedValue({ ok: true, registry: withOther })
    }
  })
  const first = selectConnection('local')
  await vi.waitFor(() => expect(window.hermesDesktop!.connections!.setLastUsed).toHaveBeenCalled())
  // The newer draft's source is still dialing, so local stays in the foreground.
  const dial = deferred<HermesConnection>()
  getConnectionFor.mockImplementation(async ({ connectionId, profile }) =>
    connectionId === 'other' ? dial.promise : descriptor(connectionId, profile)
  )
  newSessionInAgent({ connectionId: 'other', profile: 'default' })
  remembered.resolve({ ok: true, registry: withOther })
  await first
  const foreground = activeGatewayConnectionId()
  const owner = resolveNewChatOwnerRoute()
  // Settle the held dial before asserting so a failure can't wedge the mutex.
  dial.resolve(descriptor('other'))
  await ensureGatewayAgent('other', 'default')
  expect(foreground).toBe('local')
  expect(owner).toEqual({ connectionId: 'other', profile: 'default' })
  expect(resolveNewChatOwnerRoute()).toEqual({ connectionId: 'other', profile: 'default' })
})

it('does not let a superseded dial clear a newer explicit draft', async () => {
  const dial = deferred<HermesConnection>()
  getConnectionFor.mockImplementationOnce(() => dial.promise)
  const first = selectConnection('local')
  await vi.waitFor(() => expect(getConnectionFor).toHaveBeenCalled())
  await selectConnection('homelab')
  const route = { connectionId: 'homelab', profile: 'default' }
  $newChatRoute.set(route)
  dial.resolve(descriptor('local'))
  await first
  expect($newChatRoute.get()).toEqual(route)
  expect(resolveNewChatOwnerRoute()).toEqual(route)
})
