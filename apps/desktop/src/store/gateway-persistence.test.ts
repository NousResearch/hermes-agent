import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'

const released: string[] = []
const keepRoute = vi.fn((connectionId: string, _profile: string) => () => released.push(connectionId))
const openGatewayForAgent = vi.fn(async (_connectionId: string, _profile: string) => undefined)
const live = new Set<string>()
// `live`: the gateway's kept default-profile route is open. This set: only a
// named profile's socket on that gateway is open.
const liveNamedProfile = new Set<string>()

vi.mock('@/store/gateway', () => ({
  gatewayRouteState: (connectionId: string, profile: string) => ({
    key: `${connectionId}:${profile}`,
    open: profile === 'default' && live.has(connectionId)
  }),
  // Not read by the keeper; the pre-fix keeper did, so the redial test fails there on its assertion.
  liveSecondaryConnectionIds: () => new Set([...live, ...liveNamedProfile]),
  openGatewayForAgent
}))
vi.mock('@/store/gateway-kept-routes', () => ({ keepRoute }))

const { $connectionsRegistry } = await import('@/store/connection-registry-state')
const { $primaryGatewayRoute } = await import('@/store/gateway-primary-route')
const { startPersistentGateways } = await import('./gateway-persistence')

const registry = (ids: string[]): DesktopConnectionsRegistry => ({
  connections: ids.map(id => ({
    id,
    kind: id === 'local' ? 'local' : 'remote',
    label: id,
    tokenPreview: null,
    tokenSet: false
  })),
  primary: 'local',
  secureTokenStorage: true,
  version: 2
})

let stop: () => void = () => undefined

beforeEach(() => {
  vi.useFakeTimers()
  released.length = 0
  live.clear()
  liveNamedProfile.clear()
  keepRoute.mockClear()
  openGatewayForAgent.mockClear()
  $primaryGatewayRoute.set({ connectionId: 'local', mode: 'local' })
  $connectionsRegistry.set(registry(['local', 'homelab', 'vps']))
})

afterEach(() => {
  stop()
  vi.useRealTimers()
})

it('keeps every registered gateway the primary socket does not serve connected, and lets a removed one go', () => {
  stop = startPersistentGateways()

  expect(keepRoute.mock.calls).toEqual([
    ['homelab', 'default'],
    ['vps', 'default']
  ])
  expect(openGatewayForAgent.mock.calls).toEqual([
    ['homelab', 'default'],
    ['vps', 'default']
  ])

  // A dropped socket is redialed on the next tick; a live one is left alone.
  live.add('homelab')
  openGatewayForAgent.mockClear()
  vi.advanceTimersByTime(30_000)
  expect(openGatewayForAgent.mock.calls).toEqual([['vps', 'default']])

  $connectionsRegistry.set(registry(['local', 'homelab']))
  expect(released).toEqual(['vps'])
  expect(keepRoute).toHaveBeenCalledTimes(2)
})

it('with a remote primary, never starts This device in the background even if Hermes is installed', async () => {
  $primaryGatewayRoute.set({ connectionId: 'homelab', mode: 'remote' })
  stop = startPersistentGateways()

  expect(openGatewayForAgent.mock.calls).toEqual([['vps', 'default']])

  stop()
  openGatewayForAgent.mockClear()
  stop = startPersistentGateways()

  await vi.advanceTimersByTimeAsync(60_000)
  expect(openGatewayForAgent.mock.calls).toEqual([
    ['vps', 'default'],
    ['vps', 'default'],
    ['vps', 'default']
  ])
  expect(keepRoute.mock.calls).not.toContainEqual(['local', 'default'])
})

it('keeps remote peers when the user explicitly switches the primary to This device', () => {
  $primaryGatewayRoute.set({ connectionId: 'homelab', mode: 'remote' })
  stop = startPersistentGateways()
  expect(dialedRoutes()).toEqual([['vps', 'default']])

  openGatewayForAgent.mockClear()
  $primaryGatewayRoute.set({ connectionId: 'local', mode: 'local' })

  expect(dialedRoutes()).toEqual([
    ['homelab', 'default'],
    ['vps', 'default']
  ])
  expect(keepRoute.mock.calls).not.toContainEqual(['local', 'default'])
})

it('redials a dropped kept route even while a named profile on the same gateway stays open', () => {
  live.add('homelab')
  live.add('vps')
  stop = startPersistentGateways()
  expect(openGatewayForAgent).not.toHaveBeenCalled()

  live.delete('homelab')
  liveNamedProfile.add('homelab')
  vi.advanceTimersByTime(30_000)

  expect(dialedRoutes()).toEqual([['homelab', 'default']])
})

/** (connection, profile) per dial, whatever options the keeper passes. */
function dialedRoutes(): unknown[][] {
  return openGatewayForAgent.mock.calls.map(call => call.slice(0, 2))
}

it("dials nothing while no primary route is published, so the primary's own gateway never gets a second socket", () => {
  $primaryGatewayRoute.set({ connectionId: null, mode: null })
  stop = startPersistentGateways()
  vi.advanceTimersByTime(30_000)
  expect(openGatewayForAgent).not.toHaveBeenCalled()

  $primaryGatewayRoute.set({ connectionId: 'homelab', mode: 'remote' })
  expect(dialedRoutes()).toEqual([['vps', 'default']])

  // A primary teardown clears the route until the next primary publishes.
  openGatewayForAgent.mockClear()
  $primaryGatewayRoute.set({ connectionId: null, mode: null })
  vi.advanceTimersByTime(30_000)
  expect(openGatewayForAgent).not.toHaveBeenCalled()
})
