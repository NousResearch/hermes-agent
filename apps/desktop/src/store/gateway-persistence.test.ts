import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'

const released: string[] = []
const retainGatewayForRelay = vi.fn((connectionId: string) => () => released.push(connectionId))
const openGatewayForAgent = vi.fn(async (_connectionId: string, _profile: string) => undefined)
const live = new Set<string>()

vi.mock('@/store/gateway', () => ({
  liveSecondaryConnectionIds: () => new Set(live),
  openGatewayForAgent,
  retainGatewayForRelay
}))

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
  retainGatewayForRelay.mockClear()
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

  expect(retainGatewayForRelay.mock.calls.map(call => call[0])).toEqual(['homelab', 'vps'])
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
  expect(retainGatewayForRelay).toHaveBeenCalledTimes(2)
})

it('pins This device on its own socket when a remote gateway is the primary', () => {
  $primaryGatewayRoute.set({ connectionId: 'homelab', mode: 'remote' })
  stop = startPersistentGateways()

  expect(retainGatewayForRelay.mock.calls).toEqual([
    ['local', 'default', { includeLocal: true }],
    ['vps', 'default', { includeLocal: true }]
  ])
})
