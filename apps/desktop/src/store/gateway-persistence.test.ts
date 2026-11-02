import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'

const released: string[] = []
const keepRoute = vi.fn((connectionId: string, _profile: string) => () => released.push(connectionId))
const openGatewayForAgent = vi.fn(async (_connectionId: string, _profile: string) => undefined)
const live = new Set<string>()

vi.mock('@/store/gateway', () => ({
  liveSecondaryConnectionIds: () => new Set(live),
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

function installProbe(bootstrapNeeded: boolean): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    probeLocalBackend: vi.fn(async () => ({ bootstrapNeeded }))
  }
}

beforeEach(() => {
  vi.useFakeTimers()
  released.length = 0
  live.clear()
  keepRoute.mockClear()
  openGatewayForAgent.mockClear()
  $primaryGatewayRoute.set({ connectionId: 'local', mode: 'local' })
  $connectionsRegistry.set(registry(['local', 'homelab', 'vps']))
})

afterEach(() => {
  stop()
  vi.useRealTimers()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
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

it('with a remote primary, keeps This device only where Hermes is installed (never starts the installer)', async () => {
  $primaryGatewayRoute.set({ connectionId: 'homelab', mode: 'remote' })
  installProbe(true)
  stop = startPersistentGateways()
  await vi.advanceTimersByTimeAsync(0)

  expect(openGatewayForAgent.mock.calls).toEqual([['vps', 'default']])

  stop()
  openGatewayForAgent.mockClear()
  installProbe(false)
  stop = startPersistentGateways()
  await vi.advanceTimersByTimeAsync(0)

  expect(openGatewayForAgent.mock.calls).toEqual([
    ['vps', 'default'],
    ['local', 'default']
  ])
})
