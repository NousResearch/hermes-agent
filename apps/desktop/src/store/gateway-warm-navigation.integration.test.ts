import { once } from 'node:events'

import { registryBackendScopeKey } from '@hermes/shared'
import type { Mock } from 'vitest'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import WebSocket, { WebSocketServer } from 'ws'

import type { HermesConnection } from '@/global'

const published = vi.hoisted(() => ({ connection: vi.fn(), event: vi.fn() }))
vi.mock('@/hermes', async () => ({
  ...(await import('../api/client')),
  getProfiles: vi.fn(async () => ({ profiles: [] }))
}))
vi.mock('@/store/session', () => ({
  setConnection: published.connection,
  setGatewayState: vi.fn(),
  clearComposerSelectionOwner: vi.fn(),
  setComposerSelectionOwner: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const gateway = await import('./gateway')
const profile = await import('./profile')
let server: WebSocketServer
let connections: WebSocket[]
let descriptor: Mock<(target: { connectionId: string; profile: string }) => Promise<Partial<HermesConnection>>>

beforeEach(async () => {
  vi.stubGlobal('WebSocket', WebSocket)
  connections = []
  server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await once(server, 'listening')
  const address = server.address()

  if (!address || typeof address === 'string') {
    throw new Error('Expected TCP server')
  }
  server.on('connection', socket => {
    connections.push(socket)
    socket.on('message', raw => {
      const request = JSON.parse(raw.toString())
      socket.send(JSON.stringify({ jsonrpc: '2.0', id: request.id, result: { method: request.method } }))
    })
  })
  descriptor = vi.fn(async ({ connectionId, profile: name }: { connectionId: string; profile: string }) => ({
    connectionId,
    profile: name,
    mode: 'remote' as const,
    remoteKind: 'url' as const,
    authMode: 'token' as const,
    baseUrl: `http://127.0.0.1:${address.port}`,
    wsUrl: `ws://127.0.0.1:${address.port}/api/ws`,
    registryScoped: true,
    sharedRemote: false
  }))
  Object.assign(window, { hermesDesktop: { getConnectionFor: descriptor, touchBackend: vi.fn(async () => undefined) } })
  gateway.configureGatewayRegistry({ onEvent: published.event, foregroundScopes: () => new Set() })
  gateway.setPrimaryGateway({ connectionState: 'open' } as never, 'default')
  gateway.setPrimaryGatewayConnectionId('primary')
  published.connection.mockClear()
  published.event.mockClear()
})

afterEach(async () => {
  gateway.closeSecondaryGateways()

  for (const socket of server.clients) {
    socket.terminate()
  }
  await new Promise<void>(resolve => server.close(() => resolve()))
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  vi.unstubAllGlobals()
})

it('switches a warm remote without a descriptor probe, preserving background events and exact routing', async () => {
  await gateway.openGatewayForAgent('personal', 'assistant', { spawnPriority: 'foreground' })
  await gateway.openGatewayForAgent('work', 'assistant', { spawnPriority: 'foreground' })
  const before = descriptor.mock.calls.length
  let releaseProbe: () => void = () => undefined
  const probe = new Promise<void>(resolve => {
    releaseProbe = resolve
  })
  const original = descriptor.getMockImplementation()!
  descriptor.mockImplementation(async payload => {
    await probe

    return original(payload)
  })
  let switched = false
  const switching = profile.ensureGatewayAgent('work', 'assistant').then(() => {
    switched = true
  })

  try {
    await vi.waitFor(() => expect(switched).toBe(true), { timeout: 2_000 })
    expect(descriptor.mock.calls.length).toBe(before)
    expect(published.connection).toHaveBeenLastCalledWith(
      expect.objectContaining({ connectionId: 'work', profile: 'assistant' })
    )
    await expect(gateway.requestGatewayForAgent('personal', 'assistant', 'session.list')).resolves.toEqual({
      method: 'session.list'
    })
    connections[0].send(
      JSON.stringify({
        jsonrpc: '2.0',
        method: 'event',
        params: { type: 'tool.progress', session_id: 'background', payload: { text: 'still working' } }
      })
    )
    await vi.waitFor(() =>
      expect(published.event).toHaveBeenCalledWith(
        expect.objectContaining({ connectionId: 'personal', session_id: 'background' })
      )
    )
    expect(connections).toHaveLength(2)
  } finally {
    releaseProbe()
    await switching
  }
})

it('keeps deliberately opened remotes warm after navigation, but releases hover-only routes and removed connections', async () => {
  await gateway.openGatewayForAgent('personal', 'assistant', { spawnPriority: 'foreground' })
  await gateway.openGatewayForAgent('work', 'assistant', { spawnPriority: 'foreground' })
  await gateway.openGatewayForAgent('preview', 'assistant')
  gateway.pruneSecondaryGateways(new Set())
  expect(gateway.liveSecondaryConnectionIds()).toEqual(new Set(['personal', 'work']))
  await gateway.openGatewayForAgent('personal', 'assistant', { spawnPriority: 'foreground' })
  await gateway.openGatewayForAgent('work', 'assistant', { spawnPriority: 'foreground' })
  expect(connections).toHaveLength(3)
  gateway.disposeSecondariesForConnection('work')
  expect(gateway.liveSecondaryConnectionIds()).toEqual(new Set(['personal']))
})

it('invalidates cached routing on material edits and removal without interrupting a mounted session', async () => {
  await gateway.openGatewayForAgent('work', 'assistant', { spawnPriority: 'foreground' })
  expect(gateway.openGatewayConnection('work', 'assistant')?.connectionId).toBe('work')
  expect(gateway.openGatewayConnection('personal', 'assistant')).toBeNull()
  expect(gateway.openGatewayConnection('work', 'other-profile')).toBeNull()
  gateway.configureGatewayRegistry({
    onEvent: published.event,
    foregroundScopes: () => new Set([registryBackendScopeKey('work', 'assistant')])
  })
  gateway.disposeSecondariesForConnection('work', { redial: true })
  expect(gateway.openGatewayConnection('work', 'assistant')).toBeNull()
  expect(gateway.liveSecondaryConnectionIds()).toEqual(new Set(['work']))
  expect(connections[0].readyState).toBe(WebSocket.OPEN)
  gateway.configureGatewayRegistry({ onEvent: published.event, foregroundScopes: () => new Set() })
  gateway.disposeSecondariesForConnection('work')
  expect(gateway.openGatewayConnection('work', 'assistant')).toBeNull()
  expect(gateway.liveSecondaryConnectionIds()).toEqual(new Set())
})

it('does not pin app-managed SSH processes after leaving their foreground surface', async () => {
  const original = descriptor.getMockImplementation()!
  descriptor.mockImplementation(async target => ({ ...(await original(target)), remoteKind: 'ssh' }))
  await gateway.openGatewayForAgent('ssh-host', 'assistant', { spawnPriority: 'foreground' })
  expect(gateway.liveSecondaryConnectionIds()).toEqual(new Set(['ssh-host']))
  gateway.pruneSecondaryGateways(new Set())
  expect(gateway.liveSecondaryConnectionIds()).toEqual(new Set())
})
