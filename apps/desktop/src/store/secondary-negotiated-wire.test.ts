import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { HermesGateway } from '@/api/client'
import type { HermesConnection } from '@/global'
import { TimeoutError } from '@/lib/with-timeout'

import { serverOwnsComposerQueue } from './composer-queue'
import { $connectionWires, connectionWire, recordConnectionWire } from './connection-wire'
import { closeSecondaryGateways, configureGatewayRegistry, ensureGatewayForProfile, setPrimaryGateway, setPrimaryGatewayConnectionId } from './gateway'
import { isStalledDialError } from './gateway-dial-failures'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from './session'

// @donovan-yohan's reproducer from the review of #148219 on cc6ad86a, with a parking case added:
// the registry, HermesGateway, wire publisher, composer predicate and stall classifier are real;
// the Electron descriptor and the WebSocket peer are synthetic.
class ReviewSocket extends EventTarget {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3
  static instances: ReviewSocket[] = []
  static advertise: Record<string, unknown> | null = {}
  readyState = ReviewSocket.CONNECTING

  constructor(readonly url: string) {
    super()
    ReviewSocket.instances.push(this)
    const payload = ReviewSocket.advertise
    setTimeout(() => {
      this.readyState = ReviewSocket.OPEN
      this.dispatchEvent(new Event('open'))

      if (payload !== null) {
        this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify({
          jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload }
        }) }))
      }
    }, 1)
  }

  send(_data: string): void {}

  close(): void {
    if (this.readyState === ReviewSocket.CLOSED) { return }
    this.readyState = ReviewSocket.CLOSED
    this.dispatchEvent(new CloseEvent('close', { code: 1000 }))
  }
}

let primary: HermesGateway

const descriptor = {
  connectionId: 'override-remote', mode: 'remote', profile: 'worker', authMode: 'token',
  token: 'synthetic-only', wsUrl: 'wss://override.example/api/ws?ticket=synthetic',
  baseUrl: 'https://override.example'
} as HermesConnection

beforeEach(async () => {
  vi.useFakeTimers()
  vi.stubGlobal('WebSocket', ReviewSocket)
  ReviewSocket.instances = []
  ReviewSocket.advertise = {}
  $connectionWires.set({})
  _resetSessionOwnerHintsForTests({ storage: true })
  configureGatewayRegistry({ activeConnectionId: () => 'local', foregroundScopes: () => new Set(), onEvent: vi.fn() })
  primary = new HermesGateway()
  const opening = primary.connect('wss://local.example/api/ws?ticket=synthetic')
  await vi.advanceTimersByTimeAsync(2)
  await opening
  setPrimaryGateway(primary, 'default')
  setPrimaryGatewayConnectionId('local', 'local')
  window.hermesDesktop = {
    getConnection: vi.fn(async () => descriptor),
    getGatewayWsUrl: vi.fn(async () => descriptor.wsUrl)
  } as never
})

afterEach(() => {
  closeSecondaryGateways()
  primary.close()
  setPrimaryGateway(null)
  setPrimaryGatewayConnectionId(null)
  $connection.set(null)
  $connectionWires.set({})
  _resetSessionOwnerHintsForTests({ storage: true })
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

test('a profile-only canonical override records its resolved backend and enables its composer', async () => {
  ReviewSocket.advertise = { session_authority: true }
  const opening = ensureGatewayForProfile('worker')
  await vi.advanceTimersByTimeAsync(10)
  await opening
  setSessionOwnerHint('override-chat', { connectionId: 'override-remote', mode: 'remote', profile: 'worker' })
  expect($connection.get()?.connectionId).toBe('override-remote')
  expect.soft(connectionWire('override-remote')).toBe('canonical')
  expect.soft(serverOwnsComposerQueue('override-chat')).toBe(true)
})

test('a profile-only legacy override does not replace the local canonical wire', async () => {
  recordConnectionWire('local', 'canonical')
  setSessionOwnerHint('local-chat', { connectionId: 'local', mode: 'local', profile: 'default' })
  expect(serverOwnsComposerQueue('local-chat')).toBe(true)
  ReviewSocket.advertise = {}
  const opening = ensureGatewayForProfile('worker')
  await vi.advanceTimersByTimeAsync(10)
  await opening
  expect.soft(connectionWire('local')).toBe('canonical')
  expect.soft(connectionWire('override-remote')).toBe('legacy')
  expect.soft(serverOwnsComposerQueue('local-chat')).toBe(true)
})

test('control: an explicit remote source records its own wire without changing local', async () => {
  const { ensureGatewayForAgent } = await import('./gateway')
  recordConnectionWire('local', 'canonical')
  ReviewSocket.advertise = {}
  window.hermesDesktop = {
    getConnectionFor: vi.fn(async () => descriptor),
    getGatewayWsUrlFor: vi.fn(async () => descriptor.wsUrl)
  } as never
  const opening = ensureGatewayForAgent('override-remote', 'worker')
  await vi.advanceTimersByTimeAsync(10)
  await opening
  expect(connectionWire('override-remote')).toBe('legacy')
  expect(connectionWire('local')).toBe('canonical')
  expect(isStalledDialError(new TimeoutError('fixture timeout'))).toBe(true)
  expect(isStalledDialError(new Error('timed out while waiting for a free slot'))).toBe(true)
})

test('a real protocol readiness timeout is classified as a stalled dial', async () => {
  ReviewSocket.advertise = null
  const client = new HermesGateway()
  const connected = client.connect('wss://silent.example/api/ws?ticket=synthetic')
  const outcome = connected.then(() => null, error => error)

  try {
    await vi.advanceTimersByTimeAsync(15_001)
    const error = await outcome
    expect(error).toBeInstanceOf(Error)
    expect(error.message).toBe('Hermes gateway protocol readiness timed out')
    expect(client.connectionState).toBe('closed')
    expect(isStalledDialError(error)).toBe(true)
  } finally {
    client.close()
  }
})

test('a secondary whose redials open but never become ready parks after its stalled-dial budget', async () => {
  const opening = ensureGatewayForProfile('worker')
  await vi.advanceTimersByTimeAsync(10)
  await opening
  const before = ReviewSocket.instances.length

  // The peer drops and every redial opens but never sends gateway.ready.
  ReviewSocket.advertise = null
  ReviewSocket.instances.at(-1)!.close()

  for (let index = 0; index < 20; index += 1) {
    await vi.advanceTimersByTimeAsync(60_000)
  }

  const redials = ReviewSocket.instances.length - before
  expect(redials).toBeGreaterThan(0)
  expect(redials).toBeLessThanOrEqual(3)

  await vi.advanceTimersByTimeAsync(600_000)
  expect(ReviewSocket.instances.length - before).toBe(redials)
})
