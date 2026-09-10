/**
 * Tests for electron/gateway-rpc-probe.ts.
 *
 * Run with: vitest run --project electron electron/gateway-rpc-probe.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * The probe drives one JSON-RPC round-trip over a real WebSocket so boot can
 * tell "the gateway answers requests" from "the upgrade was accepted but the
 * dispatcher is dead" (#92927: a half-updated backend boots HTTP/WS-reachable
 * yet renders an empty shell). Here we inject a fake socket to replay each
 * outcome deterministically.
 */

import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  probeGatewayRpc,
  RPC_PROBE_FAILED_ERROR_CODE,
  RPC_PROBE_UNAVAILABLE_ERROR_CODE,
  rpcProbeBootError
} from './gateway-rpc-probe'

// Minimal WebSocket double: records listeners synchronously and exposes emit()
// plus a sent-frames log so tests can replay the server's side.
function makeFakeWs(): { FakeWs: new (url: string) => any; instances: any[] } {
  const instances: any[] = []

  class FakeWs {
    url: string
    closed = false
    sent: string[] = []
    listeners: Record<string, any[]> = {}
    constructor(url: string) {
      this.url = url
      this.listeners = {}
      this.closed = false
      instances.push(this)
    }
    addEventListener(type: string, fn: any) {
      ;(this.listeners[type] ||= []).push(fn)
    }
    send(data: string) {
      this.sent.push(data)
    }
    close() {
      this.closed = true
    }
    emit(type: string, event?: any) {
      for (const fn of this.listeners[type] || []) {
        fn(event)
      }
    }
  }

  return { FakeWs, instances }
}

const FAST = { connectTimeoutMs: 1_000, replyTimeoutMs: 1_000 }

function replyWithId(id: string, extra: Record<string, unknown> = {}) {
  return { data: JSON.stringify({ jsonrpc: '2.0', id, ...extra }) }
}

test('probe sends one JSON-RPC request on open and resolves on a result with the matching id', async () => {
  const { FakeWs, instances } = makeFakeWs()

  const promise = probeGatewayRpc('ws://host/api/ws?token=t', {
    WebSocketImpl: FakeWs,
    method: 'session.list',
    requestId: 'probe-1'
  })

  instances[0].emit('open')
  assert.equal(instances[0].sent.length, 1)

  const sent = JSON.parse(instances[0].sent[0])
  assert.equal(sent.jsonrpc, '2.0')
  assert.equal(sent.id, 'probe-1')
  assert.equal(sent.method, 'session.list')

  instances[0].emit('message', replyWithId('probe-1', { result: { sessions: [] } }))
  const result = await promise
  assert.deepEqual(result, { ok: true })
  assert.equal(instances[0].closed, true)
})

test('probe resolves ok on a JSON-RPC ERROR reply with the matching id (dispatcher round-trips)', async () => {
  const { FakeWs, instances } = makeFakeWs()
  const promise = probeGatewayRpc('ws://host/api/ws?token=t', { WebSocketImpl: FakeWs, requestId: 'probe-1' })

  instances[0].emit('open')
  // Unknown method on an older gateway: still a real dispatcher reply.
  instances[0].emit('message', replyWithId('probe-1', { error: { code: -32601, message: 'unknown method' } }))
  const result = await promise
  assert.deepEqual(result, { ok: true })
})

test('probe ignores gateway events and replies to other ids while waiting', async () => {
  const { FakeWs, instances } = makeFakeWs()

  const promise = probeGatewayRpc('ws://host/api/ws?token=t', {
    WebSocketImpl: FakeWs,
    requestId: 'probe-1',
    ...FAST
  })

  instances[0].emit('open')
  instances[0].emit('message', { data: JSON.stringify({ method: 'event', params: { type: 'gateway.ready' } }) })
  instances[0].emit('message', replyWithId('someone-else', { result: {} }))
  instances[0].emit('message', replyWithId('probe-1', { result: 'pong' }))
  const result = await promise
  assert.deepEqual(result, { ok: true })
})

test('probe does NOT resolve on events or replies whose id does not match — only our round-trip counts', async () => {
  const { FakeWs, instances } = makeFakeWs()

  const promise = probeGatewayRpc('ws://host/api/ws?token=t', {
    WebSocketImpl: FakeWs,
    requestId: 'probe-1',
    ...FAST
  })

  instances[0].emit('open')
  // A chatty gateway: ready event + a reply to someone else's request. If the
  // probe accepted ANY frame it would resolve here and never notice that ITS
  // request went unanswered — the exact half-broken-backend state #92927 is
  // about. It must keep waiting and fail when the socket closes unreplied.
  instances[0].emit('message', { data: JSON.stringify({ method: 'event', params: { type: 'gateway.ready' } }) })
  instances[0].emit('message', replyWithId('someone-else', { result: {} }))
  instances[0].emit('close', { code: 1000 })
  const result = await promise
  assert.equal(result.ok, false)
  assert.match(result.reason, /before replying/)
})

test('probe fails when the gateway closes after open without replying', async () => {
  const { FakeWs, instances } = makeFakeWs()
  const promise = probeGatewayRpc('ws://host/api/ws?token=t', { WebSocketImpl: FakeWs, method: 'session.list', ...FAST })

  instances[0].emit('open')
  instances[0].emit('close', { code: 1006 })
  const result = await promise
  assert.equal(result.ok, false)
  assert.match(result.reason, /before replying to "session.list"/)
  assert.match(result.reason, /1006/)
})

test('probe fails on the reply timeout when the upgrade is accepted but no reply ever comes', async () => {
  const { FakeWs, instances } = makeFakeWs()

  const promise = probeGatewayRpc('ws://host/api/ws?token=t', {
    WebSocketImpl: FakeWs,
    method: 'session.list',
    connectTimeoutMs: 1_000,
    replyTimeoutMs: 20
  })

  instances[0].emit('open')
  const result = await promise
  assert.equal(result.ok, false)
  assert.match(result.reason, /Timed out after 20ms waiting for a JSON-RPC reply/)
})

test('probe disarms the connect timer on open: a reply arriving after the connect timeout has elapsed still counts', async () => {
  const { FakeWs, instances } = makeFakeWs()

  const promise = probeGatewayRpc('ws://host/api/ws?token=t', {
    WebSocketImpl: FakeWs,
    requestId: 'probe-1',
    connectTimeoutMs: 20,
    replyTimeoutMs: 1_000
  })

  instances[0].emit('open')
  // The socket opened well before the connect timeout; the reply is just
  // slow (startup import storm). The connect timer must be disarmed at
  // open — otherwise a healthy reply is misreported as "Timed out waiting
  // for the WebSocket to open" even though the upgrade was accepted.
  await new Promise(resolve => setTimeout(resolve, 40))
  instances[0].emit('message', replyWithId('probe-1', { result: { sessions: [] } }))
  const result = await promise
  assert.deepEqual(result, { ok: true })
})

test('probe times out when the socket never opens', async () => {
  const { FakeWs } = makeFakeWs()

  const result = await probeGatewayRpc('ws://host/api/ws?token=t', {
    WebSocketImpl: FakeWs,
    connectTimeoutMs: 20,
    replyTimeoutMs: 1_000
  })

  assert.equal(result.ok, false)
  assert.match(result.reason, /Timed out after 20ms waiting for the WebSocket to open/)
})

test('probe fails gracefully when the constructor throws', async () => {
  class ThrowingWs {
    constructor() {
      throw new Error('boom')
    }
  }

  const result = await probeGatewayRpc('ws://host/api/ws?token=t', { WebSocketImpl: ThrowingWs })
  assert.equal(result.ok, false)
  assert.match(result.reason, /boom/)
})

test('probe fails with a distinct code when WebSocket is unavailable in the runtime', async () => {
  const result = await probeGatewayRpc('ws://host/api/ws?token=t', {})
  assert.equal(result.ok, false)
  assert.match(result.reason, /not available/)
  // The call sites must be able to tell a CAPABILITY problem from a torn
  // install — the unavailable case never gets the `hermes update` advice.
  assert.equal(result.code, 'websocket-unavailable')
})

describe('rpcProbeBootError (#92927 review: distinct capability message, single-source copy)', () => {
  test('a missing WebSocket is a capability message with its own code — never torn-install repair advice', () => {
    const error = rpcProbeBootError('Local Hermes backend', {
      ok: false,
      code: 'websocket-unavailable',
      reason: 'WebSocket is not available in this runtime.'
    })

    assert.equal(error.code, RPC_PROBE_UNAVAILABLE_ERROR_CODE)
    assert.match(error.message, /cannot open a WebSocket/)
    assert.ok(!error.message.includes('interrupted update'))
    assert.ok(!error.message.includes('force-build'))
    assert.ok(!error.message.includes('hermes update'))
  })

  test('a dead dispatcher carries the shared torn-install guidance with the stable failed code', () => {
    const error = rpcProbeBootError('Hermes backend for profile "default"', {
      ok: false,
      reason: 'Timed out after 8000ms waiting for a JSON-RPC reply to "session.list".'
    })

    assert.equal(error.code, RPC_PROBE_FAILED_ERROR_CODE)
    assert.match(error.message, /profile "default"/)
    assert.match(error.message, /JSON-RPC gateway did not answer/)
    assert.match(error.message, /interrupted update/)
    assert.match(error.message, /hermes desktop --force-build/)
  })

  test('both backend labels build byte-identical guidance from the single constant', () => {
    const pool = rpcProbeBootError('Hermes backend for profile "work"', { ok: false, reason: 'x' })
    const primary = rpcProbeBootError('Local Hermes backend', { ok: false, reason: 'x' })

    const guidanceOf = (error: Error) =>
      error.message.slice(error.message.indexOf('The install may be broken'))

    assert.equal(guidanceOf(pool), guidanceOf(primary))
    assert.match(guidanceOf(pool), /^The install may be broken by an interrupted update/)
  })
})
