/**
 * Tests for electron/gateway-ws-probe.ts.
 *
 * Run with: node --test electron/gateway-ws-probe.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * The probe drives a real WebSocket handshake for the "Test remote" button.
 * Here we inject a fake socket so we can deterministically replay each handshake
 * outcome (open, frame, error, early close, never-opens) without a network.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { probeGatewayWebSocket } from './gateway-ws-probe.ts'

// Minimal WebSocket double: records listeners synchronously (the probe attaches
// them in its executor) and exposes emit() so the test can replay events.
function makeFakeWs(): { FakeWs: new (url: string) => any; instances: any[] } {
  const instances = []

  class FakeWs {
    url: string
    closed = false
    listeners: Record<string, any[]> = {}
    constructor(url) {
      this.url = url
      this.listeners = {}
      this.closed = false
      instances.push(this)
    }
    addEventListener(type, fn) {
      ;(this.listeners[type] ||= []).push(fn)
    }
    close() {
      this.closed = true
    }
    emit(type, event) {
      for (const fn of this.listeners[type] || []) {
        fn(event)
      }
    }
  }

  return { FakeWs, instances }
}

const FAST = { connectTimeoutMs: 1_000, readyGraceMs: 10 }

test('probe resolves ok when the socket opens and stays open', async () => {
  const { FakeWs, instances } = makeFakeWs()
  const promise = probeGatewayWebSocket('ws://host/api/ws?token=t', { WebSocketImpl: FakeWs, ...FAST })
  instances[0].emit('open')
  const result = await promise
  assert.deepEqual(result, { ok: true })
  assert.equal(instances[0].closed, true)
})

test('probe resolves ok immediately when a frame arrives', async () => {
  const { FakeWs, instances } = makeFakeWs()

  const promise = probeGatewayWebSocket('ws://host/api/ws?token=t', {
    WebSocketImpl: FakeWs,
    connectTimeoutMs: 1_000,
    readyGraceMs: 10_000 // long grace: success must come from the frame, not the timer
  })

  instances[0].emit('open')
  instances[0].emit('message', { data: '{"jsonrpc":"2.0"}' })
  const result = await promise
  assert.deepEqual(result, { ok: true })
})

test('probe fails when the socket errors before opening', async () => {
  const { FakeWs, instances } = makeFakeWs()
  const promise = probeGatewayWebSocket('ws://host/api/ws?token=t', { WebSocketImpl: FakeWs, ...FAST })
  instances[0].emit('error', { message: 'ECONNREFUSED' })
  const result = await promise
  assert.equal(result.ok, false)
  assert.match(result.reason, /ECONNREFUSED/)
})

test('probe fails when the gateway closes before opening', async () => {
  const { FakeWs, instances } = makeFakeWs()
  const promise = probeGatewayWebSocket('ws://host/api/ws?token=t', { WebSocketImpl: FakeWs, ...FAST })
  instances[0].emit('close', { code: 1006 })
  const result = await promise
  assert.equal(result.ok, false)
  assert.match(result.reason, /before it opened/)
  assert.match(result.reason, /1006/)
})

test('probe fails when the gateway accepts then immediately closes (auth rejected)', async () => {
  const { FakeWs, instances } = makeFakeWs()
  const promise = probeGatewayWebSocket('ws://host/api/ws?token=t', { WebSocketImpl: FakeWs, ...FAST })
  instances[0].emit('open')
  instances[0].emit('close', { code: 4403, reason: 'forbidden' })
  const result = await promise
  assert.equal(result.ok, false)
  assert.match(result.reason, /credential rejected/)
  assert.match(result.reason, /4403/)
  assert.match(result.reason, /forbidden/)
})

test('probe times out when the socket never opens', async () => {
  const { FakeWs } = makeFakeWs()

  const result = await probeGatewayWebSocket('ws://host/api/ws?token=t', {
    WebSocketImpl: FakeWs,
    connectTimeoutMs: 20,
    readyGraceMs: 10
  })

  assert.equal(result.ok, false)
  assert.match(result.reason, /Timed out/)
})

test('probe fails gracefully when the constructor throws', async () => {
  class ThrowingWs {
    constructor() {
      throw new Error('bad url')
    }
  }
  const result = await probeGatewayWebSocket('ws://host/api/ws', { WebSocketImpl: ThrowingWs, ...FAST })
  assert.equal(result.ok, false)
  assert.match(result.reason, /bad url/)
})

test('probe reports unavailable when no WebSocket implementation is provided', async () => {
  const result = await probeGatewayWebSocket('ws://host/api/ws', { WebSocketImpl: undefined })
  assert.equal(result.ok, false)
  assert.match(result.reason, /not available/)
})
