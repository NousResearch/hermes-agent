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

import { afterEach, beforeEach, describe, test, vi } from 'vitest'

import {
  DEFAULT_CONNECT_TIMEOUT_MS,
  DEFAULT_PROGRESS_CHECK_INTERVAL_MS,
  probeGatewayWebSocket,
  SPAWNED_BACKEND_MAX_CONNECT_WAIT_MS,
  spawnedBackendProbeOptions
} from './gateway-ws-probe'

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

// --- waiting on a spawned backend (#96177) ----------------------------------
// A Windows cold start can stall the backend's event loop for 12-28s after
// HTTP is up, leaving the upgrade unanswered past the fixed budget. Fake timers
// drive every deadline so these cases never race the wall clock.

async function advance(ms: number) {
  await vi.advanceTimersByTimeAsync(ms)
}

// Observe settlement without awaiting, so a test can assert "still pending".
function track<T>(promise: Promise<T>) {
  const state: { result: T | undefined; settled: boolean } = { result: undefined, settled: false }
  void promise.then(result => {
    state.result = result
    state.settled = true
  })

  return state
}

const WAITING = { connectTimeoutMs: 30, readyGraceMs: 10, progressCheckIntervalMs: 5, maxConnectWaitMs: 500 }

describe('keepWaitingWhile', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  test('keeps waiting past the base budget while the callback holds, then succeeds on open', async () => {
    const { FakeWs, instances } = makeFakeWs()

    const probe = track(
      probeGatewayWebSocket('ws://host/api/ws?token=t', {
        WebSocketImpl: FakeWs,
        ...WAITING,
        keepWaitingWhile: () => true
      })
    )

    await advance(60)
    assert.equal(probe.settled, false, 'still waiting well past the 30ms base budget')

    instances[0].emit('open')
    await advance(10)
    assert.deepEqual(probe.result, { ok: true })
  })

  test('fails at the next check once the callback turns false', async () => {
    const { FakeWs } = makeFakeWs()
    let alive = true

    const probe = track(
      probeGatewayWebSocket('ws://host/api/ws?token=t', {
        WebSocketImpl: FakeWs,
        ...WAITING,
        keepWaitingWhile: () => alive
      })
    )

    await advance(40)
    assert.equal(probe.settled, false)

    alive = false
    await advance(5)
    assert.equal(probe.result?.ok, false)
    assert.match(probe.result?.reason, /backend stopped after the 30ms budget/)
  })

  test('never waits past maxConnectWaitMs even while the callback holds', async () => {
    const { FakeWs } = makeFakeWs()

    const probe = track(
      probeGatewayWebSocket('ws://host/api/ws?token=t', {
        WebSocketImpl: FakeWs,
        ...WAITING,
        maxConnectWaitMs: 60,
        keepWaitingWhile: () => true
      })
    )

    await advance(59)
    assert.equal(probe.settled, false)

    await advance(1)
    assert.equal(probe.result?.ok, false)
    assert.match(probe.result?.reason, /Timed out after 60ms .*still running at the 60ms cap/)
  })

  test('a callback that holds with no explicit cap cannot extend past the base budget', async () => {
    const { FakeWs } = makeFakeWs()

    const probe = track(
      probeGatewayWebSocket('ws://host/api/ws?token=t', {
        WebSocketImpl: FakeWs,
        connectTimeoutMs: 30,
        readyGraceMs: 10,
        keepWaitingWhile: () => true
      })
    )

    await advance(30)
    assert.equal(probe.result?.ok, false)
    assert.match(probe.result?.reason, /still running at the 30ms cap/)
  })

  test('a throwing callback fails closed at the base budget', async () => {
    const { FakeWs } = makeFakeWs()

    const probe = track(
      probeGatewayWebSocket('ws://host/api/ws?token=t', {
        WebSocketImpl: FakeWs,
        ...WAITING,
        keepWaitingWhile: () => {
          throw new Error('boom')
        }
      })
    )

    await advance(30)
    assert.deepEqual(probe.result, {
      ok: false,
      reason: 'Timed out after 30ms waiting for the WebSocket to open.'
    })
  })

  test('without a callback the base budget is a single fixed deadline', async () => {
    const { FakeWs } = makeFakeWs()

    const probe = track(
      probeGatewayWebSocket('ws://host/api/ws?token=t', {
        WebSocketImpl: FakeWs,
        connectTimeoutMs: 30,
        readyGraceMs: 10
      })
    )

    await advance(29)
    assert.equal(probe.settled, false)

    await advance(1)
    assert.deepEqual(probe.result, {
      ok: false,
      reason: 'Timed out after 30ms waiting for the WebSocket to open.'
    })
  })
})

describe('spawnedBackendProbeOptions', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  test('outlasts a cold-start stall longer than the base budget while the child is alive', async () => {
    const { FakeWs, instances } = makeFakeWs()
    const stallMs = DEFAULT_CONNECT_TIMEOUT_MS * 3

    const probe = track(
      probeGatewayWebSocket('ws://127.0.0.1:1/api/ws?token=t', {
        WebSocketImpl: FakeWs,
        ...spawnedBackendProbeOptions(() => true)
      })
    )

    await advance(stallMs)
    assert.equal(probe.settled, false)

    instances[0].emit('message')
    await advance(0)
    assert.deepEqual(probe.result, { ok: true })
  })

  test('a child that exits mid-wait fails promptly instead of holding the boot to the cap', async () => {
    const { FakeWs } = makeFakeWs()
    let alive = true

    const probe = track(
      probeGatewayWebSocket('ws://127.0.0.1:1/api/ws?token=t', {
        WebSocketImpl: FakeWs,
        ...spawnedBackendProbeOptions(() => alive)
      })
    )

    await advance(DEFAULT_CONNECT_TIMEOUT_MS * 2)
    alive = false
    await advance(DEFAULT_PROGRESS_CHECK_INTERVAL_MS)

    assert.equal(probe.result?.ok, false)
    assert.match(probe.result?.reason, /backend stopped/)
  })

  test('a live backend that never answers still fails, at the bounded cap', async () => {
    const { FakeWs } = makeFakeWs()

    const probe = track(
      probeGatewayWebSocket('ws://127.0.0.1:1/api/ws?token=t', {
        WebSocketImpl: FakeWs,
        ...spawnedBackendProbeOptions(() => true)
      })
    )

    await advance(SPAWNED_BACKEND_MAX_CONNECT_WAIT_MS - 1)
    assert.equal(probe.settled, false)

    await advance(1)
    assert.equal(probe.result?.ok, false)
    assert.match(probe.result?.reason, /cap/)
    assert.ok(SPAWNED_BACKEND_MAX_CONNECT_WAIT_MS > DEFAULT_CONNECT_TIMEOUT_MS)
  })
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

test('probe passes extra upgrade headers to the WebSocket constructor (Cloudflare Access)', async () => {
  const { FakeWs, instances } = makeFakeWs()
  const seen: any[] = []

  class HeaderFakeWs extends FakeWs {
    constructor(url, options?) {
      super(url)
      seen.push(options)
    }
  }

  const headers = { 'CF-Access-Client-Id': 'id', 'CF-Access-Client-Secret': 'secret' }

  const pending = probeGatewayWebSocket('wss://x/api/ws?token=t', {
    WebSocketImpl: HeaderFakeWs,
    headers,
    readyGraceMs: 1
  })

  instances[0].emit('open', {})
  instances[0].emit('message', {})

  const result = await pending

  assert.equal(result.ok, true)
  assert.deepEqual(seen[0], { headers })

  // No headers → the constructor is called with the URL alone (browser-safe).
  const bare = makeFakeWs()
  const seenBare: any[] = []

  class BareWs extends bare.FakeWs {
    constructor(url, options?) {
      super(url)
      seenBare.push(options)
    }
  }

  const barePending = probeGatewayWebSocket('wss://x/api/ws?token=t', { WebSocketImpl: BareWs, readyGraceMs: 1 })

  bare.instances[0].emit('open', {})
  bare.instances[0].emit('message', {})
  await barePending
  assert.equal(seenBare[0], undefined)
})
