import test from 'node:test'
import assert from 'node:assert/strict'
import { createCdpClient } from './cdp-client.mjs'

// Fake CDP primitives.
function fakeDiscoverTarget(returns, { throws = false } = {}) {
  let calls = 0
  return Object.assign(
    async () => {
      calls++
      if (throws) throw new Error('no target')
      return returns
    },
    { get calls() { return calls } }
  )
}

function fakeCDP(connectBehavior = 'ok') {
  let opened = 0
  const fake = {
    _closed: false,
    on() {},
    close() { this._closed = true },
    connect: async () => {
      opened++
      if (connectBehavior === 'throw') throw new Error('ws open failed')
      return fake
    }
  }
  return Object.assign(fake, { get opened() { return opened } })
}

test('BUG #2 (RED): connect resets handle on CDP.open failure so next call rediscovers', async () => {
  const discover = fakeDiscoverTarget({ webSocketDebuggerUrl: 'ws://x' })
  const cdp = fakeCDP('throw')
  const client = createCdpClient({ port: 9333, match: '5174', onConsole: () => {}, discoverTargetImpl: discover, CDPImpl: cdp })

  await assert.rejects(() => client.connect(), /ws open failed/)
  // After failure, handle must be null (not a half-open socket).
  assert.equal(client.handle, null, 'handle must be reset to null on failure')
  // Second call must also fail cleanly (no stale handle returned).
  await assert.rejects(() => client.connect(), /ws open failed/)
  assert.equal(client.handle, null, 'handle stays null after repeated failure')
})

test('BUG #2 (RED): invalidate() clears handle so a closed socket is not reused', async () => {
  const discover = fakeDiscoverTarget({ webSocketDebuggerUrl: 'ws://x' })
  const cdp = fakeCDP('ok')
  const client = createCdpClient({ port: 9333, match: '5174', onConsole: () => {}, discoverTargetImpl: discover, CDPImpl: cdp })

  const h = await client.connect()
  assert.ok(h, 'connected')
  client.invalidate()
  assert.equal(client.handle, null, 'handle must be cleared after invalidate()')
  // reconnecting must succeed (not return a stale/closed handle)
  const h2 = await client.connect()
  assert.ok(h2, 'reconnected after invalidate')
})

test('BUG #3 (RED): status/connect agreement — page without webSocketDebuggerUrl is NOT alive', async () => {
  // Mirror of server.mjs status(): a target with type:page but no WS url.
  const list = [{ type: 'page', url: 'http://127.0.0.1:5174/#/', title: 'Hermes' /* no webSocketDebuggerUrl */ }]
  // discoverTarget predicate: requires webSocketDebuggerUrl to be a string.
  const discover = fakeDiscoverTarget(list)
  const cdp = fakeCDP('ok')
  const client = createCdpClient({ port: 9333, match: '5174', onConsole: () => {}, discoverTargetImpl: discover, CDPImpl: cdp })

  // status() equivalent check: does the target satisfy the connectable predicate?
  const isConnectable = (t) => t.type === 'page' && typeof t.webSocketDebuggerUrl === 'string'
  assert.equal(isConnectable(list[0]), false, 'page without WS url must be considered NOT connectable')

  // And connect() must reject (the wrapped discoverTarget filters out the
  // non-connectable page and surfaces a friendly "No CDP target" error).
  const client2 = createCdpClient({
    port: 9333,
    match: '5174',
    onConsole: () => {},
    discoverTargetImpl: async () => {
      const l = await discover()
      const pages = l.filter((t) => t.type === 'page' && typeof t.webSocketDebuggerUrl === 'string')
      if (!pages.length) throw new Error('no connectable page')
      return pages[0]
    },
    CDPImpl: cdp
  })
  await assert.rejects(() => client2.connect(), /No CDP target/)
})

// --- A1: auto-invalidate on socket close ---

// CDP fake whose handles carry a fake ws that records close listeners.
function countingCDP() {
  let connectCalls = 0
  const sockets = [] // { listeners } per connect, in order
  const fake = {
    on() {},
    connect: async () => {
      connectCalls++
      const listeners = []
      sockets.push({ listeners })
      const handle = {
        on() {},
        ws: { addEventListener: (ev, fn) => { if (ev === 'close') listeners.push(fn) } }
      }
      return handle
    }
  }
  return {
    cdp: fake,
    connectCalls: () => connectCalls,
    // Fire the close listeners of the Nth socket (1-based; default = last).
    closeSocket: (n = sockets.length) => {
      for (const fn of sockets[n - 1]?.listeners ?? []) fn()
    }
  }
}

test('A1 (RED): socket close invalidates the cached handle automatically', async () => {
  const discover = fakeDiscoverTarget({ webSocketDebuggerUrl: 'ws://x' })
  const { cdp, closeSocket, connectCalls } = countingCDP()
  const client = createCdpClient({ port: 9333, match: '5174', onConsole: () => {}, discoverTargetImpl: discover, CDPImpl: cdp })

  const h1 = await client.connect()
  assert.ok(h1, 'first connect succeeds')

  closeSocket() // renderer reloads / window closes
  assert.equal(client.handle, null, 'close must auto-clear the cached handle')

  const h2 = await client.connect()
  assert.ok(h2, 'reconnect after close succeeds')
  assert.notEqual(h2, h1, 'reconnect must create a fresh handle')
  assert.equal(connectCalls(), 2, 'exactly two CDP connects')
})

test('A1: stale close event must not clear a newer handle', async () => {
  const discover = fakeDiscoverTarget({ webSocketDebuggerUrl: 'ws://x' })
  const { cdp, closeSocket } = countingCDP()
  const client = createCdpClient({ port: 9333, match: '5174', onConsole: () => {}, discoverTargetImpl: discover, CDPImpl: cdp })

  await client.connect() // socket 1
  client.invalidate()
  await client.connect() // socket 2
  const fresh = client.handle

  closeSocket(1) // STALE socket's close arrives late
  assert.equal(client.handle, fresh, 'a stale close must not null the fresh handle')

  closeSocket(2) // fresh socket closes
  assert.equal(client.handle, null, 'the fresh socket close does clear the handle')
})
