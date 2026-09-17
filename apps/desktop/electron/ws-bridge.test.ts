/**
 * Tests for electron/ws-bridge.ts.
 *
 * Run with: node --test electron/ws-bridge.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * The bridge moves remote gateway wss:// dials from the renderer into the main
 * process so node:tls (NODE_EXTRA_CA_CERTS) owns the TLS handshake. These tests
 * pin the authority/lifecycle edges that move creates:
 *
 *  1. Main-owned per-URL headers reach the `ws` dial (the contract the
 *     renderer webRequest.onBeforeSendHeaders path previously enforced).
 *  2. send/close enforce WebContents ownership; owner destruction retires
 *     that owner's sockets and pending dials.
 *  3. A canceled dial token is never promoted into the live map — including
 *     when the underlying ws opens AFTER the cancel (client connect timeout).
 *  4. Events are emitted under the dial token, so concurrent sockets never
 *     cross-deliver frames.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createWebSocketBridge, type WsLike } from './ws-bridge'

// ── Doubles ──────────────────────────────────────────────────────────────

function makeFakeIpc() {
  const handlers = new Map<string, (...args: unknown[]) => unknown>()
  return {
    handlers,
    ipc: { handle: (channel: string, fn: (...args: unknown[]) => unknown) => void handlers.set(channel, fn) } as never
  }
}

function makeSender(label: string) {
  const sent: Array<{ token: string; payload: unknown }> = []
  const listeners = new Map<string, Array<() => void>>()
  return {
    label,
    sent,
    destroyed: false,
    get destroyedListenerCount() { return (listeners.get('destroyed') ?? []).length },
    isDestroyed() { return this.destroyed },
    send(_channel: string, token: string, payload: unknown) { sent.push({ token, payload }) },
    once(event: string, fn: () => void) { (listeners.get(event) ?? listeners.set(event, []).get(event)!).push(fn) },
    destroy() {
      this.destroyed = true
      for (const fn of listeners.get('destroyed') ?? []) fn()
    }
  }
}

function makeWsFactory() {
  const instances: Array<FakeWs & { url: string; options: { headers?: Record<string, string> } }> = []

  class FakeWs implements WsLike {
    readyState = 0
    sent: unknown[] = []
    terminated = false
    closed = false
    url = ''
    options: { headers?: Record<string, string> } = {}
    private listeners = new Map<string, Array<(...args: never[]) => void>>()

    constructor(url: string, options: { headers?: Record<string, string> } = {}) {
      this.url = url
      this.options = options
      instances.push(this as never)
    }
    on(event: string, fn: (...args: never[]) => void) {
      (this.listeners.get(event) ?? this.listeners.set(event, []).get(event)!).push(fn)
    }
    private emit(event: string, ...args: unknown[]) {
      for (const fn of this.listeners.get(event) ?? []) fn(...(args as never[]))
    }
    send(data: unknown) { this.sent.push(data) }
    close() { this.closed = true; this.readyState = 3 }
    terminate() { this.terminated = true }
    simulateOpen() { this.readyState = 1; this.emit('open') }
    simulateMessage(data: string) { this.emit('message', data, false) }
    simulateClose(code = 1000) { this.readyState = 3; this.emit('close', code, Buffer.from('')) }
  }

  return { instances, FakeWs }
}

const nextTick = () => new Promise(resolve => setImmediate(resolve))

// ── Tests ────────────────────────────────────────────────────────────────

test('main-owned headers are passed to the ws dial', async () => {
  const { handlers, ipc } = makeFakeIpc()
  const { instances, FakeWs } = makeWsFactory()
  createWebSocketBridge({
    ipc,
    webSocketImpl: FakeWs as never,
    headersForUrl: url => url.includes('gw.example') ? { 'cf-access-client-id': 'abc123' } : {}
  }).install()

  const sender = makeSender('a')
  const openP = handlers.get('hermes:ws-bridge:open')!({ sender }, 'wss://gw.example/api/ws', 'tok-1') as Promise<{ ok: boolean }>
  assert.equal(instances[0].options.headers?.['cf-access-client-id'], 'abc123')
  instances[0].simulateOpen()
  assert.deepEqual(await openP, { ok: true })
})

test('send/close enforce WebContents ownership; owner destruction retires its sockets', async () => {
  const { handlers, ipc } = makeFakeIpc()
  const { instances, FakeWs } = makeWsFactory()
  const bridge = createWebSocketBridge({ ipc, webSocketImpl: FakeWs as never })
  bridge.install()

  const a = makeSender('a')
  const b = makeSender('b')
  const openP = handlers.get('hermes:ws-bridge:open')!({ sender: a }, 'wss://gw.example/api/ws', 'tok-a') as Promise<unknown>
  instances[0].simulateOpen()
  await openP
  await nextTick()

  // B cannot send or close A's socket
  assert.deepEqual(handlers.get('hermes:ws-bridge:send')!({ sender: b }, 'tok-a', 'x', false), { ok: false })
  assert.deepEqual(handlers.get('hermes:ws-bridge:close')!({ sender: b }, 'tok-a'), { ok: false })
  assert.equal(instances[0].sent.length, 0)
  assert.equal(instances[0].closed, false)

  // A can send
  assert.deepEqual(handlers.get('hermes:ws-bridge:send')!({ sender: a }, 'tok-a', 'hello', false), { ok: true })
  assert.equal(instances[0].sent.length, 1)

  // Owner destruction retires the socket and all future mutations fail
  a.destroy()
  assert.equal(instances[0].terminated, true)
  assert.equal(bridge.sockets.size, 0)
  assert.deepEqual(handlers.get('hermes:ws-bridge:send')!({ sender: a }, 'tok-a', 'y', false), { ok: false })
})

test('successful open does NOT terminate the socket (live socket stays usable)', async () => {
  const { handlers, ipc } = makeFakeIpc()
  const { instances, FakeWs } = makeWsFactory()
  const bridge = createWebSocketBridge({ ipc, webSocketImpl: FakeWs as never })
  bridge.install()

  const sender = makeSender('a')
  const openP = handlers.get('hermes:ws-bridge:open')!({ sender }, 'wss://gw.example/api/ws', 'tok-live') as Promise<{ ok: boolean }>
  instances[0].simulateOpen()
  assert.deepEqual(await openP, { ok: true })

  // finalizeDial must not terminate on success — the socket is live traffic now.
  assert.equal(instances[0].terminated, false)
  assert.equal(bridge.sockets.size, 1)
  assert.deepEqual(handlers.get('hermes:ws-bridge:send')!({ sender }, 'tok-live', 'ping', false), { ok: true })
  assert.deepEqual(instances[0].sent, ['ping'])
})

test('cancel settles the original open IPC invoke with a terminal receipt', async () => {
  const { handlers, ipc } = makeFakeIpc()
  const { instances, FakeWs } = makeWsFactory()
  const bridge = createWebSocketBridge({ ipc, webSocketImpl: FakeWs as never })
  bridge.install()

  const sender = makeSender('a')
  const openP = handlers.get('hermes:ws-bridge:open')!({ sender }, 'wss://gw.example/api/ws', 'tok-settle') as Promise<{ ok: boolean; error?: string }>

  assert.deepEqual(handlers.get('hermes:ws-bridge:cancel')!({ sender }, 'tok-settle'), { ok: true })
  // The open invoke must publish its terminal settlement, not hang forever.
  const result = await openP
  assert.equal(result.ok, false)
  assert.equal(bridge.pendingDials.size, 0)
  assert.equal(bridge.sockets.size, 0)
  assert.equal(instances[0].terminated, true)

  // A late underlying open can neither promote nor double-settle.
  instances[0].terminated = false
  instances[0].readyState = 0
  instances[0].simulateOpen()
  await nextTick()
  assert.equal(bridge.sockets.size, 0)
  assert.equal(instances[0].terminated, true)
})

test('owner destruction settles a pending dial with a terminal receipt', async () => {
  const { handlers, ipc } = makeFakeIpc()
  const { instances, FakeWs } = makeWsFactory()
  const bridge = createWebSocketBridge({ ipc, webSocketImpl: FakeWs as never })
  bridge.install()

  const sender = makeSender('a')
  const openP = handlers.get('hermes:ws-bridge:open')!({ sender }, 'wss://gw.example/api/ws', 'tok-orphan') as Promise<{ ok: boolean; error?: string }>

  sender.destroy()
  const result = await openP
  assert.equal(result.ok, false)
  assert.equal(bridge.pendingDials.size, 0)
  assert.equal(bridge.sockets.size, 0)
  assert.equal(instances[0].terminated, true)

  // Late open after teardown cannot promote.
  instances[0].terminated = false
  instances[0].readyState = 0
  instances[0].simulateOpen()
  await nextTick()
  assert.equal(bridge.sockets.size, 0)
  assert.equal(instances[0].terminated, true)
})

test('repeated dials from one sender register a single destroyed listener', async () => {
  const { handlers, ipc } = makeFakeIpc()
  const { instances, FakeWs } = makeWsFactory()
  createWebSocketBridge({ ipc, webSocketImpl: FakeWs as never }).install()

  const sender = makeSender('a')
  for (let i = 0; i < 3; i++) {
    void (handlers.get('hermes:ws-bridge:open')!({ sender }, 'wss://gw.example/api/ws', `tok-loop-${i}`) as Promise<unknown>)
    instances[i].simulateOpen()
    await nextTick()
  }
  // watchSender dedupes via WeakSet: exactly one destroyed hook per WebContents.
  assert.equal(sender.destroyedListenerCount, 1)
})

test('cancel from a non-owner sender is refused', async () => {
  const { handlers, ipc } = makeFakeIpc()
  const { instances, FakeWs } = makeWsFactory()
  createWebSocketBridge({ ipc, webSocketImpl: FakeWs as never }).install()

  const a = makeSender('a')
  const b = makeSender('b')
  void (handlers.get('hermes:ws-bridge:open')!({ sender: a }, 'wss://gw.example/api/ws', 'tok-d') as Promise<unknown>)
  assert.deepEqual(handlers.get('hermes:ws-bridge:cancel')!({ sender: b }, 'tok-d'), { ok: false })
  assert.equal(instances[0].terminated, false)
})

test('events carry the dial token end to end (concurrent sockets cannot cross-deliver)', async () => {
  const { handlers, ipc } = makeFakeIpc()
  const { instances, FakeWs } = makeWsFactory()
  createWebSocketBridge({ ipc, webSocketImpl: FakeWs as never }).install()

  const a = makeSender('a')
  void (handlers.get('hermes:ws-bridge:open')!({ sender: a }, 'wss://gw.example/api/ws', 'tok-A') as Promise<unknown>)
  void (handlers.get('hermes:ws-bridge:open')!({ sender: a }, 'wss://gw.example/api/ws', 'tok-B') as Promise<unknown>)

  instances[0].simulateOpen() // tok-A
  instances[1].simulateOpen() // tok-B
  await nextTick()
  instances[0].simulateMessage('frame-for-A')
  await nextTick()

  const byToken = (t: string) => a.sent.filter(e => e.token === t).map(e => (e.payload as { type: string; data?: string }))
  assert.deepEqual(byToken('tok-A').map(p => p.type), ['open', 'message'])
  assert.deepEqual(byToken('tok-B').map(p => p.type), ['open'])
  assert.equal(byToken('tok-A')[1].data, 'frame-for-A')
})

test('remote close during connect resolves open as failure and emits close', async () => {
  const { handlers, ipc } = makeFakeIpc()
  const { instances, FakeWs } = makeWsFactory()
  createWebSocketBridge({ ipc, webSocketImpl: FakeWs as never }).install()

  const sender = makeSender('a')
  const openP = handlers.get('hermes:ws-bridge:open')!({ sender }, 'wss://gw.example/api/ws', 'tok-e') as Promise<{ ok: boolean; error?: string }>
  instances[0].simulateClose(1006)
  const result = await openP
  assert.equal(result.ok, false)
  await nextTick()
  assert.equal(sender.sent.some(e => e.token === 'tok-e' && (e.payload as { type: string }).type === 'close'), true)
})

test('duplicate or empty tokens are refused', async () => {
  const { handlers, ipc } = makeFakeIpc()
  const { instances, FakeWs } = makeWsFactory()
  createWebSocketBridge({ ipc, webSocketImpl: FakeWs as never }).install()

  const sender = makeSender('a')
  const p1 = handlers.get('hermes:ws-bridge:open')!({ sender }, 'wss://gw.example/api/ws', 'dup') as Promise<unknown>
  const p2 = handlers.get('hermes:ws-bridge:open')!({ sender }, 'wss://gw.example/api/ws', 'dup') as Promise<{ ok: boolean }>
  assert.equal((await p2).ok, false)
  const p3 = handlers.get('hermes:ws-bridge:open')!({ sender }, 'wss://gw.example/api/ws', '') as Promise<{ ok: boolean }>
  assert.equal((await p3).ok, false)
  instances[0].simulateOpen()
  await p1
})
