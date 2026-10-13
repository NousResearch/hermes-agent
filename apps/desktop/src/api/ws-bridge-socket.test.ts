/**
 * Tests for src/api/ws-bridge-socket.ts (renderer side of the WS bridge).
 *
 * Run with: npx vitest run src/api/ws-bridge-socket.test.ts
 *
 * Pins the renderer-side concurrency/lifecycle invariants:
 *  1. Events are accepted only under the socket's own dial token — concurrent
 *     socket A's frames never replay into socket B (finding: pre-id buffer
 *     cross-contamination; fixed by token-tagged events end to end).
 *  2. close() while CONNECTING cancels the main-process dial, and a late
 *     open-result after close() closes the socket immediately (no orphan).
 *  3. The IPC listener is removed on every terminal outcome — repeated
 *     failure cycles don't accumulate listeners.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { BridgedWebSocket, shouldBridgeWebSocket } from './ws-bridge-socket'

interface RecordedCall {
  method: string
  args: unknown[]
}

function makeBridgeApi() {
  const calls: RecordedCall[] = []
  const listeners = new Set<(token: string, payload: { type: string; data?: string; code?: number; reason?: string }) => void>()
  const pendingOpens = new Map<string, (result: { ok: boolean; error?: string }) => void>()

  const api = {
    wsBridgeOpen: (url: string, token: string) => {
      calls.push({ method: 'open', args: [url, token] })

      return new Promise<{ ok: boolean; error?: string }>(resolve => pendingOpens.set(token, resolve))
    },
    wsBridgeCancel: (token: string) => {
      calls.push({ method: 'cancel', args: [token] })

      return Promise.resolve({ ok: true })
    },
    wsBridgeSend: (token: string, data: string, binary: boolean) => {
      calls.push({ method: 'send', args: [token, data, binary] })

      return Promise.resolve({ ok: true })
    },
    wsBridgeClose: (token: string, code?: number, reason?: string) => {
      calls.push({ method: 'close', args: [token, code, reason] })

      return Promise.resolve({ ok: true })
    },
    onWsBridgeEvent: (cb: (token: string, payload: { type: string; data?: string; code?: number; reason?: string }) => void) => {
      listeners.add(cb)

      return () => listeners.delete(cb)
    }
  }

  return {
    api,
    calls,
    listeners,
    resolveOpen: (token: string, result: { ok: boolean; error?: string } = { ok: true }) =>
      pendingOpens.get(token)!(result),
    emit: (token: string, payload: { type: string; data?: string; code?: number; reason?: string }) => {
      for (const cb of [...listeners]) {cb(token, payload)}
    }
  }
}

const flush = () => new Promise(resolve => setTimeout(resolve, 0))

function tokenOf(calls: RecordedCall[], method = 'open'): string {
  const call = calls.find(c => c.method === method)
  assert.ok(call, `expected ${method} call`)

  return call!.args[1] as string ?? call!.args[0] as string
}

test('concurrent sockets receive only their own token-tagged events', async () => {
  const { api, calls, emit, resolveOpen } = makeBridgeApi()
  const a = new BridgedWebSocket('wss://gw/a', api as never, 'tok-A')
  const b = new BridgedWebSocket('wss://gw/b', api as never, 'tok-B')

  const aEvents: string[] = []
  const bEvents: string[] = []
  a.addEventListener('open', () => aEvents.push('open'))
  a.addEventListener('message', e => aEvents.push(`msg:${(e as MessageEvent).data}`))
  a.addEventListener('close', () => aEvents.push('close'))
  b.addEventListener('open', () => bEvents.push('open'))
  b.addEventListener('message', e => bEvents.push(`msg:${(e as MessageEvent).data}`))
  b.addEventListener('close', () => bEvents.push('close'))

  // A opens first; while B is still awaiting its open result, A traffic flows.
  resolveOpen('tok-A')
  await flush()
  emit('tok-A', { type: 'open' })
  emit('tok-A', { type: 'message', data: 'for-A' })
  emit('tok-A', { type: 'close', code: 1000 })

  resolveOpen('tok-B')
  await flush()
  emit('tok-B', { type: 'open' })
  emit('tok-B', { type: 'message', data: 'for-B' })

  assert.deepEqual(aEvents, ['open', 'msg:for-A', 'close'])
  assert.deepEqual(bEvents, ['open', 'msg:for-B'])
  assert.equal(calls.length >= 2, true)
})

test('close() while CONNECTING cancels the dial; late open-result closes immediately', async () => {
  const { api, calls, resolveOpen } = makeBridgeApi()
  const sock = new BridgedWebSocket('wss://gw/x', api as never, 'tok-X')

  // Client connect timeout: close before open resolved.
  sock.close()
  const cancel = calls.find(c => c.method === 'cancel')
  assert.ok(cancel, 'cancel issued for connecting socket')
  assert.equal(cancel!.args[0], 'tok-X')

  // The dial resolves OK anyway (server was slow, not dead) — must be closed
  // immediately, never surfaced as open.
  let opened = false
  sock.addEventListener('open', () => { opened = true })
  resolveOpen('tok-X', { ok: true })
  await flush()
  assert.equal(opened, false)
  const close = calls.find(c => c.method === 'close')
  assert.ok(close, 'late-opened socket closed')
  assert.equal(close!.args[0], 'tok-X')
  assert.equal(sock.readyState, 3)
})

test('terminal outcomes remove the IPC listener (no accumulation across failures)', async () => {
  const { api, listeners, emit, resolveOpen } = makeBridgeApi()

  // Cycle 1: open fails
  const s1 = new BridgedWebSocket('wss://gw/1', api as never, 'tok-1')
  assert.equal(listeners.size, 1)
  resolveOpen('tok-1', { ok: false, error: 'dial refused' })
  await flush()
  assert.equal(listeners.size, 0)

  // Cycle 2: opens, then remote close
  const s2 = new BridgedWebSocket('wss://gw/2', api as never, 'tok-2')
  assert.equal(listeners.size, 1)
  resolveOpen('tok-2', { ok: true })
  await flush()
  emit('tok-2', { type: 'open' })
  emit('tok-2', { type: 'close', code: 1000 })
  assert.equal(listeners.size, 0)

  // Cycle 3: local close
  const s3 = new BridgedWebSocket('wss://gw/3', api as never, 'tok-3')
  assert.equal(listeners.size, 1)
  s3.close()
  assert.equal(listeners.size, 0)
})

test('send only flows after open, under the socket token', async () => {
  const { api, calls, emit, resolveOpen } = makeBridgeApi()
  const sock = new BridgedWebSocket('wss://gw/s', api as never, 'tok-S')

  sock.send('too-early')
  resolveOpen('tok-S', { ok: true })
  await flush()
  emit('tok-S', { type: 'open' })
  sock.send('hello')

  const sends = calls.filter(c => c.method === 'send')
  assert.equal(sends.length, 1)
  assert.deepEqual(sends[0].args, ['tok-S', 'hello', false])
})

test('shouldBridgeWebSocket: LNA-gated non-loopback ws:// origins dial through main (#54523 F3)', () => {
  const bridged = [
    'wss://gw.example.com/api/ws', // always: renderer WS pool ignores --use-system-certificates
    'ws://100.79.222.28:9119/api/ws', // Tailscale CGNAT 100.64/10 — the #54523 repro
    'ws://192.168.1.10:9119/api/ws', // RFC 1918 — same LNA class
    'ws://10.0.0.5/api/ws',
    'ws://172.16.0.1/api/ws',
    'ws://tailnet-host.ts.net:9443/api/ws', // non-loopback hostname, LNA applies
    'ws://[fd00::1]:9119/api/ws' // ULA IPv6
  ]

  for (const url of bridged) {
    assert.equal(shouldBridgeWebSocket(url), true, url)
  }
})

test('shouldBridgeWebSocket: loopback ws:// keeps the native renderer WebSocket', () => {
  const native = [
    'ws://127.0.0.1:9119/api/ws',
    'ws://127.0.0.1/api/ws',
    'ws://localhost:9119/api/ws',
    'ws://app.localhost/api/ws',
    'ws://[::1]:9119/api/ws',
    'ws://0:0:0:0:0:0:0:1/api/ws',
    'http://127.0.0.1:9119', // not a WS URL at all
    'not a url'
  ]

  for (const url of native) {
    assert.equal(shouldBridgeWebSocket(url), false, url)
  }
})

// #98988's exhibit (Cloudflare-fronted remote gateway): Chromium's renderer
// WebSocket dials /api/ws over RFC 8441 HTTP/2 Extended CONNECT, the cookie
// gate sees a plain HTTP request to a non-public path and answers JSON 401
// before the WS handler runs — the "Test remote passed, chat still says
// Could not connect" split. The renderer must never dial such a URL with
// Chromium's socket: it routes through main's node:tls `ws` dial
// (HTTP/1.1 Upgrade), which is why shouldBridgeWebSocket matches on scheme
// and loopback only, never on the /api/ws path shape.
test('shouldBridgeWebSocket: a Cloudflare-fronted remote wss:// gateway dials through main, not Chromium (#98988)', () => {
  const remote = [
    'wss://gw.example.com/api/ws?ticket=t', // remote host, ticket auth
    'wss://hermes.example.com/api/ws', // #98988's exact shape: non-loopback /api/ws
    'ws://office-gw.internal:9443/api/ws' // plain-ws remote (HTTP/2 CONNECT peer)
  ]

  for (const url of remote) {
    assert.equal(shouldBridgeWebSocket(url), true, url)
  }

  // The loopback exemption is what keeps local backends on the native socket —
  // #98988 keeps `hermes serve` loopback dials in the renderer too.
  assert.equal(shouldBridgeWebSocket('ws://127.0.0.1:52515/api/ws?token=t'), false)
  assert.equal(shouldBridgeWebSocket('ws://localhost:9/api/ws'), false)
})
