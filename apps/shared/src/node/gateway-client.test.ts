// The shared client with only a `connect` (a non-Ink client): it dials the grant, reads the owner's
// session.create contract before readiness, creates a session, and redials through `connect` with
// the recovery flag when the socket drops.
import { afterEach, expect, it, vi } from 'vitest'

import { GatewayClient, RECONNECT_BASE_MS } from './gateway-client'

class FakeSocket {
  static instances: FakeSocket[] = []
  readyState = 0
  sent: { id: number; method: string; params: Record<string, unknown> }[] = []
  private listeners = new Map<string, ((event: any) => void)[]>()

  constructor(readonly url: string, readonly protocols?: string[]) {
    FakeSocket.instances.push(this)
  }

  addEventListener(type: string, callback: (event: any) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), callback])
  }

  send(text: string) {
    this.sent.push(JSON.parse(text))
  }

  close(code = 1000) {
    this.readyState = 3
    this.emit('close', { code })
  }

  emit(type: string, event: unknown) {
    for (const callback of this.listeners.get(type) ?? []) {
      callback(event)
    }
  }

  open() {
    this.readyState = 1
    this.emit('open', {})
  }

  reply(method: string, result: unknown) {
    const request = this.sent.find(frame => frame.method === method)!

    this.emit('message', { data: JSON.stringify({ jsonrpc: '2.0', id: request.id, result }) })
  }
}

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  FakeSocket.instances = []
})

it('dials the grant, describes the owner, creates a session and redials through connect', async () => {
  vi.useFakeTimers()
  vi.stubGlobal('WebSocket', FakeSocket)

  const connect = vi.fn(async () => ({
    url: 'ws://127.0.0.1:9/api/ws', protocols: ['hermes-gateway-v1', 'hermes-gateway-ticket.t'],
    profile_id: '/home/p', instance_id: 'owner'
  }))

  const gw = new GatewayClient({ connect })
  const events: string[] = []

  gw.on('event', (event: { type: string }) => events.push(event.type))
  gw.start()
  gw.drain()
  await vi.waitFor(() => expect(FakeSocket.instances).toHaveLength(1))
  const socket = FakeSocket.instances[0]!

  expect(connect).toHaveBeenLastCalledWith(true, false)
  expect([socket.url, socket.protocols]).toEqual(['ws://127.0.0.1:9/api/ws', ['hermes-gateway-v1', 'hermes-gateway-ticket.t']])
  socket.open()
  await vi.waitFor(() => expect(socket.sent.some(frame => frame.method === 'runtime.describe')).toBe(true))
  socket.reply('runtime.describe', { session_create: { sources: ['tui'], parameters: ['request_id', 'source'] } })
  await vi.waitFor(() => expect(events).toContain('gateway.ready'))

  const created = gw.request<{ session_id: string }>('session.create', {})

  await vi.waitFor(() => expect(socket.sent.some(frame => frame.method === 'session.create')).toBe(true))
  expect(socket.sent.find(frame => frame.method === 'session.create')!.params).toMatchObject({ source: 'tui' })
  socket.reply('session.create', { session_id: 'S1', messages: [], authority_epoch: 1 })
  expect((await created).session_id).toBe('S1')

  const pending = gw.request('session.list', {})

  await vi.waitFor(() => expect(socket.sent.some(frame => frame.method === 'session.list')).toBe(true))
  socket.close(1006)
  await expect(pending).rejects.toThrow('gateway websocket closed (1006)')
  await vi.advanceTimersByTimeAsync(RECONNECT_BASE_MS)
  expect(connect).toHaveBeenLastCalledWith(false, true)
  gw.kill()
})
