import { beforeEach, describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayClient } from './json-rpc-gateway'

// Desktop in URL, Cloud or SSH mode (no `native_dial`) talks to a standalone `hermes serve`
// on `/api/ws`. That socket announces a process epoch on gateway.ready. A serve that does not
// know `replay_epoch` validates `session.events.since` with extra="forbid" and answers 4000
// to any other key. The fake server below mirrors that contract.
class FakeWebSocket extends EventTarget {
  static instances: FakeWebSocket[] = []
  readyState = 0
  sent: string[] = []

  constructor(public url: string) {
    super()
    FakeWebSocket.instances.push(this)
  }

  send(data: string): void {
    this.sent.push(data)
  }

  close(): void {
    this.readyState = 3
    this.dispatchEvent(new CloseEvent('close'))
  }

  open(): void {
    this.readyState = 1
    this.dispatchEvent(new Event('open'))
  }

  serverFrame(obj: unknown): void {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(obj) }))
  }

  request(method: string): { id: string; method: string; params: Record<string, unknown> } | undefined {
    return this.sent.map(frame => JSON.parse(frame)).find(frame => frame.method === method)
  }
}

const LEGACY_KEYS = ['session_id', 'last_seen', 'profile']

const ready = (epoch: string) => ({
  jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: { replay_epoch: epoch } }
})

function makeClient(): JsonRpcGatewayClient {
  return new JsonRpcGatewayClient({
    socketFactory: url => new FakeWebSocket(url) as unknown as WebSocket,
    heartbeatIntervalMs: 0,
    heartbeatDeadlineMs: 0,
    connectTimeoutMs: 1000
  })
}

async function reconnect(client: JsonRpcGatewayClient): Promise<FakeWebSocket> {
  client.invalidate('drop')
  const pending = client.connect('ws://x')
  const sock = FakeWebSocket.instances.at(-1)!
  sock.open()
  await pending
  sock.serverFrame(ready('proc-1'))

  return sock
}

describe('reconnect replay against a legacy backend', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
  })

  it('replays the missed events and re-delivers the pending approval', async () => {
    const client = makeClient()
    const seen: string[] = []
    const prompts: Array<{ id: string; method: string; replayed?: boolean }> = []
    client.on('message.complete', () => seen.push('complete'))
    client.onRequest(req => void prompts.push({ id: req.id, method: req.method, replayed: req.replayed }))

    const first = client.connect('ws://x')
    const firstSock = FakeWebSocket.instances.at(-1)!
    firstSock.open()
    await first
    firstSock.serverFrame(ready('proc-1'))
    firstSock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 3 } })

    const sock = await reconnect(client)
    await vi.waitFor(() => expect(sock.request('session.events.since')).toBeDefined())
    const req = sock.request('session.events.since')!
    const extra = Object.keys(req.params).filter(key => !LEGACY_KEYS.includes(key))

    sock.serverFrame(extra.length
      ? { jsonrpc: '2.0', id: req.id, error: { code: 4000, message: `${extra.join(', ')}: Extra inputs are not permitted` } }
      : {
          jsonrpc: '2.0', id: req.id,
          result: {
            events: [{ type: 'message.complete', session_id: 's1', seq: 4 }],
            latest_seq: 4,
            truncated: false,
            count: 1,
            epoch: 'proc-1',
            open_requests: [{ id: 'srq-1', method: 'approval', params: { session_id: 's1' } }]
          }
        })

    await vi.waitFor(() => expect(seen).toEqual(['complete']))
    expect(req.params).toEqual({ session_id: 's1', last_seen: 3 })
    expect(prompts).toEqual([{ id: 'srq-1', method: 'approval', replayed: true }])
    client.close()
  })

  it('still qualifies a canonical replay with that session epoch', async () => {
    const client = makeClient()
    const first = client.connect('ws://x')
    const firstSock = FakeWebSocket.instances.at(-1)!
    firstSock.open()
    await first
    firstSock.serverFrame(ready('proc-1'))
    firstSock.serverFrame({
      jsonrpc: '2.0', method: 'event',
      params: { type: 'message.delta', session_id: 's1', seq: 5, replay_epoch: 's1-A' }
    })

    const sock = await reconnect(client)
    await vi.waitFor(() => expect(sock.request('session.events.since')).toBeDefined())
    expect(sock.request('session.events.since')!.params).toEqual({
      session_id: 's1', last_seen: 5, replay_epoch: 's1-A'
    })
    client.close()
  })
})
