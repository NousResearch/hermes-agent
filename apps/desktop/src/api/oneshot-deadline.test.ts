// @vitest-environment node
import { expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

interface PendingReply {
  method: string
  respond: (result: unknown) => void
}

async function withGateway(
  mode: 'native' | 'legacy',
  run: (client: HermesGateway, replies: PendingReply[]) => Promise<void>
) {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const replies: PendingReply[] = []

  server.on('connection', (socket: any) => {
    socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: {} } }))
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())

      replies.push({
        method: frame.method,
        respond: result => socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }))
      })
    })
  })
  const client = new HermesGateway()

  try {
    const address = server.address() as { port: number }
    const query = mode === 'native' ? 'native_dial=fixture&ticket=fixture' : 'ticket=fixture'
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?${query}`)
    vi.useFakeTimers({ toFake: ['Date', 'setTimeout', 'clearTimeout'] })
    await run(client, replies)
  } finally {
    vi.useRealTimers()
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
}

test.each(['native', 'legacy'] as const)('a %s one-shot can return after the generic RPC deadline', async mode => {
  await withGateway(mode, async (client, replies) => {
    let settled = false

    const result = client.request('llm.oneshot', { template: 'commit_message' })
      .then(value => ({ value }), error => ({ error }))
      .finally(() => { settled = true })

    await vi.waitFor(() => expect(replies.some(reply => reply.method === 'llm.oneshot')).toBe(true))
    await vi.advanceTimersByTimeAsync(45_000)
    expect(settled).toBe(false)
    replies.find(reply => reply.method === 'llm.oneshot')!.respond({ text: 'fix: keep a slow model response' })
    await expect(result).resolves.toEqual({ value: { text: 'fix: keep a slow model response' } })
  })
})

test.each([
  { name: 'bounded one-shot default', method: 'llm.oneshot', timeout: undefined, elapsed: 120_001 },
  { name: 'explicit shorter deadline', method: 'llm.oneshot', timeout: 1_000, elapsed: 1_001 },
  { name: 'ordinary RPC default', method: 'model.options', timeout: undefined, elapsed: 30_001 },
  { name: 'explicit no-deadline cancellation', method: 'llm.oneshot', timeout: 0, elapsed: 120_001 }
])('$name still expires or can be cancelled', async ({ method, timeout, elapsed }) => {
  await withGateway('native', async (client, replies) => {
    let settled = false

    const controller = new AbortController()

    const result = client.request(method, {}, timeout, controller.signal)
      .then(value => ({ value }), error => ({ error }))
      .finally(() => { settled = true })

    await vi.waitFor(() => expect(replies.some(reply => reply.method === method)).toBe(true))
    await vi.advanceTimersByTimeAsync(elapsed)

    if (timeout === 0) {
      expect(settled).toBe(false)
      controller.abort()
    }

    const outcome = await result
    expect(outcome).toHaveProperty('error')
    expect((outcome as { error: Error }).error.message).toMatch(timeout === 0 ? /abort|cancel/i : /timed out/i)
  })
})
