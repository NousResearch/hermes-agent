import { describe, expect, it, vi } from 'vitest'

import { JsonRpcRequestChannel, type JsonRpcTransport } from './json-rpc-channel.js'

const spyTransport = () => {
  const sent: string[] = []
  const transport: JsonRpcTransport = { send: text => void sent.push(text) }

  return { sent, transport }
}

const lastId = (sent: string[]): string => (JSON.parse(sent.at(-1) ?? '{}') as { id: string }).id

const openRequestsFrame = (id: string) =>
  JSON.stringify({
    id,
    jsonrpc: '2.0',
    result: { open_requests: [{ id: 'srq-9', method: 'approval', params: { session_id: 's1' } }], session_id: 's1' }
  })

describe('JsonRpcRequestChannel open_requests ordering gate (#C6)', () => {
  it('presents the re-delivered requests only after the gate settles, never holding the caller', async () => {
    let releaseGate!: () => void

    const gate = new Promise<void>(resolve => {
      releaseGate = resolve
    })

    const delivered: Array<{ id: string; replayed?: boolean }> = []
    const channel = new JsonRpcRequestChannel({ beforeOpenRequests: () => gate })
    const { sent, transport } = spyTransport()

    channel.attach(transport)
    channel.onRequest(req => void delivered.push({ id: req.id, replayed: req.replayed }))

    const resume = channel.request<{ session_id: string }>('session.resume', { session_id: 's1' })

    channel.handleFrame(openRequestsFrame(lastId(sent)))

    // The caller — and with it the replay barriers / whenApplicationUsable — never waits on the gate.
    await expect(resume).resolves.toMatchObject({ session_id: 's1' })
    expect(delivered).toEqual([])

    releaseGate()

    await vi.waitFor(() => expect(delivered).toEqual([{ id: 'srq-9', replayed: true }]))
  })

  it('still presents the requests when the gate rejects', async () => {
    const delivered: string[] = []
    const channel = new JsonRpcRequestChannel({ beforeOpenRequests: () => Promise.reject(new Error('no gateway')) })
    const { sent, transport } = spyTransport()

    channel.attach(transport)
    channel.onRequest(req => void delivered.push(req.id))

    channel.request('session.resume', { session_id: 's1' })

    channel.handleFrame(openRequestsFrame(lastId(sent)))

    await vi.waitFor(() => expect(delivered).toEqual(['srq-9']))
  })

  it('delivers synchronously, in the same tick as the frame, when no gate is installed', () => {
    const delivered: string[] = []
    const channel = new JsonRpcRequestChannel()
    const { sent, transport } = spyTransport()

    channel.attach(transport)
    channel.onRequest(req => void delivered.push(req.id))

    channel.request('session.resume', { session_id: 's1' })

    channel.handleFrame(openRequestsFrame(lastId(sent)))

    expect(delivered).toEqual(['srq-9'])
  })
})
