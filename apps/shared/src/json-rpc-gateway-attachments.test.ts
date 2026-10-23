import { afterEach, describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayClient } from './json-rpc-gateway'

class UploadSocket extends EventTarget {
  readyState = WebSocket.CONNECTING as number
  frames: { id: string; method: string }[] = []

  constructor(readonly url: string) {
    super()
    queueMicrotask(() => {
      this.readyState = WebSocket.OPEN
      this.dispatchEvent(new Event('open'))
      this.receive({ method: 'event', params: { type: 'gateway.ready', payload: { heartbeat: true } } })
    })
  }

  receive(frame: unknown): void {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame) }))
  }

  send(text: string): void {
    const frame = JSON.parse(text)
    this.frames.push(frame)

    if (frame.method === 'gateway.ping') {
      this.receive({ id: frame.id, result: {} })
    }
  }

  close(): void {
    this.readyState = WebSocket.CLOSED
    this.dispatchEvent(new CloseEvent('close', { code: 1000 }))
  }
}

const setup = async () => {
  const sockets: UploadSocket[] = []

  const client = new JsonRpcGatewayClient({
    requestTimeoutMs: 30_000,
    socketFactory: url => {
      const socket = new UploadSocket(url)
      sockets.push(socket)

      return socket as unknown as WebSocket
    }
  })

  await client.connect('ws://gateway.test/api/ws?token=test')

  return { client, sockets }
}

afterEach(() => vi.useRealTimers())

describe('attachment upload connections', () => {
  it('a slow upload keeps control RPCs and heartbeats responsive beyond the normal deadline', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
    const { client, sockets } = await setup()
    const upload = client.request('file.attach', {
      session_id: 'session',
      data_url: 'data:application/zip;base64,AA=='
    })
    await Promise.resolve()
    const control = client.request('session.info', { session_id: 'session' })
    const info = sockets[0].frames.find(frame => frame.method === 'session.info')!
    sockets[0].receive({ id: info.id, result: { alive: true } })
    await expect(control).resolves.toEqual({ alive: true })
    await vi.advanceTimersByTimeAsync(70_000)
    expect(client.connectionState).toBe('open')
    expect(sockets).toHaveLength(2)
    expect(sockets[0].frames.some(frame => frame.method === 'file.attach')).toBe(false)
    const attach = sockets[1].frames.find(frame => frame.method === 'file.attach')!
    expect(attach).toBeDefined()
    sockets[1].receive({ id: attach.id, result: { attached: true } })
    await expect(upload).resolves.toEqual({ attached: true })
    expect(sockets[1].readyState).toBe(WebSocket.CLOSED)
    client.close()
  })

  it('bounds an unanswered upload and never retries an ambiguous write', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
    const { client, sockets } = await setup()
    const upload = client.request('image.attach_bytes', { content_base64: 'AA==' })
    const failed = expect(upload).rejects.toThrow('request timed out after 600s: image.attach_bytes')
    await vi.advanceTimersByTimeAsync(600_001)
    await failed
    expect(sockets).toHaveLength(2)
    expect(sockets[1].frames.filter(frame => frame.method === 'image.attach_bytes')).toHaveLength(1)
    expect(sockets[1].readyState).toBe(WebSocket.CLOSED)
    client.close()
  })

  it('closing the owner cancels the upload without sending it again on another connection', async () => {
    const { client, sockets } = await setup()
    const upload = client.request('file.attach', { data_url: 'AA==' })
    const failed = expect(upload).rejects.toThrow('WebSocket closed')
    await Promise.resolve()
    client.close()
    await failed
    expect(sockets).toHaveLength(2)
    expect(sockets[1].readyState).toBe(WebSocket.CLOSED)
  })
})
