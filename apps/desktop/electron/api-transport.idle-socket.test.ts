/**
 * A proxy or NAT between the desktop and a remote gateway can drop an idle
 * keep-alive connection without closing it. The JSON pool must retire such a
 * socket once it has sat idle, so the next GET opens a fresh connection instead
 * of stalling until its request timeout.
 */
import http from 'node:http'
import net from 'node:net'

import { afterAll, describe, expect, it } from 'vitest'

import { destroyKeepaliveAgents, downloadAgentFor, jsonAgentFor, withRetry } from './api-transport'

type AgentWithOptions = http.Agent & { options: { timeout?: number } }

const JSON_IDLE_TIMEOUT_MS = 15_000
const BODY = JSON.stringify({ ok: true })

const sockets = new Set<net.Socket>()
const servers: net.Server[] = []

function track(socket: net.Socket) {
  sockets.add(socket)
  socket.once('close', () => sockets.delete(socket))

  return socket
}

async function listen(server: net.Server) {
  servers.push(server)
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  return (server.address() as net.AddressInfo).port
}

afterAll(async () => {
  destroyKeepaliveAgents()

  for (const socket of sockets) {
    socket.destroy()
  }

  await Promise.all(
    servers.map(server => new Promise<void>(resolve => (server.listening ? server.close(() => resolve()) : resolve())))
  )
})

// Same request shape as main.ts fetchJson: pooled agent, req.setTimeout -> destroy, withRetry by verb.
function getJson(url: string, timeoutMs: number, onSocket?: (socket: net.Socket) => void): Promise<unknown> {
  return withRetry(
    (state: { bodySent: boolean }) =>
      new Promise((resolve, reject) => {
        const req = http.request(new URL(url), { agent: jsonAgentFor('http:'), method: 'GET' }, res => {
          const chunks: Buffer[] = []
          res.on('data', chunk => chunks.push(chunk))
          res.on('end', () => resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))))
          res.on('error', reject)
        })

        if (onSocket) {
          req.once('socket', onSocket)
        }

        req.on('error', reject)
        req.setTimeout(timeoutMs, () => req.destroy(new Error(`Timed out after ${timeoutMs}ms`)))
        state.bodySent = true
        req.end()
      }),
    { method: 'GET', delayFn: () => Promise.resolve() }
  )
}

// Resolves when the socket closes, or after deadlineMs without failing the test.
function closedWithin(socket: net.Socket, deadlineMs: number): Promise<boolean> {
  if (socket.destroyed) {
    return Promise.resolve(true)
  }

  return new Promise(resolve => {
    const timer = setTimeout(() => resolve(false), deadlineMs)

    socket.once('close', () => {
      clearTimeout(timer)
      resolve(true)
    })
  })
}

// Forwards each client connection's first complete HTTP response (headers and
// Content-Length body, however TCP splits them), then silently drops all
// further traffic in both directions without closing either side.
async function startBlackholingProxy(backendPort: number) {
  let connections = 0

  const proxy = net.createServer(client => {
    connections += 1
    track(client)
    const upstream = track(net.connect(backendPort, '127.0.0.1'))
    let received = Buffer.alloc(0)
    let responseEnd = Infinity
    let done = false

    upstream.on('data', data => {
      if (done) {
        return
      }

      received = Buffer.concat([received, data])
      const headerEnd = received.indexOf('\r\n\r\n')

      if (responseEnd === Infinity && headerEnd !== -1) {
        const length = /\r\ncontent-length:\s*(\d+)/i.exec(received.subarray(0, headerEnd).toString('latin1'))
        responseEnd = headerEnd + 4 + Number(length?.[1] ?? 0)
      }

      const forward = data.subarray(0, Math.max(0, data.length - Math.max(0, received.length - responseEnd)))
      client.write(forward)
      done = received.length >= responseEnd
    })
    client.on('data', data => {
      if (!done) {
        upstream.write(data)
      }
    })
    client.on('error', () => undefined)
    upstream.on('error', () => undefined)
  })

  const port = await listen(proxy)

  return { connections: () => connections, url: `http://127.0.0.1:${port}/api/status` }
}

describe('idle pooled JSON sockets', () => {
  it('retires idle JSON sockets after 15 s and leaves the download pool unbounded', () => {
    expect((jsonAgentFor('http:') as AgentWithOptions).options.timeout).toBe(JSON_IDLE_TIMEOUT_MS)
    expect((jsonAgentFor('https:') as AgentWithOptions).options.timeout).toBe(JSON_IDLE_TIMEOUT_MS)
    expect((downloadAgentFor('http:') as AgentWithOptions).options.timeout).toBeUndefined()
    expect((downloadAgentFor('https:') as AgentWithOptions).options.timeout).toBeUndefined()
  })

  it('replaces a silently dropped socket once it sat idle in the pool', async () => {
    // Headers and body leave in separate writes so the proxy cannot rely on one TCP chunk.
    const backend = http.createServer((_req, res) => {
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(BODY) })
      res.flushHeaders()
      setTimeout(() => res.end(BODY), 20)
    })

    backend.on('connection', track)
    const proxy = await startBlackholingProxy(await listen(backend))

    // Scale the configured idle timeout down so the test does not wait 15 s; the
    // mechanism is the same. Without a configured timeout nothing is overridden.
    const agent = jsonAgentFor('http:') as AgentWithOptions
    const configured = agent.options.timeout

    if (configured) {
      agent.options.timeout = 200
    }

    try {
      let pooled: net.Socket | undefined

      expect(await getJson(proxy.url, 1_000, socket => (pooled = socket))).toEqual({ ok: true })
      expect(pooled).toBeDefined()
      await closedWithin(pooled!, 2_000)

      expect(await getJson(proxy.url, 1_000)).toEqual({ ok: true })
      expect(proxy.connections()).toBe(2)
    } finally {
      agent.options.timeout = configured
    }
  }, 10_000)
})
