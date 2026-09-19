/**
 * Tests for electron/env-proxy.ts — the HTTP(S)_PROXY support the desktop's
 * GitHub API calls need (#114736).
 *
 * Node's http/https do not read the proxy environment the way curl, git and
 * browsers do, so the passive update check dialed api.github.com directly and
 * timed out for every user behind a corporate proxy whose proxy variables were
 * set correctly. These pin the two load-bearing contracts:
 *
 *  - with no proxy configured the request is untouched — no agent, Node's
 *    default direct dial (the pre-fix behaviour, kept exactly);
 *  - with one configured the request goes through a CONNECT tunnel to that
 *    proxy, proved against a real local CONNECT proxy — the proxy sees the
 *    CONNECT and the target sees the request, so nothing can go direct behind
 *    the operator's back.
 *
 * The TCP tunnels here are loopback-only; no test reaches the network.
 */
import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import http from 'node:http'
import https from 'node:https'
import net from 'node:net'
import type { AddressInfo } from 'node:net'

import { test } from 'vitest'

import { proxyAgentFor, proxyUrlFor } from './env-proxy'

const BRANCH_TIP = 'https://api.github.com/repos/nousresearch/hermes-agent/commits/main'
const CORPORATE = 'http://proxy.corp.example:8080'

test('no proxy environment: no agent at all, exactly the pre-fix request', () => {
  for (const env of [{}, { HTTP_PROXY: '' }, { HTTPS_PROXY: '   ' }, { NO_PROXY: 'api.github.com' }]) {
    assert.equal(proxyAgentFor(BRANCH_TIP, env), undefined)
  }
})

test('the proxy environment names the proxy in either casing, with or without a scheme', () => {
  assert.equal(proxyUrlFor(BRANCH_TIP, { HTTPS_PROXY: CORPORATE }), `${CORPORATE}/`)
  assert.equal(proxyUrlFor(BRANCH_TIP, { https_proxy: 'proxy.corp.example:8080' }), `${CORPORATE}/`)
  // HTTP_PROXY is the fallback for an https target, and the first set
  // variable wins (HTTPS_PROXY > HTTP_PROXY > ALL_PROXY, as on the Python side).
  assert.equal(proxyUrlFor(BRANCH_TIP, { HTTP_PROXY: 'http://fallback:3128' }), 'http://fallback:3128/')
  assert.equal(
    proxyUrlFor(BRANCH_TIP, { HTTP_PROXY: 'http://fallback:3128', ALL_PROXY: 'http://all:1080' }),
    'http://fallback:3128/'
  )
  // Credentials stay in the URL; they become Proxy-Authorization on the wire.
  assert.equal(
    proxyUrlFor(BRANCH_TIP, { HTTPS_PROXY: 'http://user:p%40ss@proxy.corp.example:8080' }),
    'http://user:p%40ss@proxy.corp.example:8080/'
  )
  // An ordinary env does not conjure a proxy out of nothing.
  assert.equal(proxyUrlFor('git@github.com:someone/hermes-agent.git', { HTTPS_PROXY: CORPORATE }), null)
  assert.equal(proxyUrlFor(BRANCH_TIP, { HTTPS_PROXY: 'not a url' }), null)
})

test('NO_PROXY keeps hosts out of the proxy; everything else goes through it', () => {
  const env = (NO_PROXY: string) => ({ HTTPS_PROXY: CORPORATE, NO_PROXY })

  // Exact host, parent domain, dot / wildcard suffixes, a port-scoped entry
  // and the operator's blanket `*`.
  for (const entry of [
    'api.github.com',
    'github.com',
    '.github.com',
    '*.github.com',
    'api.github.com:443',
    'other.example, api.github.com',
    '*'
  ]) {
    assert.equal(proxyUrlFor(BRANCH_TIP, env(entry)), null, `NO_PROXY=${entry} must bypass the proxy`)
  }

  // A different host, a different port, or a domain that merely ends with the
  // same letters must NOT bypass it.
  for (const entry of ['example.com', 'notgithub.com', 'api.github.com:8443', '']) {
    assert.equal(proxyUrlFor(BRANCH_TIP, env(entry)), `${CORPORATE}/`, `NO_PROXY=${entry} must not bypass the proxy`)
  }
})

// ---------------------------------------------------------------------------
// LIVE: a real CONNECT proxy on loopback.
// ---------------------------------------------------------------------------

interface LocalProxy {
  url: string
  /** CONNECT authorities the proxy was asked for, in order. */
  connects: string[]
  close: () => Promise<void>
}

async function startConnectProxy(handler?: (clientSocket: net.Socket, target: string) => boolean): Promise<LocalProxy> {
  const connects: string[] = []

  const server = http.createServer((_req, res) => {
    res.statusCode = 405
    res.end('CONNECT only')
  })

  server.on('connect', (req, clientSocket: net.Socket, head) => {
    const target = String(req.url)
    connects.push(target)

    if (handler && handler(clientSocket, target)) {
      return
    }

    const [host, port] = target.split(':')

    const upstream = net.connect({ host, port: Number(port) }, () => {
      clientSocket.write('HTTP/1.1 200 Connection established\r\n\r\n')

      if (head?.length) {
        upstream.write(head)
      }

      upstream.pipe(clientSocket)
      clientSocket.pipe(upstream)
    })

    upstream.once('error', () => clientSocket.destroy())
    clientSocket.once('error', () => upstream.destroy())
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  return {
    url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
    connects,
    close: () =>
      new Promise<void>(resolve => {
        server.closeAllConnections?.()
        server.close(() => resolve())
      })
  }
}

async function startTarget(body: string) {
  let hits = 0

  const server = http.createServer((_req, res) => {
    hits += 1
    res.setHeader('content-type', 'application/json')
    res.end(body)
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const port = (server.address() as AddressInfo).port

  return {
    port,
    hits: () => hits,
    close: () =>
      new Promise<void>(resolve => {
        server.closeAllConnections?.()
        server.close(() => resolve())
      })
  }
}

/** GET `url` with `agent`, resolving the body — the shape fetchGitHubApiOnce uses. */
function get(url: string, agent?: https.Agent): Promise<string> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url)
    const client = parsed.protocol === 'https:' ? https : http

    const req = client.get(parsed, { agent, timeout: 10_000 } as any, res => {
      const chunks: Buffer[] = []
      res.on('data', chunk => chunks.push(chunk))
      res.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    })

    req.once('error', reject)
  })
}

/**
 * Accept the tunnel and keep it open — the TLS layer is what the tests drive.
 * The socket is handed back so the test can close the tunnel it opened (a live
 * tunnel is not something `server.close()` waits out).
 */
function acceptAndHold(held: net.Socket[]): (clientSocket: net.Socket) => boolean {
  return clientSocket => {
    held.push(clientSocket)
    clientSocket.on('error', () => {})
    clientSocket.write('HTTP/1.1 200 Connection established\r\n\r\n')

    return true
  }
}

/**
 * Run one connection through the agent with a scripted TLS layer and collect
 * every callback the agent's createConnection delivers — the point being that
 * there is exactly one per connection, whatever the tunnel does afterwards.
 */
async function collectConnectionCalls(
  proxyUrl: string,
  onTunnel: (tunnel: EventEmitter, onSecure: () => void) => void
): Promise<Array<Error | null>> {
  const tunnel = new EventEmitter()
  const calls: Array<Error | null> = []

  // A listenerless emitter throws on 'error'; this keeps the hostile shapes
  // below about what the agent forwards, not about Node's own crash rule.
  tunnel.on('error', () => {})

  const agent = proxyAgentFor(BRANCH_TIP, { HTTPS_PROXY: proxyUrl }, {
    tlsConnect: (_options: any, onSecure: () => void) => {
      setTimeout(() => onTunnel(tunnel, onSecure), 0)

      return tunnel
    }
  })

  await new Promise<void>(resolve => {
    agent!.createConnection({ host: 'api.github.com', port: 443, servername: 'api.github.com' } as any, (error: Error | null) => {
      calls.push(error)
      resolve()
    })
  })

  return calls
}

test('live: with no proxy configured the request dials the target directly and no proxy is touched', async () => {
  const target = await startTarget('{"sha":"direct"}')
  const proxy = await startConnectProxy()

  try {
    const agent = proxyAgentFor(`http://127.0.0.1:${target.port}/`, {})

    assert.equal(agent, undefined, 'nothing configured must mean no agent (Node default agent = direct)')
    assert.equal(await get(`http://127.0.0.1:${target.port}/`, agent), '{"sha":"direct"}')
    assert.deepEqual(proxy.connects, [])
    assert.equal(target.hits(), 1)
  } finally {
    await target.close()
    await proxy.close()
  }
})

test('live: a configured proxy carries the request through CONNECT instead of direct', async () => {
  const target = await startTarget('{"sha":"through-the-proxy"}')
  const proxy = await startConnectProxy()

  try {
    // The tunnel would carry a TLS handshake; the local target answers plain
    // HTTP, so the handshake is stubbed out and the tunnel itself is what is
    // being proved — no key pair, no network.
    const passthrough = (options: any, onSecure: () => void) => {
      const socket = options.socket
      setTimeout(onSecure, 0)

      return socket
    }

    const url = `https://127.0.0.1:${target.port}/repos/nousresearch/hermes-agent/commits/main`
    const agent = proxyAgentFor(url, { HTTPS_PROXY: proxy.url }, { tlsConnect: passthrough })

    assert.ok(agent, 'HTTPS_PROXY set must produce an agent')
    assert.equal(agent.proxyUrl, `${proxy.url}/`)

    assert.equal(await get(url, agent), '{"sha":"through-the-proxy"}')
    assert.deepEqual(proxy.connects, [`127.0.0.1:${target.port}`])
    assert.equal(target.hits(), 1)
  } finally {
    await target.close()
    await proxy.close()
  }
})

test('a tunnel error during the handshake is reported once, never twice', async () => {
  const held: net.Socket[] = []
  const proxy = await startConnectProxy(acceptAndHold(held))

  try {
    const calls = await collectConnectionCalls(proxy.url, tunnel => {
      tunnel.emit('error', new Error('ECONNRESET'))
      tunnel.emit('error', new Error('late error after the reset'))
    })

    assert.equal(calls.length, 1, 'a second error on the dead tunnel must not re-enter the callback')
    assert.match(calls[0]!.message, /ECONNRESET/)
  } finally {
    for (const socket of held) {
      socket.destroy()
    }

    await proxy.close()
  }
})

test('a tunnel error after the handshake is not reported as a second connection', async () => {
  const held: net.Socket[] = []
  const proxy = await startConnectProxy(acceptAndHold(held))

  try {
    const calls = await collectConnectionCalls(proxy.url, (tunnel, onSecure) => {
      onSecure()
      // The socket is live from here on, so the request owns its errors: the
      // agent's callback already delivered the socket and must not see this one.
      tunnel.emit('error', new Error('late error after the handshake'))
    })

    assert.equal(calls.length, 1, 'the request owns socket errors once the handshake is done')
    assert.equal(calls[0], null)
  } finally {
    for (const socket of held) {
      socket.destroy()
    }

    await proxy.close()
  }
})

test('live: a proxy that refuses CONNECT fails the check instead of bypassing it', async () => {
  const target = await startTarget('{"sha":"must-not-be-reached"}')

  const proxy = await startConnectProxy(clientSocket => {
    clientSocket.end('HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')

    return true
  })

  try {
    const url = `https://127.0.0.1:${target.port}/repos/nousresearch/hermes-agent/commits/main`
    const agent = proxyAgentFor(url, { HTTPS_PROXY: proxy.url })

    await assert.rejects(get(url, agent), /refused CONNECT|instead of tunnelling/)
    // Traffic must not escape the proxy: the target never saw a direct dial.
    assert.equal(target.hits(), 0)
  } finally {
    await target.close()
    await proxy.close()
  }
})
