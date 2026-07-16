import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { once } from 'node:events'
import http from 'node:http'
import net from 'node:net'
import { promisify } from 'node:util'

import { test, vi } from 'vitest'

import { createLinkTitlePinnedResolver } from './link-title-dns'
import { createLinkTitleSocksGatewayController, startLinkTitleSocksGateway } from './link-title-socks'

const execFileAsync = promisify(execFile)

async function connectToGateway(proxyUrl: string): Promise<net.Socket> {
  const url = new URL(proxyUrl)
  const socket = net.createConnection({ host: url.hostname, port: Number(url.port) })
  await once(socket, 'connect')

  return socket
}

async function readAtLeast(socket: net.Socket, byteCount: number): Promise<Buffer> {
  const chunks: Buffer[] = []
  let bytes = 0

  while (bytes < byteCount) {
    const [chunk] = (await once(socket, 'data')) as [Buffer]
    chunks.push(chunk)
    bytes += chunk.length
  }

  return Buffer.concat(chunks)
}

async function negotiateNoAuth(socket: net.Socket): Promise<void> {
  socket.write(Buffer.from([0x05, 0x01, 0x00]))
  assert.deepEqual(await readAtLeast(socket, 2), Buffer.from([0x05, 0x00]))
}

function domainRequest(command: number, hostname: string, port = 80): Buffer {
  const domain = Buffer.from(hostname, 'ascii')

  return Buffer.from([0x05, command, 0x00, 0x03, domain.length, ...domain, port >> 8, port & 0xff])
}

test('gateway binds to loopback and rejects unsupported SOCKS commands before resolving', async () => {
  const resolve = vi.fn(async () => [{ address: '93.184.216.34', family: 4 as const }])
  const createConnection = vi.fn()

  const gateway = await startLinkTitleSocksGateway({
    connectTimeoutMs: 250,
    createConnection: createConnection as unknown as typeof net.createConnection,
    resolve
  })

  try {
    const url = new URL(gateway.proxyUrl)
    assert.equal(url.protocol, 'socks5:')
    assert.equal(url.hostname, '127.0.0.1')
    assert.ok(Number(url.port) > 0)

    const socket = await connectToGateway(gateway.proxyUrl)
    await negotiateNoAuth(socket)
    socket.write(domainRequest(0x02, 'example.com'))

    const reply = await readAtLeast(socket, 10)
    assert.equal(reply[0], 0x05)
    assert.equal(reply[1], 0x07)
    assert.equal(resolve.mock.calls.length, 0)
    assert.equal(createConnection.mock.calls.length, 0)
    socket.destroy()
  } finally {
    await gateway.close()
  }
})

test('gateway rejects unknown address types and oversized handshakes', async () => {
  const resolve = vi.fn(async () => [{ address: '93.184.216.34', family: 4 as const }])
  const gateway = await startLinkTitleSocksGateway({ connectTimeoutMs: 250, resolve })

  try {
    const unknown = await connectToGateway(gateway.proxyUrl)
    await negotiateNoAuth(unknown)
    unknown.write(Buffer.from([0x05, 0x01, 0x00, 0x09, 0x00, 0x50]))
    const reply = await readAtLeast(unknown, 10)
    assert.equal(reply[1], 0x08)
    unknown.destroy()

    const oversized = await connectToGateway(gateway.proxyUrl)
    oversized.write(Buffer.concat([Buffer.from([0x05, 0xff]), Buffer.alloc(1_100)]))
    await once(oversized, 'close')
    assert.equal(resolve.mock.calls.length, 0)
  } finally {
    await gateway.close()
  }
})

test('gateway returns the SOCKS greeting rejection shape when no supported auth method is offered', async () => {
  const resolve = vi.fn(async () => [{ address: '93.184.216.34', family: 4 as const }])
  const gateway = await startLinkTitleSocksGateway({ connectTimeoutMs: 250, resolve })

  try {
    const socket = await connectToGateway(gateway.proxyUrl)
    socket.write(Buffer.from([0x05, 0x01, 0x02]))
    assert.deepEqual(await readAtLeast(socket, 2), Buffer.from([0x05, 0xff]))
    assert.equal(resolve.mock.calls.length, 0)
    socket.destroy()
  } finally {
    await gateway.close()
  }
})

test('resolver rejection never reaches the upstream connector', async () => {
  const resolve = vi.fn(async () => {
    throw new Error('mixed or private DNS answer')
  })

  const createConnection = vi.fn()

  const gateway = await startLinkTitleSocksGateway({
    connectTimeoutMs: 250,
    createConnection: createConnection as unknown as typeof net.createConnection,
    resolve
  })

  try {
    const socket = await connectToGateway(gateway.proxyUrl)
    await negotiateNoAuth(socket)
    socket.write(domainRequest(0x01, 'mixed.example'))
    const reply = await readAtLeast(socket, 10)

    assert.equal(reply[1], 0x04)
    assert.deepEqual(resolve.mock.calls, [['mixed.example']])
    assert.equal(createConnection.mock.calls.length, 0)
    socket.destroy()
  } finally {
    await gateway.close()
  }
})

test('a timed-out client cannot start an upstream connection after slow DNS eventually resolves', async () => {
  let finishResolve: ((value: readonly { address: string; family: 4 }[]) => void) | undefined

  const pendingResolve = new Promise<readonly { address: string; family: 4 }[]>(resolve => {
    finishResolve = resolve
  })

  const createConnection = vi.fn()

  const gateway = await startLinkTitleSocksGateway({
    connectTimeoutMs: 40,
    createConnection: createConnection as unknown as typeof net.createConnection,
    resolve: () => pendingResolve
  })

  try {
    const socket = await connectToGateway(gateway.proxyUrl)
    await negotiateNoAuth(socket)
    socket.write(domainRequest(0x01, 'slow.example'))
    await once(socket, 'close')

    finishResolve?.([{ address: '93.184.216.34', family: 4 }])
    await new Promise(resolve => setTimeout(resolve, 20))

    assert.equal(createConnection.mock.calls.length, 0)
  } finally {
    await gateway.close()
  }
})

test('gateway close destroys an upstream socket that is still connecting', async () => {
  const stalled = new net.Socket()
  const createConnection = vi.fn(() => stalled)

  const gateway = await startLinkTitleSocksGateway({
    connectTimeoutMs: 5_000,
    createConnection: createConnection as unknown as typeof net.createConnection,
    resolve: async () => [{ address: '93.184.216.34', family: 4 }]
  })

  const client = await connectToGateway(gateway.proxyUrl)
  await negotiateNoAuth(client)
  client.write(domainRequest(0x01, 'pending.example'))

  for (let attempt = 0; attempt < 20 && createConnection.mock.calls.length === 0; attempt += 1) {
    await new Promise(resolve => setTimeout(resolve, 5))
  }

  assert.equal(createConnection.mock.calls.length, 1)
  await gateway.close()
  assert.equal(stalled.destroyed, true)
  client.destroy()
})

test('all approved addresses share one absolute connection deadline', async () => {
  const sockets: net.Socket[] = []

  const createConnection = vi.fn(() => {
    const socket = new net.Socket()
    sockets.push(socket)

    return socket
  })

  const gateway = await startLinkTitleSocksGateway({
    connectTimeoutMs: 40,
    createConnection: createConnection as unknown as typeof net.createConnection,
    resolve: async () => [
      { address: '93.184.216.34', family: 4 },
      { address: '2606:4700:4700::1111', family: 6 }
    ]
  })

  try {
    const client = await connectToGateway(gateway.proxyUrl)
    await negotiateNoAuth(client)
    client.write(domainRequest(0x01, 'deadline.example'))
    await once(client, 'close')

    assert.equal(createConnection.mock.calls.length, 1)
    assert.equal(sockets.every(socket => socket.destroyed), true)
  } finally {
    await gateway.close()
  }
})

test('real curl traffic preserves Host and reuses the pinned answer inside the TTL', async () => {
  let observedHost = ''
  let requests = 0

  const server = http.createServer((request, response) => {
    observedHost = request.headers.host ?? ''
    requests += 1
    response.end('<title>Pinned title</title>')
  })

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })
  const address = server.address()
  assert.ok(address && typeof address !== 'string')

  let lookupCount = 0

  const resolver = createLinkTitlePinnedResolver({
    isPublicAddress: value => value === '127.0.0.1',
    lookup: async () => {
      lookupCount += 1

      return lookupCount === 1
        ? [{ address: '127.0.0.1', family: 4 }]
        : [{ address: '10.0.0.1', family: 4 }]
    },
    now: () => 0,
    ttlMs: 30_000
  })

  const gateway = await startLinkTitleSocksGateway({ connectTimeoutMs: 1_000, resolve: resolver.resolve })
  const curlProxy = gateway.proxyUrl.replace(/^socks5:/, 'socks5h:')
  const target = `http://public-title.invalid:${address.port}/title`

  try {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const { stdout } = await execFileAsync(
        'curl',
        ['--disable', '--silent', '--show-error', '--proxy', curlProxy, '--noproxy', '', target],
        { encoding: 'utf8', timeout: 5_000 }
      )

      assert.equal(stdout, '<title>Pinned title</title>')
    }

    assert.equal(requests, 2)
    assert.equal(observedHost, `public-title.invalid:${address.port}`)
    assert.equal(lookupCount, 1)
  } finally {
    await gateway.close()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('gateway controller starts one lazy gateway and closes it with its DNS pins', async () => {
  const close = vi.fn(async () => undefined)
  const start = vi.fn(async () => ({ close, proxyUrl: 'socks5://127.0.0.1:48123' }))
  const clearPins = vi.fn()
  const controller = createLinkTitleSocksGatewayController({ clearPins, start })

  const [first, second] = await Promise.all([controller.get(), controller.get()])

  assert.equal(first, second)
  assert.equal(start.mock.calls.length, 1)

  await controller.close()
  assert.equal(close.mock.calls.length, 1)
  assert.equal(clearPins.mock.calls.length, 1)
})

test('gateway controller never returns a direct fallback and permits a later pinned retry', async () => {
  const gateway = { close: vi.fn(async () => undefined), proxyUrl: 'socks5://127.0.0.1:48123' }
  const start = vi.fn().mockRejectedValueOnce(new Error('bind failed')).mockResolvedValueOnce(gateway)
  const controller = createLinkTitleSocksGatewayController({ clearPins: vi.fn(), start })

  await assert.rejects(controller.get(), /bind failed/)
  assert.equal(await controller.get(), gateway)
  assert.equal(start.mock.calls.length, 2)
  await controller.close()
})

test('gateway controller close is terminal during an in-progress gateway shutdown', async () => {
  let finishClose: (() => void) | undefined

  const close = vi.fn(
    () =>
      new Promise<void>(resolve => {
        finishClose = resolve
      })
  )

  const start = vi.fn(async () => ({ close, proxyUrl: 'socks5://127.0.0.1:48123' }))
  const clearPins = vi.fn()
  const controller = createLinkTitleSocksGatewayController({ clearPins, start })

  await controller.get()
  const closing = controller.close()

  await assert.rejects(controller.get(), /closed/i)
  assert.equal(start.mock.calls.length, 1)

  finishClose?.()
  await closing
  await controller.close()

  assert.equal(close.mock.calls.length, 1)
  assert.equal(clearPins.mock.calls.length, 1)
  await assert.rejects(controller.get(), /closed/i)
})
