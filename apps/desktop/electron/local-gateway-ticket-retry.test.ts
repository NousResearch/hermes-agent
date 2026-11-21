import { expect, test } from 'vitest'

import { mintLocalGatewayTicket, nativeGatewayHttpHeaders } from './local-gateway'
import { shortSocketTmpDir } from './local-gateway.test-helpers'

// A stalled gateway main loop breaks the control socket before any verdict arrives. The mint layer
// re-dials those two transient errors alone (≤3 attempts, 500ms→1500ms backoff, ~10s budget);
// stale verdicts still surface on the first connection for redialLocalGateway's re-ensure.

test.skipIf(process.platform === 'win32')('a transient socket break is re-dialled: two breaks then a grant answer the one request', async () => {
  const fs = await import('node:fs/promises')
  const path = await import('node:path')
  const net = await import('node:net')
  const home = await shortSocketTmpDir('ticket-retry-')
  await fs.chmod(home, 0o700)
  const endpoint = { profile_id: home, instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' }
  const connections: number[] = []

  const server = net.createServer(socket => {
    connections.push(connections.length + 1)

    if (connections.length < 3) {
      // Consume the request so a clean end() can close: a paused server stream with unread
      // data never emits 'end' and holds server.close() open forever. First break destroys
      // (RST → client 'error' → bootstrap failed); second break ends (FIN → 'close' → connection closed).

      socket.resume()

      return connections.length === 1 ? socket.destroy() : socket.end()
    }

    socket.once('data', () => socket.end(JSON.stringify({ protocol: 1, id: 1, ok: true, result: { profile_id: home, instance_id: 'owner', runtime_protocol: 1, ticket: 'grant-after-breaks' } }) + '\n'))
  })

  await new Promise<void>(resolve => server.listen(path.join(home, 'gateway.sock'), resolve))
  await fs.chmod(path.join(home, 'gateway.sock'), 0o600)

  try {
    const startedAt = Date.now()
    const headers = await nativeGatewayHttpHeaders({ gatewayEndpoint: endpoint, baseUrl: endpoint.api_origin }, endpoint.api_origin + '/api/config')

    expect(headers).toEqual({ 'X-Hermes-Gateway-Ticket': 'grant-after-breaks' })
    expect(connections).toEqual([1, 2, 3])
    expect(Date.now() - startedAt).toBeGreaterThanOrEqual(2000) // both backoffs really elapsed
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
    await fs.rm(home, { recursive: true, force: true })
  }
})

test.skipIf(process.platform === 'win32')('three transient breaks exhaust the re-dial and the last error surfaces untouched', async () => {
  const fs = await import('node:fs/promises')
  const path = await import('node:path')
  const net = await import('node:net')
  const home = await shortSocketTmpDir('ticket-exhaust-')
  await fs.chmod(home, 0o700)
  const endpoint = { profile_id: home, instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' }
  let connections = 0

  const server = net.createServer(socket => {
    connections += 1
    socket.destroy() // request written but never read: every dial breaks transiently
  })

  await new Promise<void>(resolve => server.listen(path.join(home, 'gateway.sock'), resolve))
  await fs.chmod(path.join(home, 'gateway.sock'), 0o600)

  try {
    await expect(mintLocalGatewayTicket(endpoint)).rejects.toThrow(/Gateway ticket (bootstrap failed|connection closed)/)
    expect(connections).toBe(3)
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
    await fs.rm(home, { recursive: true, force: true })
  }
})

test.skipIf(process.platform === 'win32')('an invalid ticket verdict is not transient: one connection, no re-dial', async () => {
  const fs = await import('node:fs/promises')
  const path = await import('node:path')
  const net = await import('node:net')
  const home = await shortSocketTmpDir('ticket-stale-')
  await fs.chmod(home, 0o700)
  const endpoint = { profile_id: home, instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' }
  let connections = 0

  const server = net.createServer(socket => {
    connections += 1
    socket.once('data', () => socket.end('not a ticket reply\n'))
  })

  await new Promise<void>(resolve => server.listen(path.join(home, 'gateway.sock'), resolve))
  await fs.chmod(path.join(home, 'gateway.sock'), 0o600)

  try {
    await expect(mintLocalGatewayTicket(endpoint)).rejects.toThrow('Invalid gateway ticket response')
    expect(connections).toBe(1)
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
    await fs.rm(home, { recursive: true, force: true })
  }
})

test.skipIf(process.platform === 'win32')('a noncanonical home never reaches the socket, let alone a re-dial', async () => {
  const fs = await import('node:fs/promises')
  const home = await shortSocketTmpDir('ticket-noncanon-')
  const alias = `${home}-alias`

  await fs.symlink(home, alias)

  const endpoint = { profile_id: alias, instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' }

  try {
    await expect(mintLocalGatewayTicket(endpoint)).rejects.toThrow('Noncanonical gateway profile')
  } finally {
    await fs.rm(alias, { force: true })
    await fs.rm(home, { recursive: true, force: true })
  }
})
