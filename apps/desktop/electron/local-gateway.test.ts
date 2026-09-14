import { expect, test } from 'vitest'

import { createLocalGatewayDials, ensureLocalGateway } from './local-gateway'

test('canonical ensure cannot cross a rejected update or profile lifecycle gate', async () => {
  let ran = false
  await expect(ensureLocalGateway(async () => { ran = true;

 return { code: 0, stdout: '{}' } }, async () => {
    throw new Error('profile retired during update')
  })).rejects.toThrow('profile retired during update')
  expect(ran).toBe(false)
})

test.skipIf(process.platform === 'win32')('native HTTP mints fresh purpose-bound grants without browser credentials', async () => {
  const fs = await import('node:fs/promises')
  const os = await import('node:os')
  const path = await import('node:path')
  const net = await import('node:net')
  const { nativeGatewayHttpHeaders } = await import('./local-gateway')
  const home = await fs.mkdtemp(path.join(os.tmpdir(), 'desktop-http-'))
  const endpoint = { profile_id: home, instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' }
  const requests: any[] = []

  const server = net.createServer(socket => socket.once('data', chunk => {
    const request = JSON.parse(chunk.toString())
    requests.push(request)
    socket.end(JSON.stringify({ protocol: 1, id: 1, ok: true, result: { profile_id: home, instance_id: 'owner', ticket: `grant-${requests.length}` } }) + '\n')
  }))

  await new Promise<void>(resolve => server.listen(path.join(home, 'gateway.sock'), resolve))
  await fs.chmod(path.join(home, 'gateway.sock'), 0o600)

  try {
    const descriptor = { gatewayEndpoint: endpoint, baseUrl: endpoint.api_origin }
    const a = await nativeGatewayHttpHeaders(descriptor, endpoint.api_origin + '/api/config')
    const b = await nativeGatewayHttpHeaders(descriptor, endpoint.api_origin + '/api/config')
    expect(a).toEqual({ 'X-Hermes-Gateway-Ticket': 'grant-1' })
    expect(b).toEqual({ 'X-Hermes-Gateway-Ticket': 'grant-2' })
    expect(requests.map(r => r.params)).toEqual([1, 2].map(() => ({ profile_id: home, instance_id: 'owner', purpose: 'native-http' })))
    await expect(nativeGatewayHttpHeaders(descriptor, 'http://127.0.0.1:4567/api/config')).rejects.toThrow('origin')
    expect(requests).toHaveLength(2)
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
    await fs.rm(home, { recursive: true, force: true })
  }
})

test('ensure consumes structured readiness without acquiring a child owner', async () => {
  const endpoint = { profile_id: '/private/profile', instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' }
  const connection = await ensureLocalGateway(async () => ({ code: 0, stdout: JSON.stringify({ state: 'ready', endpoint }) }))
  expect(connection.baseUrl).toBe(endpoint.api_origin)
  expect(connection.gatewayEndpoint).toEqual(endpoint)
  expect(connection).not.toHaveProperty('process')
  expect(connection.token).toBe('')
  await expect(ensureLocalGateway(async () => ({ code: 5, stdout: JSON.stringify({ state: 'starting', reason_code: 'deadline' }) }))).rejects.toThrow('starting')
})

test('private dial credential is one-use and bound to the requesting native window', () => {
  const dials = createLocalGatewayDials()
  const dial = new URL(dials.prepare('http://127.0.0.1:1234', 'private-ticket', 7))
  expect(dial.searchParams.get('ticket')).toBe('private-ticket')
  dial.searchParams.delete('ticket')
  const url = dial.toString()
  expect(url).not.toContain('private-ticket')
  const details = { url, webContentsId: 8, resourceType: 'webSocket', requestHeaders: { Origin: 'http://renderer', 'Sec-WebSocket-Protocol': 'hermes-gateway-v1, hermes-gateway-ticket.private-ticket' } }
  expect(dials.headers(details)).toBeNull()
  const headers = dials.headers({ ...details, webContentsId: 7 })!
  expect(headers).not.toHaveProperty('Origin')
  expect(headers['Sec-WebSocket-Protocol']).toBe('hermes-gateway-v1, hermes-gateway-ticket.private-ticket')
  expect(dials.headers({ ...details, webContentsId: 7 })).toBeNull()
})

test('a dial against a replaced local gateway forgets the cached endpoint once and re-ensures', async () => {
  const { redialLocalGateway } = await import('./local-gateway')
  const endpoints = [{ instance_id: 'dead' }, { instance_id: 'alive' }]
  const forgotten: string[] = []
  let ensures = 0

  const result = await redialLocalGateway({
    ensure: async () => endpoints[Math.min(ensures++, 1)],
    forget: async () => { forgotten.push('primary') },
    use: async endpoint => {
      if (endpoint.instance_id === 'dead') {throw new Error('Gateway ticket bootstrap failed')}

      return `ticket-for-${endpoint.instance_id}`
    }
  })

  expect(result).toBe('ticket-for-alive')
  expect(forgotten).toEqual(['primary'])
  expect(ensures).toBe(2)
})

test('a dial that keeps failing after one re-ensure surfaces the error instead of looping', async () => {
  const { redialLocalGateway } = await import('./local-gateway')
  let ensures = 0
  let forgets = 0
  await expect(redialLocalGateway({
    ensure: async () => { ensures++;

 return { instance_id: 'still-dead' } },
    forget: async () => { forgets++ },
    use: async () => { throw new Error('Invalid gateway ticket response') }
  })).rejects.toThrow('Invalid gateway ticket response')
  expect(ensures).toBe(2)
  expect(forgets).toBe(1)
})

test.skipIf(process.platform === 'win32')('a stopped gateway that unlinked its control socket is a stale owner, not a raw filesystem error', async () => {
  const fs = await import('node:fs/promises')
  const os = await import('node:os')
  const path = await import('node:path')
  const net = await import('node:net')
  const { mintLocalGatewayTicket, redialLocalGateway } = await import('./local-gateway')
  const home = await fs.mkdtemp(path.join(os.tmpdir(), 'desktop-redial-'))
  const socketPath = path.join(home, 'gateway.sock')
  const endpoint = { profile_id: home, instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' }

  const serve = async (ticket: string) => {
    const server = net.createServer(socket => socket.once('data', () => {
      socket.end(JSON.stringify({ protocol: 1, id: 1, ok: true, result: { profile_id: home, instance_id: 'owner', ticket } }) + '\n')
    }))

    await new Promise<void>(resolve => server.listen(socketPath, resolve))
    await fs.chmod(socketPath, 0o600)

    return server
  }

  const stop = async (server: ReturnType<typeof net.createServer>) => {
    await new Promise<void>(resolve => server.close(() => resolve()))
    await fs.rm(socketPath, { force: true })
  }

  let replacement: ReturnType<typeof net.createServer> | null = null
  const original = await serve('grant-original')

  try {
    expect(await mintLocalGatewayTicket(endpoint)).toBe('grant-original')
    await stop(original)

    let ensures = 0
    let forgets = 0

    const result = await redialLocalGateway({
      ensure: async () => {
        ensures += 1

        if (ensures === 2) {replacement = await serve('grant-replacement')}

        return endpoint
      },
      forget: () => { forgets += 1 },
      use: e => mintLocalGatewayTicket(e)
    })

    expect(result).toBe('grant-replacement')
    expect(forgets).toBe(1)
    expect(ensures).toBe(2)
  } finally {
    if (replacement) {await stop(replacement)}
    await fs.rm(home, { recursive: true, force: true })
  }
})

test.skipIf(process.platform === 'win32')('a group-accessible control socket is refused as unsafe, and unrelated errors are never stale', async () => {
  const fs = await import('node:fs/promises')
  const os = await import('node:os')
  const path = await import('node:path')
  const net = await import('node:net')
  const { isStaleLocalGatewayError, mintLocalGatewayTicket } = await import('./local-gateway')
  const home = await fs.mkdtemp(path.join(os.tmpdir(), 'desktop-unsafe-'))
  const socketPath = path.join(home, 'gateway.sock')
  const endpoint = { profile_id: home, instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' }
  const server = net.createServer(socket => socket.end())
  await new Promise<void>(resolve => server.listen(socketPath, resolve))
  await fs.chmod(socketPath, 0o660)

  try {
    const failure = await mintLocalGatewayTicket(endpoint).catch(error => error)
    expect(failure).toBeInstanceOf(Error)
    expect(failure.message).toBe('Unsafe gateway control path')
    expect(isStaleLocalGatewayError(failure)).toBe(true)
    expect(isStaleLocalGatewayError(new Error('EACCES: permission denied'))).toBe(false)
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
    await fs.rm(home, { recursive: true, force: true })
  }
})
