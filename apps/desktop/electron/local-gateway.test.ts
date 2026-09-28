import { expect, test } from 'vitest'

import { createLocalGatewayDials, ensureLocalGateway, routedGatewayEndpoint, runGatewayEnsure } from './local-gateway'

test('the ensure client inherits the caller-scrubbed parent env, not the raw Desktop env', async () => {
  // #68367: a sibling profile's `gateway ensure` must not see the launch profile's dotenv
  // credentials. The parent env passed in IS the environment; only HERMES_HOME and the
  // backend's own entries are layered on top.
  const printEnv = ['-e', 'process.stdout.write(JSON.stringify({ leak: process.env.LEAK ?? null, home: process.env.HERMES_HOME, own: process.env.OWN }))']
  const result = await runGatewayEnsure(
    { command: process.execPath, args: printEnv, env: { OWN: '1' }, shell: false },
    process.cwd(),
    '/home/x/.hermes',
    { PATH: process.env.PATH ?? '', OWN: '0' }
  )
  expect(JSON.parse(result.stdout)).toEqual({ leak: null, home: '/home/x/.hermes', own: '1' })
})

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
  // macOS: os.tmpdir() is /var/..., a symlink to /private/var; the gateway canonicalises
  // profile_id, so the endpoint must carry the realpath or identities never match.
  const home = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'desktop-http-')))
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

test.skipIf(process.platform === 'win32')('a served secondary mints its ticket through the multiplexer control socket, bound to its own profile', async () => {
  const fs = await import('node:fs/promises')
  const os = await import('node:os')
  const path = await import('node:path')
  const net = await import('node:net')
  const { mintLocalGatewayTicket } = await import('./local-gateway')
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'desktop-mux-')))
  const secondary = path.join(root, 'profiles', 'cold')
  await fs.mkdir(secondary, { recursive: true, mode: 0o700 })
  await fs.chmod(root, 0o700)
  // Only the multiplexer root has a control socket; profiles/cold has none (it is served).
  const requests: any[] = []

  const server = net.createServer(socket => socket.once('data', chunk => {
    requests.push(JSON.parse(chunk.toString()).params)
    socket.end(JSON.stringify({ protocol: 1, id: 1, ok: true, result: { profile_id: secondary, instance_id: 'mux', ticket: 'served-grant' } }) + '\n')
  }))

  await new Promise<void>(resolve => server.listen(path.join(root, 'gateway.sock'), resolve))
  await fs.chmod(path.join(root, 'gateway.sock'), 0o600)
  const endpoint = { profile_id: secondary, instance_id: 'mux', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' }

  try {
    // Without control_home the secondary's own (absent) socket is probed: stale owner, not a grant.
    await expect(mintLocalGatewayTicket(endpoint, 'native-http')).rejects.toThrow('Gateway ticket control socket missing')
    expect(await mintLocalGatewayTicket({ ...endpoint, control_home: root }, 'native-http')).toBe('served-grant')
    expect(requests).toEqual([{ profile_id: secondary, instance_id: 'mux', purpose: 'native-http' }])
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
    await fs.rm(root, { recursive: true, force: true })
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

test('an ensure child that never reached the protocol boundary is diagnosed from stderr, not as a JSON parse error', async () => {
  // `main`'s hermes has no `gateway ensure` subcommand: argparse usage on stderr, nothing on stdout.
  const legacy = { code: 2, stdout: '', stderr: "usage: hermes gateway [-h] ...\nhermes gateway: 'ensure' is not a `hermes gateway` command.\nRun `hermes gateway --help` to see all commands.\n" }
  await expect(ensureLocalGateway(async () => legacy)).rejects.toThrow(/produced no result \(exit 2\): Run `hermes gateway --help`/)
  // A missing profile: the CLI exits 1 before the ensure command runs.
  await expect(ensureLocalGateway(async () => ({ code: 1, stdout: '', stderr: "Error: Profile 'gone' does not exist.\n" }))).rejects.toThrow(/exit 1\): Error: Profile 'gone' does not exist\./)
  // A protocol outcome is never re-diagnosed: `incompatible` stays the gateway's own verdict.
  await expect(ensureLocalGateway(async () => ({ code: 3, stdout: '{"endpoint":null,"reason_code":"runtime_protocol","state":"incompatible"}', stderr: 'noise' }))).rejects.toThrow('Gateway incompatible (runtime_protocol)')
  await expect(ensureLocalGateway(async () => ({ code: 0, stdout: '', stderr: '' }))).rejects.toThrow(/produced no result \(exit 0\)\. Update Hermes/)
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
  // The dial may cross a loopback proxy that rewrites host:port but keeps the nonce.
  const proxied = url.replace('127.0.0.1:1234', '127.0.0.1:4321')
  const headers = dials.headers({ ...details, url: proxied, webContentsId: 7 })!
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
  // macOS: os.tmpdir() is /var/..., a symlink to /private/var; the gateway canonicalises
  // profile_id, so the endpoint must carry the realpath or identities never match.
  const home = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'desktop-redial-')))
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
  // macOS: os.tmpdir() is /var/..., a symlink to /private/var; the gateway canonicalises
  // profile_id, so the endpoint must carry the realpath or identities never match.
  const home = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'desktop-unsafe-')))
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

test('a ?profile= request on the shared host descriptor mints for the sibling profile home', () => {
  const endpoint = { profile_id: '/h/.hermes', instance_id: 'i', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1', capabilities: [], supervisor: 'none', control_home: null }
  expect(routedGatewayEndpoint(endpoint, 'http://127.0.0.1:1/api/sessions?profile=p2', '/h/.hermes')).toMatchObject({ profile_id: '/h/.hermes/profiles/p2', control_home: '/h/.hermes' })
  expect(routedGatewayEndpoint(endpoint, 'http://127.0.0.1:1/api/sessions?profile=default', '/h/.hermes')).toBe(endpoint)
  expect(routedGatewayEndpoint({ ...endpoint, profile_id: '/h/.hermes/profiles/p2', control_home: '/h/.hermes' }, 'http://127.0.0.1:1/x?profile=default', '/h/.hermes')).toMatchObject({ profile_id: '/h/.hermes' })
  expect(() => routedGatewayEndpoint(endpoint, 'http://127.0.0.1:1/x?profile=../evil', '/h/.hermes')).toThrow('Invalid profile route')
})
