import { spawn } from 'node:child_process'
import crypto from 'node:crypto'

import { hiddenWindowsChildOptions } from './windows-child-options'

export function runGatewayEnsure(backend, cwd: string, home: string): Promise<{ code: number; stdout: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(backend.command, backend.args, hiddenWindowsChildOptions({ cwd, env: { ...process.env, HERMES_HOME: home, ...backend.env }, shell: backend.shell, stdio: ['ignore', 'pipe', 'pipe'] }))
    let stdout = ''
    // Only the bounded ensure client is ours. Never retain/kill its detached owner.
    const timer = setTimeout(() => child.kill(), 40_000)
    child.stdout.on('data', data => { stdout += data.toString();

 if (stdout.length > 65536) {child.kill()} })
    child.stderr.resume()
    child.on('error', () => { clearTimeout(timer); reject(new Error('Could not run hermes gateway ensure')) })
    child.on('close', code => { clearTimeout(timer); resolve({ code: code ?? 7, stdout }) })
  })
}

import fs from 'node:fs/promises'
import net from 'node:net'
import path from 'node:path'

export interface GatewayEndpoint {
  profile_id: string
  instance_id: string
  authority_epoch: number
  runtime_protocol: number
  api_origin: string
  capabilities: string[]
  supervisor: string
}

export async function ensureLocalGateway(run: () => Promise<{ code: number; stdout: string }>, beforeEnsure?: () => Promise<void>) {
  await beforeEnsure?.()
  const result = await run()
  const payload = JSON.parse(result.stdout)

  if (result.code !== 0 || payload.state !== 'ready') {
    throw new Error(`Gateway ${payload.state || 'inaccessible'} (${payload.reason_code || 'ensure_failed'}). Use hermes gateway status for recovery.`)
  }

  const endpoint: GatewayEndpoint = payload.endpoint
  const origin = new URL(endpoint.api_origin)

  if (!['http:', 'https:'].includes(origin.protocol) || !['127.0.0.1', '[::1]'].includes(origin.hostname)
      || origin.username || origin.password || origin.pathname !== '/' || origin.search || origin.hash
      || !origin.port || endpoint.runtime_protocol !== 1 || !endpoint.instance_id || !endpoint.profile_id
      || !endpoint.capabilities.includes('session-authority-v1')) {
    throw new Error('Invalid local gateway endpoint')
  }

  return { baseUrl: endpoint.api_origin, wsUrl: `${endpoint.api_origin.replace(/^http/, 'ws')}/api/ws?native_dial=unminted`, mode: 'local', source: 'local', authMode: 'native', token: '', gatewayEndpoint: endpoint }
}

// The cached connection descriptor pins the gateway instance that answered the
// first `gateway ensure`. When that owner is gone (crash, `gateway stop`,
// update restart) every ticket mint against the stale control socket fails and
// the renderer's reconnect backoff would loop on the dead descriptor forever.
// Forget the cached descriptor exactly once and re-run the canonical ensure;
// a second failure is a real error and surfaces to the caller.
export function isStaleLocalGatewayError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return message.startsWith('Gateway ticket') || message === 'Invalid gateway ticket response' || message === 'Unsafe gateway control path'
}

export async function redialLocalGateway<TEndpoint, TResult>(deps: {
  ensure: () => Promise<TEndpoint>
  forget: () => Promise<void> | void
  use: (endpoint: TEndpoint) => Promise<TResult>
}): Promise<TResult> {
  const first = await deps.ensure()

  try {
    return await deps.use(first)
  } catch (error) {
    if (!isStaleLocalGatewayError(error)) {throw error}
    await deps.forget()

    return deps.use(await deps.ensure())
  }
}

// A one-use ticket crosses private IPC, then leaves the URL before dialing.
// Public descriptors and persisted connection state never contain credentials.
export function createLocalGatewayDials() {
  const pending = new Map<string, { ticket: string; webContentsId: number; expires: number }>()

  return {
    prepare(origin: string, ticket: string, webContentsId: number) {
      for (const [key, value] of pending) {if (value.expires <= Date.now()) {pending.delete(key)}}

      if (pending.size >= 100) {throw new Error('Too many pending gateway dials')}
      const url = `${origin.replace(/^http/, 'ws')}/api/ws?native_dial=${crypto.randomUUID()}`
      pending.set(url, { ticket, webContentsId, expires: Date.now() + 25_000 })

      return `${url}&ticket=${encodeURIComponent(ticket)}`
    },
    headers(details: { url: string; webContentsId?: number; resourceType: string; requestHeaders?: Record<string, string> }) {
      const grant = pending.get(details.url)

      if (!grant || grant.webContentsId !== details.webContentsId || details.resourceType !== 'webSocket') {return null}
      pending.delete(details.url)

      if (grant.expires <= Date.now()) {return null}
      const headers = { ...details.requestHeaders }

      for (const key of Object.keys(headers)) {if (key.toLowerCase() === 'origin') {delete headers[key]}}

      return headers
    }
  }
}

async function privateNode(file: string, kind: 'directory' | 'socket' | 'file') {
  const node = await fs.lstat(file)
  const valid = { directory: node.isDirectory(), socket: node.isSocket(), file: node.isFile() }[kind]

  if (!valid || node.uid !== process.getuid?.() || (node.mode & 0o077)) {throw new Error('Unsafe gateway control path')}
}

export async function nativeGatewayHttpHeaders(descriptor: { gatewayEndpoint: GatewayEndpoint; baseUrl: string }, url: string) {
  if (new URL(url).origin !== descriptor.gatewayEndpoint.api_origin || descriptor.baseUrl !== descriptor.gatewayEndpoint.api_origin) {
    throw new Error('Native gateway HTTP origin mismatch')
  }

  return { 'X-Hermes-Gateway-Ticket': await mintLocalGatewayTicket(descriptor.gatewayEndpoint, 'native-http') }
}

let windowsTicketClient: ((endpoint: GatewayEndpoint, purpose: 'interactive' | 'native-http') => Promise<string>) | undefined

export function configureWindowsGatewayTicketClient(client: NonNullable<typeof windowsTicketClient>) {
  windowsTicketClient = client
}

function isMissingNodeError(error: unknown): boolean {
  const code = (error as NodeJS.ErrnoException)?.code

  return code === 'ENOENT' || code === 'ENOTDIR'
}

async function resolveControlSocket(home: string): Promise<string> {
  const socketPath = path.join(home, 'gateway.sock')

  try { await privateNode(socketPath, 'socket') } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {throw error}
    const pointer = path.join(home, 'gateway.sock.path')
    await privateNode(pointer, 'file')
    const target = (await fs.readFile(pointer, 'utf8')).trim()
    const hash = crypto.createHash('sha256').update(home).digest('hex').slice(0, 16)

    if (!path.isAbsolute(target) || path.basename(path.dirname(target)) !== `hermes-gw-${hash}` || path.basename(target) !== 'control.sock') {throw new Error('Invalid gateway control pointer')}
    await privateNode(path.dirname(target), 'directory')
    await privateNode(target, 'socket')

    return target
  }

  return socketPath
}

export async function mintLocalGatewayTicket(endpoint: GatewayEndpoint, purpose: 'interactive' | 'native-http' = 'interactive'): Promise<string> {
  if (process.platform === 'win32') {
    if (!windowsTicketClient) {throw new Error('Gateway ticket client is not configured')}

    return windowsTicketClient(endpoint, purpose)
  }

  const home = endpoint.profile_id

  if (await fs.realpath(home) !== home) {throw new Error('Noncanonical gateway profile')}
  await privateNode(home, 'directory')
  let socketPath: string

  try { socketPath = await resolveControlSocket(home) } catch (error) {
    // `gateway stop` unlinks both the socket and its pointer: the owner is gone,
    // which the redial must treat as stale rather than as a raw filesystem error.
    if (isMissingNodeError(error)) {throw new Error('Gateway ticket control socket missing')}
    throw error
  }

  return new Promise((resolve, reject) => {
    const socket = net.createConnection(socketPath)
    let buffer = ''
    const deadline = setTimeout(() => socket.destroy(new Error('Gateway ticket deadline')), 5000)
    socket.on('error', () => reject(new Error('Gateway ticket bootstrap failed')))
    socket.on('close', () => { clearTimeout(deadline); reject(new Error('Gateway ticket connection closed')) })
    socket.on('connect', () => socket.write(JSON.stringify({ protocol: 1, id: 1, verb: 'session-ticket', params: { profile_id: home, instance_id: endpoint.instance_id, purpose } }) + '\n'))
    socket.on('data', chunk => {
      buffer += chunk.toString()

      if (buffer.length > 65536) {return socket.destroy(new Error('Oversized gateway ticket response'))}

      if (!buffer.includes('\n')) {return}

      try {
        const reply = JSON.parse(buffer.split('\n')[0])
        const grant = reply.result

        if (!reply.ok || reply.protocol !== 1 || reply.id !== 1 || grant?.instance_id !== endpoint.instance_id || grant?.profile_id !== home || typeof grant?.ticket !== 'string') {throw new Error('Invalid gateway ticket')}
        resolve(grant.ticket)
      } catch { reject(new Error('Invalid gateway ticket response')) }

      socket.destroy()
    })
  })
}
