import { spawn } from 'node:child_process'
import crypto from 'node:crypto'

import { hiddenWindowsChildOptions, windowsShellCommand } from './windows-child-options'

// `hermes gateway ensure` answers with a protocol verdict (ready, or `starting: deadline`) within
// its own startup deadline: `--timeout` defaults to 60 s (hermes_cli/gateway_runtime.py
// DEFAULT_ENSURE_TIMEOUT). The client is killed only after that deadline plus interpreter start
// and verdict output, so a slow-but-valid cold start is never reported as "produced no result".
const GATEWAY_ENSURE_PROTOCOL_DEADLINE_MS = 60_000
const GATEWAY_ENSURE_CLIENT_GRACE_MS = 15_000

/** Run one bounded `hermes ... gateway <verb>` client (`ensure` by default; `restart` for a stale owner). */
export function runGatewayEnsure(
  backend, cwd: string, home: string, parentEnv: NodeJS.ProcessEnv = process.env,
  { timeoutMs = GATEWAY_ENSURE_PROTOCOL_DEADLINE_MS + GATEWAY_ENSURE_CLIENT_GRACE_MS, label = 'hermes gateway ensure' }: { timeoutMs?: number; label?: string } = {}
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    // shell: true hands cmd.exe one command line it cuts at the first unquoted space,
    // so an install under `C:\Users\John Doe\...` must arrive quoted (#74064).
    const child = spawn(windowsShellCommand(backend.command, Boolean(backend.shell)), backend.args, hiddenWindowsChildOptions({ cwd, env: { ...parentEnv, HERMES_HOME: home, ...backend.env }, shell: backend.shell, stdio: ['ignore', 'pipe', 'pipe'] }))
    let stdout = ''
    let stderr = ''
    let expired: Error | undefined
    let forceTimer: ReturnType<typeof setTimeout> | undefined

    const expire = (message: string) => {
      if (expired) { return }
      expired = new Error(message)
      child.kill()
      forceTimer = setTimeout(() => { child.kill('SIGKILL'); reject(expired) }, 1000)
    }

    // Only the bounded CLI client is ours. Never retain/kill its detached owner.
    const timer = setTimeout(() => expire(`${label} timed out`), timeoutMs)
    child.stdout.on('data', data => { stdout += data.toString();

 if (stdout.length > 65536) { stdout = stdout.slice(0, 65536); expire(`${label} output exceeded limit`) } })
    // Diagnostics only (never protocol): an older `hermes` without the subcommand, a missing
    // profile or an import crash explain themselves here while stdout stays empty.
    child.stderr.on('data', data => { stderr = (stderr + data.toString()).slice(-4096) })
    child.on('error', () => { clearTimeout(timer); clearTimeout(forceTimer); reject(new Error(`Could not run ${label}`)) })
    // A client that exits on SIGTERM still resolves (callers such as restartLocalGatewayOwner log a
    // non-zero exit and re-ensure); only one that outlives the force-kill deadline rejects. An
    // expired run never reads as success, and its last diagnostic line names the deadline.
    child.on('close', code => {
      clearTimeout(timer)
      clearTimeout(forceTimer)

      if (expired) { resolve({ code: code || 7, stdout, stderr: `${stderr}\n${expired.message}`.trim() }) }
      else { resolve({ code: code ?? 7, stdout, stderr }) }
    })
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
  /** Multiplexer home whose control socket answers for a served secondary; null when the profile owns its own. */
  control_home?: string | null
  /** Commit the owner booted from; null when it cannot name one (see `client_code_sha`). */
  code_sha?: string | null
}

/** Restarts a ready owner whose code differs from this client's; true when a restart was requested. */
export type StaleGatewayRestart = (endpoint: GatewayEndpoint, stale: { gateway: string; client: string }) => Promise<boolean>

/** The profile whose `hermes --profile P gateway restart` replaces the process behind `endpoint`:
 *  the multiplexer root (`default`) for a served secondary, else the endpoint's own profile home. */
export function gatewayOwnerProfile(endpoint: GatewayEndpoint): string {
  const home = endpoint.control_home || endpoint.profile_id

  return path.basename(path.dirname(home)) === 'profiles' ? path.basename(home) : 'default'
}

/** One restart per stale (gateway, client) code pair per process. An owner a restart cannot move
 *  (a service unit pinned to another checkout) is then attached as-is instead of being restarted
 *  on every dial. */
export function createStaleGatewayRestarter(
  restart: (ownerProfile: string) => Promise<void>,
  log: (message: string) => void
): StaleGatewayRestart {
  // Concurrent dials (primary + a served profile) that see the same stale owner share ONE restart;
  // a dial arriving after it finished attaches as-is.
  const restarts = new Map<string, { done: boolean; promise: Promise<void> }>()

  return async (endpoint, stale) => {
    const key = `${stale.gateway}\n${stale.client}`.toLowerCase()
    const pending = restarts.get(key)

    if (pending && !pending.done) {
      await pending.promise

      return true
    }

    if (pending) {
      log(`[gateway] owner still runs ${stale.gateway} (this Hermes is ${stale.client}) after a restart; attaching as-is`)

      return false
    }

    log(`[gateway] owner runs ${stale.gateway} but this Hermes is ${stale.client}; restarting it`)
    const entry = { done: false, promise: restart(gatewayOwnerProfile(endpoint)) }
    restarts.set(key, entry)

    try {
      await entry.promise
    } finally {
      entry.done = true
    }

    return true
  }
}

/** The single-line JSON `hermes gateway ensure --json` prints, or a diagnosis of why there is none.
 *
 * Every protocol outcome (ready/starting/incompatible/...) is JSON on stdout. Empty or
 * non-JSON stdout means the command never reached the protocol boundary: an older `hermes`
 * on PATH that has no `ensure` subcommand, a profile that does not exist, an interpreter
 * crash. A bare JSON.parse there surfaced as "Unexpected end of JSON input" with the real
 * reason discarded on stderr. */
export function parseGatewayEnsureOutput(result: { code: number; stdout: string; stderr?: string }): Record<string, any> {
  try {
    const payload = JSON.parse(result.stdout)

    if (payload && typeof payload === 'object' && !Array.isArray(payload)) {return payload}
  } catch { /* diagnosed below */ }

  const reason = String(result.stderr ?? '').trim().split(/\r?\n/).filter(Boolean).pop()

  throw new Error(`hermes gateway ensure produced no result (exit ${result.code})${reason ? `: ${reason}` : ''}. Update Hermes or check the profile, then retry.`)
}

/** `{ gateway, client }` commits when a ready owner provably runs other code than the ensure
 *  client (a gateway that outlived `hermes update` answers model calls 503 "Restart required");
 *  null when they match or either side cannot name its code. */
export function staleGatewayCode(payload: Record<string, any>): { gateway: string; client: string } | null {
  const gateway = payload?.endpoint?.code_sha
  const client = payload?.client_code_sha

  if (typeof gateway !== 'string' || !gateway || typeof client !== 'string' || !client) {return null}

  return gateway.toLowerCase() === client.toLowerCase() ? null : { gateway, client }
}

/** Loopback-only, bare-origin, session-authority endpoint or throw. */
function assertLocalGatewayEndpoint(endpoint: GatewayEndpoint): void {
  const origin = new URL(endpoint.api_origin)

  if (!['http:', 'https:'].includes(origin.protocol) || !['127.0.0.1', '[::1]'].includes(origin.hostname)
      || origin.username || origin.password || origin.pathname !== '/' || origin.search || origin.hash
      || !origin.port || endpoint.runtime_protocol !== 1 || !endpoint.instance_id || !endpoint.profile_id
      || !endpoint.capabilities.includes('session-authority-v1')) {
    throw new Error('Invalid local gateway endpoint')
  }
}

export async function ensureLocalGateway(
  run: () => Promise<{ code: number; stdout: string; stderr?: string }>,
  beforeEnsure?: () => Promise<void>,
  restartStale?: StaleGatewayRestart
) {
  await beforeEnsure?.()
  let result = await run()
  let payload = parseGatewayEnsureOutput(result)
  const stale = restartStale && result.code === 0 && payload.state === 'ready' ? staleGatewayCode(payload) : null

  // Attaching would pin a stale owner (503 "Restart required" forever): replace it once, then
  // take whatever answers the re-ensure.
  if (stale && await restartStale!(payload.endpoint, stale)) {
    result = await run()
    payload = parseGatewayEnsureOutput(result)
  }

  if (result.code !== 0 || payload.state !== 'ready') {
    throw new Error(`Gateway ${payload.state || 'inaccessible'} (${payload.reason_code || 'ensure_failed'}). Use hermes gateway status for recovery.`)
  }

  const endpoint: GatewayEndpoint = payload.endpoint
  assertLocalGatewayEndpoint(endpoint)

  return { baseUrl: endpoint.api_origin, wsUrl: `${endpoint.api_origin.replace(/^http/, 'ws')}/api/ws?native_dial=unminted`, mode: 'local', source: 'local', authMode: 'native', token: '', gatewayEndpoint: endpoint }
}

// The cached connection descriptor pins the gateway instance that answered the
// first `gateway ensure`. When that owner is gone (crash, `gateway stop`,
// update restart) every ticket mint against the stale control socket fails and
// the renderer's reconnect backoff would loop on the dead descriptor forever.
// Forget the cached descriptor exactly once and re-run the canonical ensure;
// a second failure is a real error and surfaces to the caller. A pointer mid-rewrite
// or left by an earlier owner, and a profile home moved since caching, may recover.
// Unsafe path permissions or types require repair, not another ensure of the same home.
const STALE_CONTROL_ERRORS = new Set(['Invalid gateway ticket response', 'Invalid gateway control pointer', 'Noncanonical gateway profile'])

export function isStaleLocalGatewayError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return message.startsWith('Gateway ticket') || STALE_CONTROL_ERRORS.has(message)
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

function dialNonce(url: string): string | null {
  try {
    const parsed = new URL(url)

    return parsed.pathname === '/api/ws' ? parsed.searchParams.get('native_dial') : null
  } catch {
    return null
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
      // Keyed by the one-use nonce, not the full URL: a loopback proxy the app dials
      // through (tests, tunnels) rewrites host:port but carries the query through.
      const nonce = crypto.randomUUID()
      const url = `${origin.replace(/^http/, 'ws')}/api/ws?native_dial=${nonce}`
      pending.set(nonce, { ticket, webContentsId, expires: Date.now() + 25_000 })

      return `${url}&ticket=${encodeURIComponent(ticket)}`
    },
    headers(details: { url: string; webContentsId?: number; resourceType: string; requestHeaders?: Record<string, string> }) {
      const nonce = dialNonce(details.url)
      const grant = nonce ? pending.get(nonce) : undefined

      if (!nonce || !grant || grant.webContentsId !== details.webContentsId || details.resourceType !== 'webSocket') {return null}
      pending.delete(nonce)

      if (grant.expires <= Date.now()) {return null}
      const headers = { ...details.requestHeaders }

      for (const key of Object.keys(headers)) {if (key.toLowerCase() === 'origin') {delete headers[key]}}

      return headers
    }
  }
}

// A profile home keeps the operator's mode (Python `home_mode_unsafe`: HERMES_HOME_MODE 0750/0701,
// a 0755 home): read/search bits grant nothing against the 0600 socket; only write by another user
// could swap it. Group-write is safe only for a proven-private group without an ACL, which Node
// cannot establish (no xattrs/NSS): a group-writable home returns true so the caller hands the
// ticket to the Python bridge, which applies that exact policy. The socket, pointer and fallback
// directory stay owner-only.
async function privateNode(file: string, kind: 'directory' | 'socket' | 'file' | 'home'): Promise<boolean> {
  const node = await fs.lstat(file)
  const valid = { directory: node.isDirectory(), home: node.isDirectory(), socket: node.isSocket(), file: node.isFile() }[kind]
  const uid = process.getuid?.()
  const quoted = shellQuote(file)

  // Not stale (a re-ensure of the same home would hit the same refusal): name the repair.
  if (!valid) {
    const found = node.isSymbolicLink() ? 'a symlink' : node.isDirectory() ? 'a directory' : node.isFile() ? 'a regular file' : node.isSocket() ? 'a socket' : 'another file type'

    // The socket, pointer and fallback directory are the gateway's to recreate; a home is not.
    throw new Error(kind === 'home'
      ? `Unsafe gateway control path: profile home ${quoted} is ${found}, not a directory: move it aside and recreate the profile`
      : `Unsafe gateway control path: ${quoted} should be a ${kind} but is ${found}: remove it, then run hermes gateway restart`)
  }

  if (node.uid !== uid) {
    throw new Error(`Unsafe gateway control path: ${quoted} is owned by uid ${node.uid}, not this user (uid ${uid}): run sudo chown ${uid} ${quoted}`)
  }

  if (kind === 'home' && node.mode & 0o002) {
    throw new Error(`Unsafe gateway control path: profile home ${quoted} is writable by other users (mode ${(node.mode & 0o777).toString(8)}): run chmod o-w ${quoted}`)
  }

  if (kind !== 'home' && node.mode & 0o077) {
    throw new Error(`Unsafe gateway control path: ${quoted} is accessible to other users (mode ${(node.mode & 0o777).toString(8)}): run chmod go-rwx ${quoted}`)
  }

  return kind === 'home' && Boolean(node.mode & 0o020)
}

function shellQuote(file: string): string {
  return `'${file.replace(/'/g, `'\\''`)}'`
}

/** The endpoint a `?profile=<name>` request is scoped to on a shared host descriptor: the same
 *  daemon, ticket minted for the sibling profile's home (`profiles/<name>` under the launch root).
 *  The gateway refuses a ticket whose profile it does not serve, so a bad name fails closed. */
export function routedGatewayEndpoint(endpoint: GatewayEndpoint, urlOrProfile: string, launchHome: string): GatewayEndpoint {
  const profile = (/^[a-z]+:\/\//.test(urlOrProfile) ? new URL(urlOrProfile).searchParams.get('profile') : urlOrProfile)?.trim()

  if (!profile || profile === 'current') {return endpoint}
  const own = path.basename(endpoint.profile_id)
  const ownName = path.basename(path.dirname(endpoint.profile_id)) === 'profiles' ? own : 'default'

  if (profile === ownName) {return endpoint}

  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(profile)) {throw new Error('Invalid profile route')}
  const root = path.basename(path.dirname(endpoint.profile_id)) === 'profiles' ? path.dirname(path.dirname(endpoint.profile_id)) : launchHome
  const home = profile === 'default' ? root : path.join(root, 'profiles', profile)

  return { ...endpoint, profile_id: home, control_home: endpoint.control_home || endpoint.profile_id }
}

export async function nativeGatewayHttpHeaders(descriptor: { gatewayEndpoint: GatewayEndpoint; baseUrl: string }, url: string, launchHome = descriptor.gatewayEndpoint.profile_id) {
  if (new URL(url).origin !== descriptor.gatewayEndpoint.api_origin || descriptor.baseUrl !== descriptor.gatewayEndpoint.api_origin) {
    throw new Error('Native gateway HTTP origin mismatch')
  }

  return { 'X-Hermes-Gateway-Ticket': await mintLocalGatewayTicket(routedGatewayEndpoint(descriptor.gatewayEndpoint, url, launchHome), 'native-http') }
}

// The installed runtime's own ticket client: every Windows mint (SID-validated pipe), and a POSIX
// home whose group-write only Python can prove private.
let pythonTicketClient: ((endpoint: GatewayEndpoint, purpose: 'interactive' | 'native-http') => Promise<string>) | undefined

export function configurePythonGatewayTicketClient(client: NonNullable<typeof pythonTicketClient>) {
  pythonTicketClient = client
}

async function mintGroupWritableHome(endpoint: GatewayEndpoint, purpose: 'interactive' | 'native-http', homes: string[]) {
  if (!pythonTicketClient) {throw new Error(`Unsafe gateway control path: profile home is group-writable and this Desktop cannot verify the group is private: run chmod g-w ${homes.map(shellQuote).join(' ')}`)}

  try {
    return await pythonTicketClient(endpoint, purpose)
  } catch (error) {
    if ((error as { reason?: string })?.reason !== 'unsafe_control_permissions') {throw error}
    const quoted = homes.map(shellQuote).join(' ')

    throw new Error(`Profile home is writable by a group other accounts share: run chmod g-w ${quoted}`)
  }
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

/** The Desktop's interactive socket is the one shared-primary route: its renderer selects every
 *  served sibling with a `profile` param, so it asks for a host-scoped grant explicitly. Every
 *  other ticket (and every other client's) stays bound to the one profile it names. */
export function ticketScope(purpose: 'interactive' | 'native-http'): { scope?: 'host' } {
  return purpose === 'interactive' ? { scope: 'host' } : {}
}

export async function mintLocalGatewayTicket(endpoint: GatewayEndpoint, purpose: 'interactive' | 'native-http' = 'interactive'): Promise<string> {
  if (process.platform === 'win32') {
    if (!pythonTicketClient) {throw new Error('Gateway ticket client is not configured')}

    return pythonTicketClient(endpoint, purpose)
  }

  const home = endpoint.profile_id
  // A served secondary has no socket of its own: the default multiplexer's control socket
  // mints its tickets (bound to the secondary's profile_id). Same rule as
  // hermes_cli.gateway_runtime.control_home_for.
  const controlHome = endpoint.control_home || home

  const groupWritable: string[] = []

  for (const dir of new Set([home, controlHome])) {
    if (await fs.realpath(dir) !== dir) {throw new Error('Noncanonical gateway profile')}

    if (await privateNode(dir, 'home')) {groupWritable.push(dir)}
  }

  if (groupWritable.length) {return mintGroupWritableHome(endpoint, purpose, groupWritable)}

  let socketPath: string

  try { socketPath = await resolveControlSocket(controlHome) } catch (error) {
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
    socket.on('connect', () => socket.write(JSON.stringify({ protocol: 1, id: 1, verb: 'session-ticket', params: { profile_id: home, instance_id: endpoint.instance_id, purpose, ...ticketScope(purpose) } }) + '\n'))
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
