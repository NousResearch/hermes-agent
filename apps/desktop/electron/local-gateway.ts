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
  backend,
  cwd: string,
  home: string,
  parentEnv: NodeJS.ProcessEnv = process.env,
  {
    timeoutMs = GATEWAY_ENSURE_PROTOCOL_DEADLINE_MS + GATEWAY_ENSURE_CLIENT_GRACE_MS,
    label = 'hermes gateway ensure'
  }: { timeoutMs?: number; label?: string } = {}
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    // shell: true hands cmd.exe one command line it cuts at the first unquoted space,
    // so an install under `C:\Users\John Doe\...` must arrive quoted (#74064).
    const child = spawn(
      windowsShellCommand(backend.command, Boolean(backend.shell)),
      backend.args,
      hiddenWindowsChildOptions({
        cwd,
        env: { ...parentEnv, HERMES_HOME: home, ...backend.env },
        shell: backend.shell,
        stdio: ['ignore', 'pipe', 'pipe']
      })
    )
    let stdout = ''
    let stderr = ''
    let expired: Error | undefined
    let forceTimer: ReturnType<typeof setTimeout> | undefined

    const expire = (message: string) => {
      if (expired) {
        return
      }
      expired = new Error(message)
      child.kill()
      forceTimer = setTimeout(() => {
        child.kill('SIGKILL')
        reject(expired)
      }, 1000)
    }

    // Only the bounded CLI client is ours. Never retain/kill its detached owner.
    const timer = setTimeout(() => expire(`${label} timed out`), timeoutMs)
    child.stdout.on('data', data => {
      stdout += data.toString()

      if (stdout.length > 65536) {
        stdout = stdout.slice(0, 65536)
        expire(`${label} output exceeded limit`)
      }
    })
    // Diagnostics only (never protocol): an older `hermes` without the subcommand, a missing
    // profile or an import crash explain themselves here while stdout stays empty.
    child.stderr.on('data', data => {
      stderr = (stderr + data.toString()).slice(-4096)
    })
    child.on('error', () => {
      clearTimeout(timer)
      clearTimeout(forceTimer)
      reject(new Error(`Could not run ${label}`))
    })
    // A client that exits on SIGTERM still resolves (callers such as restartLocalGatewayOwner log a
    // non-zero exit and re-ensure); only one that outlives the force-kill deadline rejects. An
    // expired run never reads as success, and its last diagnostic line names the deadline.
    child.on('close', code => {
      clearTimeout(timer)
      clearTimeout(forceTimer)

      if (expired) {
        resolve({ code: code || 7, stdout, stderr: `${stderr}\n${expired.message}`.trim() })
      } else {
        resolve({ code: code ?? 7, stdout, stderr })
      }
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

/** Requests a restart of a ready owner whose code differs from this client's. Resolves as soon as the
 *  restart is requested: `{ done }` settles when the restart client exits; null when none was requested.
 *  Never awaited to completion before re-discovery: a supervised restart waits for the old owner to
 *  drain, and a turn parked on approval/clarify there waits for an answer only this window can give. */
export type StaleGatewayRestart = (
  endpoint: GatewayEndpoint,
  stale: { gateway: string; client: string }
) => Promise<null | { done: Promise<void> }>

/** The profile whose `hermes --profile P gateway restart` replaces the process behind `endpoint`:
 *  the multiplexer root (`default`) for a served secondary, else the endpoint's own profile home. */
export function gatewayOwnerProfile(endpoint: GatewayEndpoint): string {
  const home = endpoint.control_home || endpoint.profile_id

  return path.basename(path.dirname(home)) === 'profiles' ? path.basename(home) : 'default'
}

/** One restart per (owner process, stale gateway, client) per Desktop process. An owner a restart cannot move
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
    const key = `${gatewayOwnerProfile(endpoint)}\n${stale.gateway}\n${stale.client}`.toLowerCase()
    const pending = restarts.get(key)

    if (pending && !pending.done) {
      return { done: pending.promise }
    }

    if (pending) {
      log(
        `[gateway] owner still runs ${stale.gateway} (this Hermes is ${stale.client}) after a restart; attaching as-is`
      )

      return null
    }

    log(`[gateway] owner runs ${stale.gateway} but this Hermes is ${stale.client}; restarting it`)
    const entry = { done: false, promise: restart(gatewayOwnerProfile(endpoint)).catch(() => undefined) }
    restarts.set(key, entry)
    void entry.promise.then(() => {
      entry.done = true
    })

    return { done: entry.promise }
  }
}

/** The single-line JSON `hermes gateway ensure --json` prints, or a diagnosis of why there is none.
 *
 * Every protocol outcome (ready/starting/incompatible/...) is JSON on stdout. Empty or
 * non-JSON stdout means the command never reached the protocol boundary: an older `hermes`
 * on PATH that has no `ensure` subcommand, a profile that does not exist, an interpreter
 * crash. A bare JSON.parse there surfaced as "Unexpected end of JSON input" with the real
 * reason discarded on stderr. */
export function parseGatewayEnsureOutput(result: {
  code: number
  stdout: string
  stderr?: string
}): Record<string, any> {
  try {
    const payload = JSON.parse(result.stdout)

    if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
      return payload
    }
  } catch {
    /* diagnosed below */
  }

  const reason = String(result.stderr ?? '')
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .pop()

  throw new Error(
    `hermes gateway ensure produced no result (exit ${result.code})${reason ? `: ${reason}` : ''}. Update Hermes or check the profile, then retry.`
  )
}

/** `{ gateway, client }` commits when a ready owner provably runs other code than the ensure
 *  client (a gateway that outlived `hermes update` answers model calls 503 "Restart required");
 *  null when they match or either side cannot name its code. */
export function staleGatewayCode(payload: Record<string, any>): { gateway: string; client: string } | null {
  const gateway = payload?.endpoint?.code_sha
  const client = payload?.client_code_sha

  if (typeof gateway !== 'string' || !gateway || typeof client !== 'string' || !client) {
    return null
  }

  return gateway.toLowerCase() === client.toLowerCase() ? null : { gateway, client }
}

/** Loopback-only, bare-origin, session-authority endpoint or throw. */
function assertLocalGatewayEndpoint(endpoint: GatewayEndpoint): void {
  const origin = new URL(endpoint.api_origin)

  if (
    !['http:', 'https:'].includes(origin.protocol) ||
    !['127.0.0.1', '[::1]'].includes(origin.hostname) ||
    origin.username ||
    origin.password ||
    origin.pathname !== '/' ||
    origin.search ||
    origin.hash ||
    !origin.port ||
    endpoint.runtime_protocol !== 1 ||
    !endpoint.instance_id ||
    !endpoint.profile_id ||
    !endpoint.capabilities.includes('session-authority-v1')
  ) {
    throw new Error('Invalid local gateway endpoint')
  }
}

/** Poll interval while a requested stale-owner restart is in flight. */
export const STALE_RESTART_POLL_MS = 1000

/** `gateway ensure`'s drain_reason for a restart's after-turn wait (gateway/drain_control.py RESTART_WAIT_REASON). */
const RESTART_WAIT_REASON = 'restart_wait'

/** A restart's after-turn wait that still hands out its endpoint: attachable, like Python's `attachable()`. */
function isRestartWait(payload: Record<string, any>): boolean {
  return payload.state === 'draining' && payload.reason_code === RESTART_WAIT_REASON && Boolean(payload.endpoint)
}

/** Re-ensure while a stale-owner restart runs, without waiting for it to finish (#139041): the first
 *  answer that is no longer the stale owner wins — its restart wait (attach to answer a parked
 *  prompt), or a replacement. Once the restart client exits, whatever answers next is final. */
async function reensureDuringRestart(
  run: () => Promise<{ code: number; stdout: string; stderr?: string }>,
  done: Promise<void>,
  staleInstance: string
): Promise<{ result: { code: number; stdout: string; stderr?: string }; payload: Record<string, any> }> {
  let finished = false
  void done.then(() => {
    finished = true
  })

  for (;;) {
    const last = finished
    const result = await run()
    const payload = parseGatewayEnsureOutput(result)
    const restartWait = isRestartWait(payload)
    const replaced = payload.state === 'ready' && payload.endpoint?.instance_id !== staleInstance

    if (last || restartWait || replaced) {
      return { result, payload }
    }
    let timer: ReturnType<typeof setTimeout> | undefined
    await Promise.race([
      done,
      new Promise(resolve => {
        timer = setTimeout(resolve, STALE_RESTART_POLL_MS)
      })
    ])
    clearTimeout(timer)
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
  const restart = stale ? await restartStale!(payload.endpoint, stale) : null

  if (restart) {
    ;({ result, payload } = await reensureDuringRestart(run, restart.done, payload.endpoint.instance_id))
  }

  // A graceful restart's after-turn wait still owns its turns (#139041): attach to it so a turn parked
  // on approval/clarify can be answered; new input queues for the next owner, and the next dial after
  // the owner exits reaches its replacement.
  const restartWait = isRestartWait(payload)

  if (!restartWait && (result.code !== 0 || payload.state !== 'ready')) {
    // A refused replacement (another install's or a newer Hermes's gateway) names its remedy in
    // `detail`; a start that died carries its own output there. Either beats a pointer elsewhere.
    const detail =
      typeof payload.detail === 'string' && payload.detail.trim() ? ` ${payload.detail.trim().split(/\r?\n/)[0]}.` : ''
    const recovery = payload.reason_code === 'update_paused' ? 'restart' : 'status'

    throw new Error(
      `Gateway ${payload.state || 'inaccessible'} (${payload.reason_code || 'ensure_failed'}).${detail} Use hermes gateway ${recovery} for recovery.`
    )
  }

  const endpoint: GatewayEndpoint = payload.endpoint
  assertLocalGatewayEndpoint(endpoint)

  return {
    baseUrl: endpoint.api_origin,
    wsUrl: `${endpoint.api_origin.replace(/^http/, 'ws')}/api/ws?native_dial=unminted`,
    mode: 'local',
    source: 'local',
    authMode: 'native',
    token: '',
    gatewayEndpoint: endpoint
  }
}

// The cached connection descriptor pins the gateway instance that answered the
// first `gateway ensure`. When that owner is gone (crash, `gateway stop`,
// update restart) every ticket mint against the stale control socket fails and
// the renderer's reconnect backoff would loop on the dead descriptor forever.
// Forget the cached descriptor exactly once and re-run the canonical ensure;
// a second failure is a real error and surfaces to the caller. A pointer mid-rewrite
// or left by an earlier owner, and a profile home moved since caching, may recover.
// Unsafe path permissions or types require repair, not another ensure of the same home.
const STALE_CONTROL_ERRORS = new Set([
  'Invalid gateway ticket response',
  'Invalid gateway control pointer',
  'Noncanonical gateway profile'
])

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
    if (!isStaleLocalGatewayError(error)) {
      throw error
    }
    await deps.forget()

    return deps.use(await deps.ensure())
  }
}

/** A fresh one-use local dial: mint `purpose` for the routed profile's home and hand it to
 *  `prepare`, re-ensuring once when the cached owner is gone (`redialLocalGateway`). */
export function localGatewayDial<T extends { baseUrl: string; gatewayEndpoint?: GatewayEndpoint }>(opts: {
  ensure: () => Promise<T>
  forget: () => Promise<void> | void
  launchHome: string
  mint?: typeof mintLocalGatewayTicket
  prepare: (baseUrl: string, ticket: string) => string
  profile: string
  purpose?: 'interactive' | 'native-http'
}): Promise<string> {
  const { mint = mintLocalGatewayTicket, purpose = 'interactive' } = opts

  return redialLocalGateway({
    ensure: opts.ensure,
    forget: opts.forget,
    // A shared-primary descriptor answers for a sibling profile too: the ticket must name
    // THAT profile's home or its sessions are another owner's.
    use: async current => {
      // Callers check the first descriptor; a re-ensure may still answer without one.
      if (!current.gatewayEndpoint) {
        throw new Error('Local gateway connection has no gateway endpoint')
      }

      return opts.prepare(
        current.baseUrl,
        await mint(await routedTicketEndpoint(current.gatewayEndpoint, opts.profile, opts.launchHome), purpose)
      )
    }
  })
}

function dialOf(url: string): { nonce: string; origin: string; pathname: string } | null {
  try {
    const parsed = new URL(url)
    const nonce = parsed.searchParams.get('native_dial')

    return nonce ? { nonce, origin: parsed.origin, pathname: parsed.pathname } : null
  } catch {
    return null
  }
}

// A one-use ticket crosses private IPC, then leaves the URL before dialing.
// Public descriptors and persisted connection state never contain credentials.
// A plugin socket's native-http grant never enters a URL at all: main attaches
// it as the header its REST twin sends, to that one dial only.
export function createLocalGatewayDials() {
  const pending = new Map<
    string,
    { ticket: string; webContentsId: number; expires: number; origin: string; pathname: string; header: boolean }
  >()

  const reserve = (url: URL, ticket: string, webContentsId: number, header: boolean) => {
    for (const [key, value] of pending) {
      if (value.expires <= Date.now()) {
        pending.delete(key)
      }
    }

    if (pending.size >= 100) {
      throw new Error('Too many pending gateway dials')
    }
    // Keyed by the one-use nonce, not the full URL: a loopback proxy the app dials
    // through (tests, tunnels) rewrites host:port but carries the query through.
    // A header grant is the exception: main attaches it, so it is bound to its origin.
    const nonce = crypto.randomUUID()
    url.searchParams.set('native_dial', nonce)
    pending.set(nonce, {
      ticket,
      webContentsId,
      expires: Date.now() + 25_000,
      origin: url.origin,
      pathname: url.pathname,
      header
    })

    return url.toString()
  }

  return {
    prepare(origin: string, ticket: string, webContentsId: number) {
      return `${reserve(new URL(`${origin.replace(/^http/, 'ws')}/api/ws`), ticket, webContentsId, false)}&ticket=${encodeURIComponent(ticket)}`
    },
    /** `path`: an `/api/plugins/<id>/...` path (query allowed) the caller already validated. */
    preparePlugin(origin: string, path: string, ticket: string, webContentsId: number) {
      return reserve(new URL(path, origin.replace(/^http/, 'ws')), ticket, webContentsId, true)
    },
    headers(details: {
      url: string
      webContentsId?: number
      resourceType: string
      requestHeaders?: Record<string, string>
    }) {
      const dial = dialOf(details.url)
      const grant = dial ? pending.get(dial.nonce) : undefined

      if (
        !dial ||
        !grant ||
        grant.pathname !== dial.pathname ||
        grant.webContentsId !== details.webContentsId ||
        details.resourceType !== 'webSocket' ||
        (grant.header && grant.origin !== dial.origin)
      ) {
        return null
      }

      pending.delete(dial.nonce)

      if (grant.expires <= Date.now()) {
        return null
      }
      const headers = { ...details.requestHeaders }

      for (const key of Object.keys(headers)) {
        if (['origin', 'x-hermes-gateway-ticket'].includes(key.toLowerCase())) {
          delete headers[key]
        }
      }

      return grant.header ? { ...headers, 'X-Hermes-Gateway-Ticket': grant.ticket } : headers
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
  const valid = {
    directory: node.isDirectory(),
    home: node.isDirectory(),
    socket: node.isSocket(),
    file: node.isFile()
  }[kind]
  const uid = process.getuid?.()
  const quoted = shellQuote(file)

  // Not stale (a re-ensure of the same home would hit the same refusal): name the repair.
  if (!valid) {
    const found = node.isSymbolicLink()
      ? 'a symlink'
      : node.isDirectory()
        ? 'a directory'
        : node.isFile()
          ? 'a regular file'
          : node.isSocket()
            ? 'a socket'
            : 'another file type'

    // The socket, pointer and fallback directory are the gateway's to recreate; a home is not.
    throw new Error(
      kind === 'home'
        ? `Unsafe gateway control path: profile home ${quoted} is ${found}, not a directory: move it aside and recreate the profile`
        : `Unsafe gateway control path: ${quoted} should be a ${kind} but is ${found}: remove it, then run hermes gateway restart`
    )
  }

  if (node.uid !== uid) {
    throw new Error(
      `Unsafe gateway control path: ${quoted} is owned by uid ${node.uid}, not this user (uid ${uid}): run sudo chown ${uid} ${quoted}`
    )
  }

  if (kind === 'home' && node.mode & 0o002) {
    throw new Error(
      `Unsafe gateway control path: profile home ${quoted} is writable by other users (mode ${(node.mode & 0o777).toString(8)}): run chmod o-w ${quoted}`
    )
  }

  if (kind !== 'home' && node.mode & 0o077) {
    throw new Error(
      `Unsafe gateway control path: ${quoted} is accessible to other users (mode ${(node.mode & 0o777).toString(8)}): run chmod go-rwx ${quoted}`
    )
  }

  return kind === 'home' && Boolean(node.mode & 0o020)
}

function shellQuote(file: string): string {
  return `'${file.replace(/'/g, `'\\''`)}'`
}

// Exact `/api/profiles/<suffix>` collection paths, mirroring the gateway's native-http scope check
// (hermes_cli/dashboard_auth/native_http.py::_check_profile_scope). Matched on the whole suffix, so a
// profile NAMED `projects` or `sessions` still gets its own ticket on `/api/profiles/projects/soul`.
const PROFILE_COLLECTION_ROUTES = new Set([
  'active',
  'import',
  'projects/tree',
  'sessions',
  'sessions/pull-requests',
  'sessions/sidebar'
])

/** The batched sidebar names its owner in `recents_profile`. A named profile route carries its
 *  owner in the first path segment; all other profile-scoped routes use `profile`. */
function scopedProfileOf(url: URL): null | string {
  const pathname = url.pathname.replace(/\/+$/, '')

  if (pathname === '/api/profiles/sessions/sidebar') {
    return url.searchParams.get('recents_profile')
  }

  const prefix = '/api/profiles/'
  const suffix = pathname.startsWith(prefix) ? pathname.slice(prefix.length) : ''
  const segment = suffix.split('/', 1)[0]

  if (segment && !PROFILE_COLLECTION_ROUTES.has(suffix)) {
    try {
      return decodeURIComponent(segment)
    } catch {
      return null
    }
  }

  return url.searchParams.get('profile')
}

/** The endpoint a `?profile=<name>` request is scoped to on a shared host descriptor: the same
 *  daemon, ticket minted for the sibling profile's home (`profiles/<name>` under the launch root).
 *  The gateway refuses a ticket whose profile it does not serve, so a bad name fails closed. */
export function routedGatewayEndpoint(
  endpoint: GatewayEndpoint,
  urlOrProfile: string,
  _launchHome: string
): GatewayEndpoint {
  const profile = (/^[a-z]+:\/\//.test(urlOrProfile) ? scopedProfileOf(new URL(urlOrProfile)) : urlOrProfile)?.trim()

  if (!profile || profile === 'current') {
    return endpoint
  }
  const own = path.basename(endpoint.profile_id)
  const ownName = path.basename(path.dirname(endpoint.profile_id)) === 'profiles' ? own : 'default'

  if (profile === ownName) {
    return endpoint
  }

  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(profile)) {
    throw new Error('Invalid profile route')
  }
  // The root home is the endpoint's own profile_id (Python resolved it); the raw launch HERMES_HOME
  // may be a symlink, and a ticket for an unresolved sibling home fails 'Noncanonical' (B2-17).
  const root =
    path.basename(path.dirname(endpoint.profile_id)) === 'profiles'
      ? path.dirname(path.dirname(endpoint.profile_id))
      : endpoint.profile_id
  const home = profile === 'default' ? root : path.join(root, 'profiles', profile)

  return { ...endpoint, profile_id: home, control_home: endpoint.control_home || endpoint.profile_id }
}

/** `routedGatewayEndpoint` bound to the home the gateway reserved for the routed profile. It serves
 *  `profiles/<name>` at that directory's resolved home, so a symlinked or junction-linked named
 *  profile mints for its target; the gateway still refuses a home it does not serve. */
export async function routedTicketEndpoint(
  endpoint: GatewayEndpoint,
  urlOrProfile: string,
  launchHome: string
): Promise<GatewayEndpoint> {
  const routed = routedGatewayEndpoint(endpoint, urlOrProfile, launchHome)

  // The endpoint's own home and the root it was routed from are advertised: they keep their checks.
  if (routed.profile_id === endpoint.profile_id || routed.profile_id === routed.control_home) {
    return routed
  }

  const home = await reservedHome(routed.profile_id).catch(error => {
    // Not stale: re-ensuring the shared connection cannot bring a deleted profile back.
    if (isMissingNodeError(error)) {
      throw new Error(
        `Profile '${path.basename(routed.profile_id)}' does not exist: no directory at ${routed.profile_id}`
      )
    }
    throw error
  })

  return { ...routed, profile_id: home }
}

/** What a native grant is minted for: the URL's own profile selector or, when it names none, a
 *  JSON body's `profile`. The gateway checks both against the grant, and a session write such as
 *  `PATCH /api/sessions/<id>` names its owner in the body alone. */
export function nativeGrantRoute(url: string, body?: unknown): string {
  if (scopedProfileOf(new URL(url))?.trim()) {
    return url
  }
  const profile =
    body && typeof body === 'object' && !Array.isArray(body) ? (body as { profile?: unknown }).profile : undefined

  return typeof profile === 'string' && profile.trim() ? profile : url
}

export async function nativeGatewayHttpHeaders(
  descriptor: { gatewayEndpoint: GatewayEndpoint; baseUrl: string },
  url: string,
  launchHome = descriptor.gatewayEndpoint.profile_id,
  method = 'GET',
  body?: unknown
) {
  if (
    new URL(url).origin !== descriptor.gatewayEndpoint.api_origin ||
    descriptor.baseUrl !== descriptor.gatewayEndpoint.api_origin
  ) {
    throw new Error('Native gateway HTTP origin mismatch')
  }

  const purpose =
    method.toUpperCase() === 'POST' && new URL(url).pathname.replace(/\/$/, '') === '/api/profiles'
      ? 'native-profile-create'
      : 'native-http'

  return {
    'X-Hermes-Gateway-Ticket': await mintLocalGatewayTicket(
      await routedTicketEndpoint(descriptor.gatewayEndpoint, nativeGrantRoute(url, body), launchHome),
      purpose
    )
  }
}

// The installed runtime's own ticket client: every Windows mint (SID-validated pipe), and a POSIX
// home whose group-write only Python can prove private.
let pythonTicketClient:
  | ((endpoint: GatewayEndpoint, purpose: 'interactive' | 'native-http' | 'native-profile-create') => Promise<string>)
  | undefined

export function configurePythonGatewayTicketClient(client: NonNullable<typeof pythonTicketClient>) {
  pythonTicketClient = client
}

async function mintGroupWritableHome(
  endpoint: GatewayEndpoint,
  purpose: 'interactive' | 'native-http' | 'native-profile-create',
  homes: string[]
) {
  if (!pythonTicketClient) {
    throw new Error(
      `Unsafe gateway control path: profile home is group-writable and this Desktop cannot verify the group is private: run chmod g-w ${homes.map(shellQuote).join(' ')}`
    )
  }

  try {
    return await pythonTicketClient(endpoint, purpose)
  } catch (error) {
    if ((error as { reason?: string })?.reason !== 'unsafe_control_permissions') {
      throw error
    }
    const quoted = homes.map(shellQuote).join(' ')

    throw new Error(`Profile home is writable by a group other accounts share: run chmod g-w ${quoted}`)
  }
}

function isMissingNodeError(error: unknown): boolean {
  const code = (error as NodeJS.ErrnoException)?.code

  return code === 'ENOENT' || code === 'ENOTDIR'
}

/** Python's `posixpath.realpath`: each ordinary component keeps the spelling it was given, and a
 *  symlink is replaced by its target text, joined onto the prefix resolved so far (an absolute
 *  target restarts from `/`). Unlike Python it throws for a missing component instead of keeping it. */
async function posixRealpath(dir: string): Promise<string> {
  const rest = dir.split('/').reverse()
  let resolved = dir.startsWith('/') ? '/' : process.cwd()
  let links = 0

  while (rest.length) {
    const name = rest.pop()

    if (!name || name === '.') {
      continue
    }

    if (name === '..') {
      resolved = resolved.slice(0, resolved.lastIndexOf('/')) || '/'

      continue
    }

    const next = resolved === '/' ? `/${name}` : `${resolved}/${name}`

    if (!(await fs.lstat(next)).isSymbolicLink()) {
      resolved = next

      continue
    }

    if (++links > 40) {
      throw Object.assign(new Error(`Too many levels of symbolic links: ${dir}`), { code: 'ELOOP' })
    }
    const target = await fs.readlink(next)

    if (target.startsWith('/')) {
      resolved = '/'
    }
    rest.push(...target.split('/').reverse())
  }

  return resolved
}

/** The home the gateway reserves for `dir`: its `Path.resolve()`. On POSIX that keeps the given
 *  spelling of every ordinary component, where macOS realpath(3) answers a case-insensitive volume
 *  in its stored letter case (#138754). Windows mints through the runtime bridge, whose
 *  `Path.resolve()` is the final on-disk path, as Node's realpath is. */
export async function reservedHome(dir: string): Promise<string> {
  return process.platform === 'win32' ? fs.realpath(dir) : posixRealpath(dir)
}

async function resolveControlSocket(home: string): Promise<string> {
  const socketPath = path.join(home, 'gateway.sock')

  try {
    await privateNode(socketPath, 'socket')
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw error
    }
    const pointer = path.join(home, 'gateway.sock.path')
    await privateNode(pointer, 'file')
    const target = (await fs.readFile(pointer, 'utf8')).trim()
    const hash = crypto.createHash('sha256').update(home).digest('hex').slice(0, 16)

    if (
      !path.isAbsolute(target) ||
      path.basename(path.dirname(target)) !== `hermes-gw-${hash}` ||
      path.basename(target) !== 'control.sock'
    ) {
      throw new Error('Invalid gateway control pointer')
    }
    await privateNode(path.dirname(target), 'directory')
    await privateNode(target, 'socket')

    return target
  }

  return socketPath
}

/** The Desktop's interactive socket is the one shared-primary route: its renderer selects every
 *  served sibling with a `profile` param, so it asks for a host-scoped grant explicitly. Every
 *  other ticket (and every other client's) stays bound to the one profile it names. */
export function ticketScope(purpose: 'interactive' | 'native-http' | 'native-profile-create'): { scope?: 'host' } {
  return purpose === 'interactive' ? { scope: 'host' } : {}
}

export async function mintLocalGatewayTicket(
  endpoint: GatewayEndpoint,
  purpose: 'interactive' | 'native-http' | 'native-profile-create' = 'interactive'
): Promise<string> {
  if (process.platform === 'win32') {
    if (!pythonTicketClient) {
      throw new Error('Gateway ticket client is not configured')
    }

    return pythonTicketClient(endpoint, purpose)
  }

  const home = endpoint.profile_id
  // A served secondary has no socket of its own: the default multiplexer's control socket
  // mints its tickets (bound to the secondary's profile_id). Same rule as
  // hermes_cli.gateway_runtime.control_home_for.
  const controlHome = endpoint.control_home || home

  const groupWritable: string[] = []

  for (const dir of new Set([home, controlHome])) {
    // Identity, not spelling: an advertised home is refused only when it leads somewhere else.
    if ((await reservedHome(dir)) !== dir) {
      throw new Error('Noncanonical gateway profile')
    }

    if (await privateNode(dir, 'home')) {
      groupWritable.push(dir)
    }
  }

  if (groupWritable.length) {
    return mintGroupWritableHome(endpoint, purpose, groupWritable)
  }

  let socketPath: string

  try {
    socketPath = await resolveControlSocket(controlHome)
  } catch (error) {
    // `gateway stop` unlinks both the socket and its pointer: the owner is gone,
    // which the redial must treat as stale rather than as a raw filesystem error.
    if (isMissingNodeError(error)) {
      throw new Error('Gateway ticket control socket missing')
    }
    throw error
  }

  return new Promise((resolve, reject) => {
    const socket = net.createConnection(socketPath)
    let buffer = ''
    const deadline = setTimeout(() => socket.destroy(new Error('Gateway ticket deadline')), 5000)
    socket.on('error', () => reject(new Error('Gateway ticket bootstrap failed')))
    socket.on('close', () => {
      clearTimeout(deadline)
      reject(new Error('Gateway ticket connection closed'))
    })
    socket.on('connect', () =>
      socket.write(
        JSON.stringify({
          protocol: 1,
          id: 1,
          verb: 'session-ticket',
          params: { profile_id: home, instance_id: endpoint.instance_id, purpose, ...ticketScope(purpose) }
        }) + '\n'
      )
    )
    socket.on('data', chunk => {
      buffer += chunk.toString()

      if (buffer.length > 65536) {
        return socket.destroy(new Error('Oversized gateway ticket response'))
      }

      if (!buffer.includes('\n')) {
        return
      }

      try {
        const reply = JSON.parse(buffer.split('\n')[0])
        const grant = reply.result

        if (
          !reply.ok ||
          reply.protocol !== 1 ||
          reply.id !== 1 ||
          grant?.instance_id !== endpoint.instance_id ||
          grant?.profile_id !== home ||
          typeof grant?.ticket !== 'string'
        ) {
          throw new Error('Invalid gateway ticket')
        }
        resolve(grant.ticket)
      } catch {
        reject(new Error('Invalid gateway ticket response'))
      }

      socket.destroy()
    })
  })
}
