import { isStableOpen, reconnectBackoffDelayMs } from '@hermes/shared'

import type { HermesConnection } from '@/global'
import { isBrowserHostedDesktop } from '@/lib/platform'
import { RECONNECT_ATTEMPT_TIMEOUT_MS, withTimeout } from '@/lib/with-timeout'

import { $apiRequestScope, getApiRequestConnection, getApiRequestProfile, hermesApi, profileScoped } from './client'

/** Resolve the ACTIVE backend's connection descriptor, (connectionId,
 *  profile)-scoped — mirroring how store/profile resolves $connection: a
 *  registry agent's descriptor comes from getConnectionFor (its SOURCE
 *  connection), everything else from the profile-keyed local pool. The
 *  getConnectionFor bridge is optional (older Desktop mains); without it the
 *  profile-scoped pool lookup is the best available answer.
 *
 *  Both branches are IPC round-trips into the main process with no timeout of
 *  their own (#93454) — a wedged main-process round-trip otherwise hangs
 *  pluginSocket's connect() forever instead of falling back to the polling
 *  fallback every consumer already has. Bound the same way store/gateway's
 *  openSecondary bounds the same *For/plain pair.
 *
 *  Exported for tests. */
export async function activeConnection(): Promise<HermesConnection> {
  const getConnectionFor = window.hermesDesktop.getConnectionFor
  const connectionId = getApiRequestConnection()
  const profile = getApiRequestProfile()

  if (connectionId && getConnectionFor) {
    return withTimeout(
      getConnectionFor({ connectionId, profile }),
      RECONNECT_ATTEMPT_TIMEOUT_MS,
      `Timed out connecting to profile "${profile}"`
    )
  }

  return withTimeout(
    window.hermesDesktop.getConnection(profile),
    RECONNECT_ATTEMPT_TIMEOUT_MS,
    `Timed out connecting to profile "${profile}"`
  )
}

/** Options for a plugin REST call — mirrors the app's own `hermesDesktop.api`
 *  shape, minus the path (which is namespace-derived). */
export interface PluginRestOptions {
  method?: string
  body?: unknown
  /** Single-file multipart upload (see HermesApiRequest.upload). */
  upload?: { filename: string; contentType?: string; bytes: ArrayBuffer }
  timeoutMs?: number
}

// Normalize `path` to a leading-slash suffix relative to `/api/plugins/<id>`.
// The namespace is the boundary — reject `..` so a relative segment can't
// normalize out into another plugin's API or a core route. Check the path
// portion only (before any query/hash).
function pluginPathSuffix(caller: string, path: string): string {
  const suffix = path.startsWith('/') ? path : `/${path}`

  if (suffix.split(/[?#]/, 1)[0].split('/').includes('..')) {
    throw new Error(`${caller}: illegal path traversal in "${path}"`)
  }

  return suffix
}

/** The plugin REST door. Every call is scoped BY CONSTRUCTION to the plugin's
 *  own backend namespace — `path` is relative to `/api/plugins/<pluginId>`
 *  ('/board' → `/api/plugins/kanban/board`), so a plugin can't address another
 *  plugin's API or a core route through it. Profile-aware like every desktop
 *  REST call. Broader reach (core endpoints, another namespace) is the future
 *  declared-capability seam; today the namespace IS the boundary. */
export async function pluginRest<T>(pluginId: string, path: string, opts: PluginRestOptions = {}): Promise<T> {
  if (!window.hermesDesktop?.api) {
    throw new Error('Hermes desktop bridge unavailable')
  }

  const suffix = pluginPathSuffix('pluginRest', path)

  return hermesApi<T>({
    path: `/api/plugins/${pluginId}${suffix}`,
    method: opts.method,
    body: opts.body,
    upload: opts.upload,
    timeoutMs: opts.timeoutMs,
    ...profileScoped()
  })
}

/** Refusals in a row that do not clear by themselves before the socket stops for its current
 *  (connection, profile) scope. Every attempt is gateway traffic that keeps an idle-exit gateway awake. */
const MAX_REFUSED_PLUGIN_DIALS = 5

/** What a plugin socket dials: a URL, null to stay on polling, or main's named refusal. */
type PluginSocketTarget = null | string | { refused: 'permission' }

/** The URL a plugin socket dials, or null to stay on polling. A local gateway
 *  (`authMode: 'native'`, no static token) gets a fresh one-use URL from main:
 *  its native grant rides a request header main attaches to that one dial. */
async function pluginSocketUrl(connection: HermesConnection, pluginId: string, suffix: string): Promise<PluginSocketTarget> {
  if (connection.authMode === 'native') {
    const mint = window.hermesDesktop.getPluginWsUrl

    // An IPC round-trip with no deadline of its own, bounded like activeConnection().
    return mint
      ? withTimeout(
          mint({ connectionId: getApiRequestConnection(), path: suffix, pluginId, profile: getApiRequestProfile() }),
          RECONNECT_ATTEMPT_TIMEOUT_MS,
          `Timed out preparing plugin socket "${pluginId}"`
        )
      : null
  }

  const base = connection.baseUrl.replace(/^http/, 'ws')
  const join = suffix.includes('?') ? '&' : '?'

  return `${base}/api/plugins/${pluginId}${suffix}${join}token=${encodeURIComponent(connection.token)}`
}

const requestScopeKey = () => `${getApiRequestConnection() ?? ''}\n${getApiRequestProfile() ?? ''}`

/** The plugin WebSocket door — the live twin of `pluginRest`, scoped the same
 *  way: `path` is relative to `/api/plugins/<pluginId>` ('/events' → the
 *  plugin's own event stream). Token-mode backends auth via the same query
 *  credential the app's own sockets use, a local gateway via the native grant
 *  its REST twin uses; OAuth remotes resolve null (callers keep their polling
 *  fallback — every consumer must have one anyway, since a socket can drop).
 *
 *  Auto-reconnects with capped backoff until disposed. A dial that never opens,
 *  or a connection descriptor or grant that can't be resolved yet, keeps
 *  retrying: the browser cannot tell a refused upgrade from a gateway that is
 *  restarting, draining or being re-ensured. It stops after
 *  MAX_REFUSED_PLUGIN_DIALS refusals in a row that do not clear by themselves:
 *  main refusing the grant mint for a permission reason, or a server that
 *  accepts and closes at once. A connection or profile change drops a dial
 *  still resolving for the old scope, closes a socket bound to it, resets the
 *  count and dials for the new scope. */
export function pluginSocket(pluginId: string, path: string, onMessage: (data: unknown) => void): () => void {
  const suffix = pluginPathSuffix('pluginSocket', path)

  let socket: null | WebSocket = null
  let disposed = false
  let attempt = 0
  let refused = 0
  let scope = requestScopeKey()
  // Bumped on every connection or profile change. A lookup, backoff timer or socket started under an
  // older generation belongs to another scope: its result is dropped and never counts as a refusal.
  let generation = 0

  const retry = (dial: number, refusal: boolean) => {
    if (disposed || dial !== generation) {
      return
    }

    refused = refusal ? refused + 1 : refused

    if (refused >= MAX_REFUSED_PLUGIN_DIALS) {
      return
    }

    // Full-jitter exponential backoff: same rationale as the gateway
    // socket reconnect loops — an immediate-retry loop across many
    // desktop clients floods the gateway with connection attempts
    // during a restart.
    window.setTimeout(() => {
      if (dial === generation) {
        void connect()
      }
    }, reconnectBackoffDelayMs(attempt, { baseDelayMs: 500, capMs: 30_000 }))
    attempt += 1
  }

  const connect = async () => {
    const dial = generation
    const stale = () => disposed || dial !== generation

    // No bridge: stay on the polling fallback.
    if (!window.hermesDesktop?.getConnection) {
      return
    }

    let connection: HermesConnection

    try {
      connection = await activeConnection()
    } catch {
      // The descriptor can't be resolved yet or the lookup timed out (gateway starting, draining,
      // or re-ensured after a stale ticket): a connection failure, like a dial that never opened.
      return retry(dial, false)
    }

    // OAuth cookie auth (WS tickets are single-use, core-managed) and the browser-hosted
    // Desktop (its session token never rides a URL): stay on the polling fallback rather than
    // half-working.
    if (stale() || connection.authMode === 'oauth' || isBrowserHostedDesktop()) {
      return
    }

    let url: PluginSocketTarget

    try {
      url = await pluginSocketUrl(connection, pluginId, suffix)
    } catch {
      // The local mint failed or timed out (owner away, re-ensuring): a connection failure.
      return retry(dial, false)
    }

    if (stale() || !url) {
      return
    }

    if (typeof url !== 'string') {
      return retry(dial, true)
    }

    const current = new WebSocket(url)
    socket = current
    let openedAt: null | number = null

    current.onopen = () => {
      openedAt = Date.now()
    }

    current.onmessage = event => {
      if (stale()) {
        return
      }

      attempt = 0

      try {
        onMessage(JSON.parse(String(event.data)))
      } catch {
        // Non-JSON frame — plugin streams are JSON by contract; skip it.
      }
    }

    current.onclose = () => {
      if (socket === current) {
        socket = null
      }

      if (stale()) {
        return
      }

      // Only a socket that stayed up resets the ladder. An accept-then-close server
      // (Kanban on an archived board) is up and answering, so it counts as a refusal.
      if (isStableOpen(openedAt)) {
        attempt = 0
        refused = 0
      }

      retry(dial, openedAt !== null && !isStableOpen(openedAt))
    }
  }

  // Another connection or profile is another gateway or another consent. A socket bound to the
  // old scope would show that profile's stream in this view: close it, and dial for the new scope
  // with a fresh count.
  const unlisten = $apiRequestScope.listen(() => {
    const next = requestScopeKey()

    if (next === scope || disposed) {
      return
    }

    scope = next
    generation += 1
    attempt = 0
    refused = 0
    const old = socket
    socket = null
    old?.close()
    void connect()
  })

  void connect()

  return () => {
    disposed = true
    unlisten()
    socket?.close()
  }
}
