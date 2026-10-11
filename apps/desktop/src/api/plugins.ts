import { reconnectBackoffDelayMs } from '@hermes/shared'

import type { HermesConnection } from '@/global'
import { RECONNECT_ATTEMPT_TIMEOUT_MS, withTimeout } from '@/lib/with-timeout'

import { getApiRequestConnection, getApiRequestProfile, hermesApi, profileScoped } from './client'

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

/** The plugin WebSocket door — the live twin of `pluginRest`, scoped the same
 *  way: `path` is relative to `/api/plugins/<pluginId>` ('/events' → the
 *  plugin's own event stream). Token-mode backends auth via the same query
 *  credential the app's own sockets use; OAuth remotes dial a URL minted by
 *  the main process (`getPluginWsUrl` — a single-use WS ticket per attempt,
 *  the same discipline as the gateway socket). When no fresh URL is available
 *  the caller keeps its polling fallback (every consumer must have one
 *  anyway, since a socket can drop). Auto-reconnects with backoff until
 *  disposed. */
export function pluginSocket(pluginId: string, path: string, onMessage: (data: unknown) => void): () => void {
  const suffix = pluginPathSuffix('pluginSocket', path)

  let socket: null | WebSocket = null
  let disposed = false
  let attempt = 0

  const scheduleRetry = () => {
    if (disposed) {
      return
    }

    // Full-jitter exponential backoff: same rationale as the gateway socket
    // reconnect loops — an immediate-retry loop across many desktop clients
    // floods the gateway with connection attempts during a restart. Also the
    // only retry path for a connect-time bail (below), which owns no socket
    // and therefore gets no onclose of its own.
    window.setTimeout(() => void connect(), reconnectBackoffDelayMs(attempt, { baseDelayMs: 500, capMs: 30_000 }))
    attempt += 1
  }

  const wire = (opened: WebSocket) => {
    opened.onmessage = event => {
      attempt = 0

      try {
        onMessage(JSON.parse(String(event.data)))
      } catch {
        // Non-JSON frame — plugin streams are JSON by contract; skip it.
      }
    }

    opened.onclose = () => {
      socket = null
      scheduleRetry()
    }
  }

  const connect = async () => {
    if (disposed) {
      return
    }

    const connection = await activeConnection().catch(() => null)

    if (disposed) {
      return
    }

    // No bridge: stay on the polling fallback rather than half-working. A
    // TRANSIENT null (bridge round-trip timeout) retries through the same
    // backoff instead of latching the fallback until the next page action —
    // a connect-time bail owns no socket, so nothing else would reschedule.
    if (!connection) {
      scheduleRetry()

      return
    }

    if (connection.authMode === 'oauth') {
      const mintWsUrl = window.hermesDesktop?.getPluginWsUrl

      if (typeof mintWsUrl !== 'function') {
        // Older Desktop main: no minting door for plugin sockets — stay on
        // the polling fallback rather than half-working.
        return
      }

      try {
        const result = await mintWsUrl(
          { connectionId: getApiRequestConnection(), profile: getApiRequestProfile() },
          `/api/plugins/${pluginId}${suffix}`
        )

        if (disposed) {
          return
        }

        const wsUrl = typeof result === 'string' ? result : result?.ok ? result.wsUrl : null

        if (!wsUrl) {
          // No usable credential for this backend right now (or a rejected
          // mint): polling stays the refresh path for this attempt.
          return
        }

        socket = new WebSocket(wsUrl)
        wire(socket)
      } catch {
        // Mint transport failure: never hammer the mint endpoint from the
        // reconnect loop; the polling fallback keeps the board live.
        return
      }

      return
    }

    const base = connection.baseUrl.replace(/^http/, 'ws')
    const join = suffix.includes('?') ? '&' : '?'
    socket = new WebSocket(
      `${base}/api/plugins/${pluginId}${suffix}${join}token=${encodeURIComponent(connection.token)}`
    )
    wire(socket)
  }

  void connect()

  return () => {
    disposed = true
    socket?.close()
  }
}
