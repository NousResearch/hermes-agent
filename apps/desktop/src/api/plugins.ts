import { reconnectBackoffDelayMs, resolveGatewayWsUrl } from '@hermes/shared'

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

async function freshPluginSocketUrl(connection: HermesConnection, pluginId: string, suffix: string): Promise<string> {
  const desktop = window.hermesDesktop
  const connectionId = connection.registryScoped ? connection.connectionId : null
  const mint =
    connectionId && desktop.getGatewayWsUrlFor
      ? () => desktop.getGatewayWsUrlFor!({ connectionId, profile: connection.profile ?? null })
      : desktop.getGatewayWsUrl

  const gatewayUrl = new URL(await resolveGatewayWsUrl({ getGatewayWsUrl: mint }, connection))
  const gatewaySuffix = '/api/ws'

  if (!gatewayUrl.pathname.endsWith(gatewaySuffix)) {
    throw new Error(`Unexpected gateway WebSocket path: ${gatewayUrl.pathname}`)
  }

  const endpoint = new URL(`/api/plugins/${pluginId}${suffix}`, gatewayUrl.origin)
  const basePath = gatewayUrl.pathname.slice(0, -gatewaySuffix.length)

  gatewayUrl.pathname = `${basePath}${endpoint.pathname}`
  endpoint.searchParams.forEach((value, key) => gatewayUrl.searchParams.set(key, value))
  gatewayUrl.hash = endpoint.hash

  return gatewayUrl.toString()
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
 *  plugin's own event stream). It resolves the active backend's fresh gateway
 *  WebSocket URL before every dial, so OAuth remotes mint a new single-use
 *  ticket while token-mode backends reuse their long-lived credential.
 *  Auto-reconnects with backoff until disposed. */
export function pluginSocket(pluginId: string, path: string, onMessage: (data: unknown) => void): () => void {
  const suffix = pluginPathSuffix('pluginSocket', path)

  let socket: null | WebSocket = null
  let disposed = false
  let attempt = 0
  let reconnectTimer: null | number = null

  const scheduleReconnect = () => {
    if (disposed || reconnectTimer != null) {
      return
    }

    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null
      void connect()
    }, reconnectBackoffDelayMs(attempt, { baseDelayMs: 500, capMs: 30_000 }))
    attempt += 1
  }

  const connect = async () => {
    const connection = await activeConnection().catch(() => null)

    if (disposed) {
      return
    }

    if (!connection) {
      scheduleReconnect()

      return
    }

    let socketUrl: string

    try {
      socketUrl = await freshPluginSocketUrl(connection, pluginId, suffix)
    } catch {
      scheduleReconnect()

      return
    }

    if (disposed) {
      return
    }

    socket = new WebSocket(socketUrl)

    socket.onmessage = event => {
      attempt = 0

      try {
        onMessage(JSON.parse(String(event.data)))
      } catch {
        // Non-JSON frame — plugin streams are JSON by contract; skip it.
      }
    }

    socket.onclose = () => {
      socket = null

      // Full-jitter exponential backoff: same rationale as the gateway socket
      // reconnect loops — an immediate-retry loop across many desktop clients
      // floods the gateway with connection attempts during a restart.
      scheduleReconnect()
    }
  }

  void connect()

  return () => {
    disposed = true
    if (reconnectTimer != null) {
      window.clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    socket?.close()
  }
}
