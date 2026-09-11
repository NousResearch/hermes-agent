import type { HermesConnection } from '@/global'
import { reconnectBackoffDelayMs } from '@/lib/reconnect-backoff'
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

// Keep these segment rules in sync with the defense-in-depth URL parsing check
// in apps/desktop/electron/media-protocol.ts::assertSafePluginMediaSegment.
function assertSafePluginMediaSegment(segment: string, path: string, message: string): void {
  let inspected = segment

  while (true) {
    if (!inspected || inspected === '.' || inspected === '..' || inspected.includes('/') || inspected.includes('\\')) {
      throw new Error(`${message} in "${path}"`)
    }

    // `%252e` decodes once to `%2e`. Inspect nested escapes for path syntax
    // before minting a URL, while preserving literal percent filenames whose
    // follow-up decode is malformed.
    if (!/%[0-9a-f]{2}/i.test(inspected)) {
      return
    }

    try {
      inspected = decodeURIComponent(inspected)
    } catch {
      return
    }
  }
}

function pluginMediaPathSuffix(path: string): string {
  if (path.includes('?') || path.includes('#')) {
    throw new Error(`pluginMediaUrl: query and fragment are not allowed in "${path}"`)
  }

  const rawPath = path.replace(/^\/+/, '')

  if (!rawPath) {
    throw new Error('pluginMediaUrl: media path is required')
  }

  const segments = rawPath.split('/').map(segment => {
    let decoded: string

    try {
      decoded = decodeURIComponent(segment)
    } catch {
      throw new Error(`pluginMediaUrl: malformed path encoding in "${path}"`)
    }

    assertSafePluginMediaSegment(decoded, path, 'pluginMediaUrl: illegal path traversal')

    return encodeURIComponent(decoded)
  })

  return `/${segments.join('/')}`
}

function pluginMediaId(pluginId: string): string {
  let decoded: string

  try {
    decoded = decodeURIComponent(pluginId)
  } catch {
    throw new Error(`pluginMediaUrl: invalid plugin id "${pluginId}"`)
  }

  assertSafePluginMediaSegment(decoded, pluginId, 'pluginMediaUrl: invalid plugin id')

  return encodeURIComponent(pluginId)
}

/** Build a seekable, authenticated media URL for a plugin-owned backend route.
 *  The custom protocol keeps gateway credentials out of the renderer and maps
 *  the URL back to `/api/plugins/<pluginId>` in Electron's main process. */
export function pluginMediaUrl(pluginId: string, path: string): null | string {
  if (!window.hermesDesktop) {
    return null
  }

  const suffix = pluginMediaPathSuffix(path)
  const url = new URL(`hermes-media://plugin/${pluginMediaId(pluginId)}${suffix}`)
  const profile = getApiRequestProfile()
  const connectionId = getApiRequestConnection()

  if (profile) {
    url.searchParams.set('profile', profile)
  }

  if (connectionId) {
    url.searchParams.set('connectionId', connectionId)
  }

  return url.toString()
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
 *  credential the app's own sockets use; OAuth remotes resolve null (callers
 *  keep their polling fallback — every consumer must have one anyway, since a
 *  socket can drop). Auto-reconnects with backoff until disposed. */
export function pluginSocket(pluginId: string, path: string, onMessage: (data: unknown) => void): () => void {
  const suffix = pluginPathSuffix('pluginSocket', path)

  let socket: null | WebSocket = null
  let disposed = false
  let attempt = 0

  const connect = async () => {
    const connection = await activeConnection().catch(() => null)

    // No bridge / OAuth cookie auth (WS tickets are single-use, core-managed):
    // stay on the polling fallback rather than half-working.
    if (disposed || !connection || connection.authMode === 'oauth') {
      return
    }

    const base = connection.baseUrl.replace(/^http/, 'ws')
    const join = suffix.includes('?') ? '&' : '?'
    socket = new WebSocket(
      `${base}/api/plugins/${pluginId}${suffix}${join}token=${encodeURIComponent(connection.token)}`
    )

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

      if (!disposed) {
        // Full-jitter exponential backoff: same rationale as the gateway
        // socket reconnect loops — an immediate-retry loop across many
        // desktop clients floods the gateway with connection attempts
        // during a restart.
        window.setTimeout(() => void connect(), reconnectBackoffDelayMs(attempt, { baseDelayMs: 500, capMs: 30_000 }))
        attempt += 1
      }
    }
  }

  void connect()

  return () => {
    disposed = true
    socket?.close()
  }
}
