import { isGatewayAuthRejection } from './connection-config'
import type { mintLocalGatewayTicket } from './local-gateway';
import { type GatewayEndpoint, localGatewayDial } from './local-gateway'

/** What the renderer's `pluginSocket` asks for: its plugin namespace, the path under it, and the
 *  (connectionId, profile) route every other plugin call resolves through. */
export interface PluginWsUrlRequest {
  connectionId?: unknown
  path?: unknown
  pluginId?: unknown
  profile?: unknown
}

/** `hermes:plugin:ws-url`'s reply: a one-use URL, null to stay on polling (not a local gateway), or
 *  the one refusal main can name, which the renderer stops on. Any other failure rejects, and the
 *  renderer keeps its capped retry: it cannot tell a refusal from a gateway that is away. */
export type PluginWsUrlReply = null | string | { refused: 'permission' }

/** A grant mint refused for a permission reason: an unsafe control path or a group-writable home
 *  (this Desktop's own checks, or the runtime bridge's `unsafe_control_permissions`). Unlike an
 *  owner that is restarting, draining or being re-ensured, it does not clear by itself. */
export function isMintPermissionRefusal(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return message.startsWith('Unsafe gateway control path') || message.startsWith('Profile home is writable by a group')
    || (error as { reason?: unknown })?.reason === 'unsafe_control_permissions'
}

interface ResolvedBackend {
  baseUrl: string
  gatewayEndpoint?: GatewayEndpoint
  authMode?: string
  headers?: Record<string, string>
}

/** A path segment as the server reads it (decoded once), or null when it is not valid encoding. */
function decodedSegment(segment: string): null | string {
  try {
    return decodeURIComponent(segment)
  } catch {
    return null
  }
}

/** `/api/plugins/<pluginId><path>` as the dial will carry it, or throw. The namespace is the
 *  boundary: the renderer's own check is not trusted here, so a traversal or another core route can
 *  never get a grant. Segments are checked decoded (`%2e%2e`, `%2f`, `%5c`, and `%25` double
 *  encoding are refused), and the result is what the URL parser makes of it. */
export function pluginWsPath(pluginId: unknown, path: unknown): string {
  if (typeof pluginId !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(pluginId) || pluginId.includes('..')) {
    throw new Error('Invalid plugin id')
  }

  const prefix = `/api/plugins/${pluginId}/`

  const url = typeof path === 'string' && path.startsWith('/') && path.length <= 2048 && !/[#\\]/.test(path)
    && !path.split('?', 1)[0].split('/').some(segment => {
      const decoded = decodedSegment(segment)

      return decoded === null || decoded === '.' || decoded === '..' || /[/\\%]/.test(decoded)
    })
    ? new URL(`${prefix.slice(0, -1)}${path}`, 'ws://plugin.invalid')
    : null

  if (!url?.pathname.startsWith(prefix)) {
    throw new Error('Invalid plugin socket path')
  }

  return `${url.pathname}${url.search}`
}

/** The server-side socket name for a plugin's event stream, or null when the id
 *  does not fit the name the ticket door accepts (a plugin id valid in a REST
 *  path may be too liberal for a socket name — minting would only be refused). */
export function pluginEventsSocketName(pluginId: unknown): null | string {
  return typeof pluginId === 'string' && /^[a-z0-9][a-z0-9_-]{0,127}$/.test(pluginId)
    ? `plugin-events/${pluginId}`
    : null
}

/** True for the one plugin path a remote ticket may open: the plugin's own event
 *  stream. Every other plugin path stays on the renderer's fallback (null). */
export function isPluginEventsPath(path: string): boolean {
  return /^\/events(?:\?|$)/.test(path)
}

/** A remote (wss) URL for `path` (already scoped to `/api/plugins/<id>`, it may
 *  carry its own query — `?board=..&since=..`) with the one-use ticket merged
 *  into that query: the ticket is the only credential, it never lands in a path. */
export function pluginWsUrlWithTicket(baseUrl: string, path: string, ticket: string): string {
  const base = new URL(baseUrl)
  const scheme = base.protocol === 'https:' ? 'wss' : 'ws'
  const prefix = base.pathname.replace(/\/+$/, '')
  const queryAt = path.indexOf('?')
  const pathname = queryAt < 0 ? path : path.slice(0, queryAt)
  const search = queryAt < 0 ? '' : path.slice(queryAt + 1)
  const params = new URLSearchParams(search)
  params.set('ticket', ticket)

  return `${scheme}://${base.host}${prefix}${pathname}?${params.toString()}`
}

/** A fresh one-use plugin socket URL for a local gateway, null for any other backend (those keep
 *  the renderer's token path). The grant is the `native-http` one the plugin's REST calls already
 *  carry, minted for the routed profile and attached by `prepare` to that one dial as a header.
 *  The route resolves exactly as `hermes:gateway:ws-url(-for)` does: registry-scoped when the
 *  renderer names a connection, else the profile-keyed local pool. An OAuth remote gets the same
 *  one-use treatment through `mintRemotePluginTicket`: a named ticket (`plugin-events/<id>`)
 *  minted with the OAuth session, valid for exactly this plugin's event stream. */
export function createPluginWsUrlHandler(deps: {
  ensureBackend: (profile: unknown) => Promise<ResolvedBackend>
  ensureRegistryBackend: (connectionId: unknown, profile: unknown) => Promise<ResolvedBackend>
  forgetLocalGatewayDescriptor: (profile: unknown) => Promise<void> | void
  forgetRegistryLocalGatewayDescriptor: (connectionId: unknown, profile: unknown) => Promise<void> | void
  launchHome: string
  mint?: typeof mintLocalGatewayTicket
  mintRemotePluginTicket?: (baseUrl: string, headers: Record<string, string>, socket: string) => Promise<string>
  prepare: (origin: string, path: string, ticket: string, webContentsId: number) => string
}) {
  return async (request: PluginWsUrlRequest, webContentsId: number): Promise<PluginWsUrlReply> => {
    const path = pluginWsPath(request?.pluginId, request?.path)
    const { connectionId, profile } = request
    const ensure = () => (connectionId ? deps.ensureRegistryBackend(connectionId, profile) : deps.ensureBackend(profile))

    const backend = await ensure()

    if (!backend.gatewayEndpoint) {
      // Not a local gateway: an OAuth remote opens its plugin event stream with a
      // named one-use ticket instead. Only that stream is named — any other
      // plugin path keeps the renderer's polling fallback (null), and an id the
      // ticket door would refuse never asks for a mint.
      const socket = pluginEventsSocketName(request?.pluginId)
      // `path` is already scoped (`/api/plugins/<id>/events?..`); the events check reads the
      // plugin-relative suffix, which pluginWsPath above has validated.
      const suffix = path.slice(`/api/plugins/${String(request?.pluginId)}`.length)

      if (backend.authMode !== 'oauth' || !socket || !isPluginEventsPath(suffix) || !deps.mintRemotePluginTicket) {
        return null
      }

      try {
        const ticket = await deps.mintRemotePluginTicket(backend.baseUrl, backend.headers ?? {}, socket)

        return pluginWsUrlWithTicket(backend.baseUrl, path, ticket)
      } catch (error) {
        // An auth rejection resolves to null: the plugin stays on its polling fallback and this
        // dial is not retried, so the stream is not picked again until the scope changes or the
        // connection signs in again (re-login belongs to the gateway socket's own re-auth path).
        // Any other fault rejects, and the renderer retries with capped backoff.
        if (isGatewayAuthRejection(error)) {return null}
        throw error
      }
    }

    try {
      return await localGatewayDial({
        ensure,
        forget: () => (connectionId ? deps.forgetRegistryLocalGatewayDescriptor(connectionId, profile) : deps.forgetLocalGatewayDescriptor(profile)),
        launchHome: deps.launchHome,
        mint: deps.mint,
        prepare: (baseUrl, ticket) => deps.prepare(baseUrl, path, ticket, webContentsId),
        profile: typeof profile === 'string' ? profile : '',
        purpose: 'native-http'
      })
    } catch (error) {
      if (isMintPermissionRefusal(error)) {return { refused: 'permission' }}
      throw error
    }
  }
}
