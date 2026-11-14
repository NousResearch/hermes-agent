// Mixed remote mode (app-global remote + an explicit override on the active
// profile): the window's primary backend is that override, but the app-global
// remote still owns every inherited profile's sessions, the default profile's
// Telegram chat included (#81198). This dials that remote separately for those
// session reads and mutations; every other topology keeps the primary route.

import { authModeFromStatus, normalizeRemoteBaseUrl } from './connection-config'
import { type ProfileSessionAggregateFetchOptions, resolveProfileSessionAggregateRoute } from './profile-session-routing'

interface RemoteConnection {
  authMode?: string
  baseUrl: string
  token?: string
}

export interface GlobalRemoteSessionDeps {
  globalRemoteActive: () => boolean
  primaryHasRemoteOverride: () => boolean
  request: (conn: RemoteConnection, path: string, method: string, body?: unknown) => Promise<unknown>
  requestPrimary: (path: string, method: string, body?: unknown) => Promise<unknown>
  resolveGlobalRemote: () => Promise<null | RemoteConnection>
  waitReady: (conn: RemoteConnection) => Promise<unknown>
}

export function createGlobalRemoteSessionRoute(deps: GlobalRemoteSessionDeps) {
  let pending: null | Promise<RemoteConnection> = null

  const connect = async (): Promise<RemoteConnection> => {
    const remote = await deps.resolveGlobalRemote()

    if (!remote) {
      throw new Error('The app-global Hermes backend is not remote.')
    }

    await deps.waitReady(remote)

    return remote
  }

  // A failed dial or request drops the cached connection so the next read redials.
  const requestGlobal = async (path: string, method: string, body?: unknown) => {
    const attempt = pending || (pending = connect())

    try {
      return await deps.request(await attempt, path, method, body)
    } catch (error) {
      if (pending === attempt) {
        pending = null
      }

      throw error
    }
  }

  const fetchOptions = (): ProfileSessionAggregateFetchOptions => ({
    globalRemote: deps.globalRemoteActive(),
    primaryProfileRemoteOverride: deps.primaryHasRemoteOverride(),
    fetchJsonForGlobalRemote: (_profile, path) => requestGlobal(path, 'GET')
  })

  return {
    fetchOptions,
    request: (path: string, method: string, body?: unknown) =>
      resolveProfileSessionAggregateRoute(fetchOptions()) === 'global-remote'
        ? requestGlobal(path, method, body)
        : deps.requestPrimary(path, method, body),
    reset: () => {
      pending = null
    }
  }
}

type FetchPublicJson = (url: string, options?: { timeoutMs?: number }) => Promise<unknown>

export async function probeRemoteAuthMode(rawUrl: string, fetchPublicJson: FetchPublicJson) {
  // Determine how a remote gateway expects callers to authenticate, WITHOUT
  // sending any credentials. ``/api/status`` is public on every Hermes
  // gateway (it backs the portal liveness probe) and reports:
  //   auth_required: true  → OAuth gate is engaged (cookie + ws-ticket auth)
  //   auth_required: false → loopback/--insecure: legacy session-token auth
  // ``/api/auth/providers`` (also public, only meaningful when gated) gives
  // the human-facing provider name(s) for the login button label.
  //
  // The settings UI calls this as the user types a URL so it can render an
  // OAuth login button vs a session-token entry box. Network/parse failures
  // surface as ``reachable: false`` rather than throwing, so a half-typed or
  // unreachable URL degrades to "can't tell yet" instead of a hard error.
  const baseUrl = normalizeRemoteBaseUrl(rawUrl)

  let status

  try {
    status = await fetchPublicJson(`${baseUrl}/api/status`, { timeoutMs: 8_000 })
  } catch (error: any) {
    return {
      baseUrl,
      reachable: false,
      authMode: 'unknown',
      providers: [],
      version: null,
      error: error instanceof Error ? error.message : String(error)
    }
  }

  const authRequired = authModeFromStatus(status) === 'oauth'
  let providers = []

  if (authRequired) {
    // Best-effort: a gated gateway exposes the registered providers so the
    // button can read "Sign in with Nous Research" instead of a generic
    // label, and so a username/password provider can be distinguished from
    // an OAuth-redirect one (``supports_password``). A failure here doesn't
    // change the auth mode, so swallow it.
    try {
      const body = (await fetchPublicJson(`${baseUrl}/api/auth/providers`, { timeoutMs: 8_000 })) as any

      if (Array.isArray(body?.providers)) {
        providers = body.providers
          .filter(p => p && typeof p === 'object')
          .map(p => ({
            name: String(p.name || ''),
            displayName: String(p.display_name || p.name || ''),
            supportsPassword: Boolean(p.supports_password)
          }))
          .filter(p => p.name)
      }
    } catch {
      // Provider listing is optional metadata; the auth mode is already known.
    }
  }

  return {
    baseUrl,
    reachable: true,
    authMode: authRequired ? 'oauth' : 'token',
    providers,
    version: status?.version || null,
    error: null
  }
}
