import { buildHermesWebSocketUrl, normalizeBasePath } from '@hermes/shared'

import type { HermesApiRequest, HermesStagedUpload } from '@/global'
import { consumeWebappSession, WEBAPP_LAUNCH_REQUIRED } from '@/lib/browser-launch-session'
import { hasBrowserHostBootstrap } from '@/lib/platform'

export interface BrowserBootstrapWindow {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_BASE_PATH__?: string
  __HERMES_SESSION_TOKEN__?: string
  __HERMES_UI_SURFACE__?: string
  hermesDesktop?: Window['hermesDesktop']
}

export interface BrowserBootstrap {
  authRequired: boolean
  basePath: string
  /** Webapp on an unauthenticated bind: the operator's launch link is the only credential source. */
  privateSession: boolean
  token: string
  stagedUploads: Map<string, HermesStagedUpload>
}

const SESSION_HEADER = 'X-Hermes-Session-Token'
const DEFAULT_TIMEOUT_MS = 30_000
const REAUTH_EVENT = 'hermes:browser-reauth-required'

export class BrowserReauthRequiredError extends Error {
  readonly loginUrl: string
  readonly needsOauthLogin = true

  constructor(message: string, loginUrl: string) {
    super(message)
    this.name = 'BrowserReauthRequiredError'
    this.loginUrl = loginUrl
  }
}

export function browserBootstrap(): BrowserBootstrap | null {
  if (!hasBrowserHostBootstrap()) {return null}

  const win = window as unknown as BrowserBootstrapWindow
  const authRequired = win.__HERMES_AUTH_REQUIRED__ === true
  const basePath = normalizeBasePath(win.__HERMES_BASE_PATH__)
  const privateSession = !authRequired && win.__HERMES_UI_SURFACE__ === 'webapp'
  const token = authRequired ? '' : privateSession ? consumeWebappSession(basePath) : String(win.__HERMES_SESSION_TOKEN__ || '').trim()

  return {
    authRequired,
    basePath,
    privateSession,
    token,
    stagedUploads: new Map()
  }
}

export function endpointUrl(path: string, basePath: string, profile?: null | string): URL {
  const suffix = path.startsWith('/') ? path : `/${path}`
  const normalizedBase = normalizeBasePath(basePath)
  const url = new URL(`${normalizedBase}${suffix}`, window.location.origin)

  if (url.origin !== window.location.origin) {
    throw new Error('Hermes API paths must remain on the Webapp origin')
  }

  if (
    normalizedBase &&
    url.pathname !== normalizedBase &&
    !url.pathname.startsWith(`${normalizedBase}/`)
  ) {
    throw new Error('Hermes API paths must remain inside the configured base path')
  }

  if (profile && !url.searchParams.has('profile')) {
    url.searchParams.set('profile', profile)
  }

  return url
}

export function websocketUrl(
  basePath: string,
  path: string,
  credential: { ticket?: string; token?: string },
  profile?: null | string
): string {
  const authParam = credential.ticket
    ? (['ticket', credential.ticket] as const)
    : credential.token
      ? (['token', credential.token] as const)
      : undefined

  return buildHermesWebSocketUrl({
    authParam,
    basePath,
    params: profile ? { profile } : undefined,
    path
  })
}

function reauthError(
  bootstrap: BrowserBootstrap,
  response: Response,
  text: string
): BrowserReauthRequiredError | null {
  if (!bootstrap.authRequired || response.status !== 401) {return null}

  let payload: { detail?: unknown; error?: unknown; login_url?: unknown } = {}

  try {
    payload = JSON.parse(text) as typeof payload
  } catch {
    return null
  }

  if (
    !['session_expired', 'unauthenticated'].includes(String(payload.error || '')) ||
    typeof payload.login_url !== 'string'
  ) {
    return null
  }

  let target: URL

  try {
    target = new URL(payload.login_url, window.location.origin)
  } catch {
    return null
  }

  if (target.origin !== window.location.origin) {return null}

  return new BrowserReauthRequiredError(
    `${String(payload.error)}: ${String(payload.detail || 'Unauthorized')}`,
    target.href
  )
}

export function navigateToBrowserLogin(error: BrowserReauthRequiredError): void {
  const event = new CustomEvent(REAUTH_EVENT, {
    cancelable: true,
    detail: { loginUrl: error.loginUrl }
  })

  if (window.dispatchEvent(event)) {window.location.assign(error.loginUrl)}
}

interface BrowserFetchRequest {
  body?: BodyInit
  headers?: HeadersInit
  method?: string
  path: string
  profile?: null | string
  timeoutMs?: number
}

export async function browserFetch(
  bootstrap: BrowserBootstrap,
  request: BrowserFetchRequest
): Promise<{ response: Response; text: string }> {
  if (!bootstrap.authRequired && !bootstrap.token) {throw new Error(WEBAPP_LAUNCH_REQUIRED)}

  const controller = request.timeoutMs === undefined ? null : new AbortController()
  const timeout = controller ? window.setTimeout(() => controller.abort(), request.timeoutMs) : null
  const headers = new Headers(request.headers)

  if (bootstrap.token) {headers.set(SESSION_HEADER, bootstrap.token)}

  try {
    const response = await fetch(endpointUrl(request.path, bootstrap.basePath, request.profile), {
      body: request.body,
      credentials: 'same-origin',
      headers,
      method: request.method || 'GET',
      signal: controller?.signal
    })

    const text = await response.text()

    if (!response.ok) {
      if (!bootstrap.authRequired && response.status === 401) {throw new Error(WEBAPP_LAUNCH_REQUIRED)}

      const authError = reauthError(bootstrap, response, text)

      if (authError) {
        navigateToBrowserLogin(authError)
        throw authError
      }
    }

    return { response, text }
  } finally {
    if (timeout !== null) {window.clearTimeout(timeout)}
  }
}

export async function browserApi<T>(bootstrap: BrowserBootstrap, request: HermesApiRequest): Promise<T> {
  const headers = new Headers()
  let body: BodyInit | undefined

  if (request.upload) {
    const form = new FormData()

    const blob = new Blob([request.upload.bytes], {
      type: request.upload.contentType || 'application/octet-stream'
    })

    form.append('file', blob, request.upload.filename || 'file')
    body = form
  } else if (request.body !== undefined) {
    headers.set('Content-Type', 'application/json')
    body = JSON.stringify(request.body)
  }

  const { response, text } = await browserFetch(bootstrap, {
    body,
    headers,
    method: request.method,
    path: request.path,
    profile: request.profile,
    timeoutMs: request.timeoutMs ?? DEFAULT_TIMEOUT_MS
  })

  if (!response.ok) {
    throw new Error(`${response.status}: ${text || response.statusText}`)
  }

  if (!text) {return null as T}

  if (/^\s*<(?:!doctype|html)/i.test(text)) {
    throw new Error(`Hermes API returned HTML for ${request.path}`)
  }

  return JSON.parse(text) as T
}

export interface FileEndpointQuery {
  path: string
  profile?: null | string
  session_id?: string
}

/**
 * URL for a download link or media element, which cannot send the session
 * header. Gated hosts authenticate it with the same-origin cookie. A token
 * session mints a ticket bound to this exact route and query: the session
 * token grants host shells, so it must never reach download history or a
 * copyable media address.
 */
export async function fileEndpointUrl(
  bootstrap: BrowserBootstrap,
  route: 'download' | 'stream',
  query: FileEndpointQuery
): Promise<URL> {
  const params = Object.fromEntries(Object.entries(query).filter(([, value]) => value)) as Record<string, string>
  const url = endpointUrl(`/api/files/${route}`, bootstrap.basePath)

  Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value))

  if (bootstrap.token) {
    const result = await browserApi<null | { ticket?: string }>(bootstrap, {
      body: { ...params, route },
      method: 'POST',
      path: '/api/files/ticket'
    })

    const ticket = String(result?.ticket || '')

    if (!ticket) {throw new Error('Hermes did not return a file ticket')}
    url.searchParams.set('ticket', ticket)
  }

  return url
}

export async function authenticatedWebsocketUrl(
  bootstrap: BrowserBootstrap,
  path: '/api/host-terminal' | '/api/ws',
  profile?: null | string
): Promise<string> {
  if (!bootstrap.authRequired) {
    if (!bootstrap.token) {throw new Error(WEBAPP_LAUNCH_REQUIRED)}

    return websocketUrl(bootstrap.basePath, path, { token: bootstrap.token }, profile)
  }

  const result = await browserApi<{ ticket?: string }>(bootstrap, {
    method: 'POST',
    path: '/api/auth/ws-ticket'
  })

  const ticket = String(result.ticket || '').trim()

  if (!ticket) {throw new Error('Hermes did not return a WebSocket ticket')}

  return websocketUrl(bootstrap.basePath, path, { ticket }, profile)
}
