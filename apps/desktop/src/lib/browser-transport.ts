import { BROWSER_NATIVE_DIAL, buildHermesWebSocketUrl, normalizeBasePath } from '@hermes/shared'

import type { HermesApiRequest } from '@/global'
import { consumeWebappSession, webappSignInRequired } from '@/lib/browser-launch-session'
import { type BrowserHostGlobals, hasBrowserHostBootstrap } from '@/lib/platform'

export interface BrowserBootstrap {
  authRequired: boolean
  basePath: string
  /** Webapp on an unauthenticated bind: the operator's launch link is the only credential source. */
  privateSession: boolean
  token: string
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

  const win = window as Window & BrowserHostGlobals
  const authRequired = win.__HERMES_AUTH_REQUIRED__ === true
  const basePath = normalizeBasePath(win.__HERMES_BASE_PATH__)
  const privateSession = !authRequired && win.__HERMES_UI_SURFACE__ === 'webapp'
  const token = privateSession ? consumeWebappSession(basePath) : ''

  return {
    authRequired,
    basePath,
    privateSession,
    token
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

/** A socket URL on this origin; its only credential is a one-use ticket, never the session token. */
export function websocketUrl(
  basePath: string,
  path: string,
  credential: { ticket?: string },
  profile?: null | string
): string {
  return buildHermesWebSocketUrl({
    authParam: credential.ticket ? ['ticket', credential.ticket] : undefined,
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
  if (!bootstrap.authRequired && !bootstrap.token) {throw webappSignInRequired()}

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
      if (!bootstrap.authRequired && response.status === 401) {throw webappSignInRequired()}

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
 * Where each ticketed file URL reads. A download uses the workspace route native Desktop downloads
 * with, which reads an SSH-backed profile's files on that machine, like the file tree that lists
 * them; media plays through the managed stream route.
 */
export const FILE_ENDPOINTS = { download: '/api/fs/download', stream: '/api/files/stream' } as const

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
  const url = endpointUrl(FILE_ENDPOINTS[route], bootstrap.basePath)

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

/**
 * The live socket to the gateway serving this page. Every dial mints its own one-use ticket on the
 * same origin; a failed mint is the reason to show. `native_dial` makes the client speak the
 * gateway's canonical contract and move the ticket into the socket's subprotocol.
 */
export async function gatewayWebsocketUrl(bootstrap: BrowserBootstrap, profile?: null | string): Promise<string> {
  const result = await browserApi<null | { ticket?: string }>(bootstrap, {
    method: 'POST',
    path: '/api/webapp/gateway-ticket',
    profile
  })

  const ticket = String(result?.ticket || '').trim()

  if (!ticket) {
    throw new Error('Hermes did not return a gateway ticket')
  }

  const url = new URL(gatewaySocketUrl(bootstrap, profile))
  url.searchParams.set('ticket', ticket)

  return url.toString()
}

/** The credential-free canonical socket URL a browser connection descriptor reports. */
export function gatewaySocketUrl(bootstrap: BrowserBootstrap, profile?: null | string): string {
  const url = new URL(websocketUrl(bootstrap.basePath, '/api/ws', {}, profile))
  url.searchParams.set('native_dial', BROWSER_NATIVE_DIAL)

  return url.toString()
}

/**
 * The host terminal's socket URL for one dial. Its one-use ticket is minted by an authenticated
 * request (the session header on a loopback Webapp, the sign-in cookie on a gated host) and opens
 * only the host terminal, so the session token never rides a socket URL that logs, history and
 * proxies keep.
 */
export async function authenticatedWebsocketUrl(
  bootstrap: BrowserBootstrap,
  path: '/api/host-terminal',
  profile?: null | string
): Promise<string> {
  const result = await browserApi<null | { ticket?: string }>(bootstrap, {
    method: 'POST',
    path: '/api/webapp/terminal-ticket'
  })

  const ticket = String(result?.ticket || '').trim()

  if (!ticket) {throw new Error('Hermes did not return a terminal ticket')}

  return websocketUrl(bootstrap.basePath, path, { ticket }, profile)
}

/** The voice streams beside `/api/ws`, each opened only by a ticket minted under its socket name. */
const AUDIO_SOCKETS = {
  '/api/audio/speak-stream': 'speak-stream',
  '/api/audio/transcribe-stream': 'transcribe-stream'
} as const

export type AudioSocketPath = keyof typeof AUDIO_SOCKETS

/**
 * A voice stream's socket URL for one dial. Its one-use ticket is minted for that stream alone: the
 * gateway's ticket opens only `/api/ws`, so the stream cannot reuse it.
 */
export async function audioWebsocketUrl(
  bootstrap: BrowserBootstrap,
  path: AudioSocketPath,
  profile?: null | string
): Promise<string> {
  const result = await browserApi<null | { ticket?: string }>(bootstrap, {
    method: 'POST',
    path: `/api/webapp/audio-ticket?socket=${AUDIO_SOCKETS[path]}`
  })

  const ticket = String(result?.ticket || '').trim()

  if (!ticket) {throw new Error('Hermes did not return a voice stream ticket')}

  return websocketUrl(bootstrap.basePath, path, { ticket }, profile)
}
