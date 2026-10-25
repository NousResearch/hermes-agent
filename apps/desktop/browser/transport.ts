import type { HermesApiRequest, HermesConnection } from '../src/global'

import { browserMessage } from './messages'

declare global {
  interface Window {
    __HERMES_SESSION_TOKEN__?: string
    __HERMES_AUTH_REQUIRED__?: boolean
    __HERMES_BASE_PATH__?: string
    __HERMES_INITIAL_PROFILE__?: string
    __HERMES_DASHBOARD_PROFILE__?: string
  }
}

let redirecting = false
export const connectionId = 'browser'
let activeProfile: string | undefined

function initialProfile(): string | undefined {
  return new URLSearchParams(window.location.search).get('profile') || window.__HERMES_INITIAL_PROFILE__ || window.__HERMES_DASHBOARD_PROFILE__ || undefined
}

export function setActiveProfile(profile?: string) { activeProfile = profile }

export function backendPath(path: string): string {
  if (!path.startsWith('/') || path.startsWith('//') || path.includes('\\')) {throw new Error('Backend paths must be relative to this server.')}
  const base = window.__HERMES_BASE_PATH__ || ''

  return `${base}${path}`
}

export function withScope(path: string, profile?: null | string): string {
  if (!path.startsWith('/') || path.startsWith('//') || path.includes('\\')) {throw new Error('Backend paths must be relative to this server.')}
  const url = new URL(path, window.location.origin)

  if (profile?.trim() && !url.searchParams.has('profile')) {url.searchParams.set('profile', profile.trim())}

  return `${url.pathname}${url.search}`
}

export async function parseApiResponse<T>(response: Response): Promise<T> {
  const text = await response.text()
  let value: unknown

  try { value = text ? JSON.parse(text) : undefined } catch { /* report a bounded, markup-free error */ }

  if (!response.ok) {
    const record = value && typeof value === 'object' ? value as Record<string, unknown> : undefined
    const message = record?.detail || record?.error || `${response.status} ${response.statusText}`.trim()
    throw Object.assign(new Error(String(message)), { statusCode: response.status })
  }

  if (text && value === undefined) {throw new Error(`Expected a JSON response (${response.status})`)}

  return value as T
}

export async function backendFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)

  if (!window.__HERMES_AUTH_REQUIRED__ && window.__HERMES_SESSION_TOKEN__) {
    headers.set('X-Hermes-Session-Token', window.__HERMES_SESSION_TOKEN__)
  }

  const scopedPath = path.startsWith('/api/') ? withScope(path, activeProfile || initialProfile()) : path
  const response = await fetch(backendPath(scopedPath), { ...init, headers, credentials: 'same-origin' })

  if (response.status === 401) {
    const body: { error?: string; login_url?: string } = await response.clone().json().catch(() => ({}))

    if (!redirecting && ['unauthenticated', 'session_expired'].includes(body.error || '') && body.login_url) {
      const login = new URL(body.login_url, window.location.origin)

      if (login.origin === window.location.origin) {
        redirecting = true
        login.searchParams.set('next', `${window.location.pathname}${window.location.search}${window.location.hash}`)
        window.location.assign(login.toString())
      }
    }
  }

  return response
}

async function performRequestJson<T>(request: HermesApiRequest): Promise<T> {
  if (request.connectionId && request.connectionId !== connectionId) {throw new Error(browserMessage('backendTab'))}
  const headers = new Headers({ Accept: 'application/json' })
  let body: BodyInit | undefined

  if (request.upload) {
    const data = new FormData()
    data.append('file', new File([request.upload.bytes], request.upload.filename, { type: request.upload.contentType || 'application/octet-stream' }))
    body = data
  } else if (request.body !== undefined) {
    headers.set('Content-Type', 'application/json')
    body = JSON.stringify(request.body)
  }

  const signal = AbortSignal.timeout(request.timeoutMs || 60_000)

  return parseApiResponse<T>(await backendFetch(withScope(request.path, request.profile || activeProfile || initialProfile()), {
    method: request.method || (body === undefined ? 'GET' : 'POST'), headers, body, signal
  }))
}

// Share only overlapping config reads: completed values are never cached, and a
// write strands older flights so its follow-up refresh always reaches the server.
const configReads = new Map<string, Promise<unknown>>()

export async function requestJson<T>(request: HermesApiRequest): Promise<T> {
  const method = (request.method || (request.body === undefined && !request.upload ? 'GET' : 'POST')).toUpperCase()

  if (method !== 'GET') {
    configReads.clear()

    try { return await performRequestJson<T>(request) } finally { configReads.clear() }
  }

  if (request.body !== undefined || request.upload || !['/api/config', '/api/config/defaults'].includes(request.path)) {
    return performRequestJson<T>(request)
  }

  const key = JSON.stringify([window.location.origin, window.__HERMES_BASE_PATH__,
    window.__HERMES_AUTH_REQUIRED__, window.__HERMES_SESSION_TOKEN__, request.connectionId,
    withScope(request.path, request.profile || activeProfile || initialProfile()), request.timeoutMs || 60_000])

  let flight = configReads.get(key)

  if (!flight) {
    flight = performRequestJson<T>(request).finally(() => {
      if (configReads.get(key) === flight) {configReads.delete(key)}
    })
    configReads.set(key, flight)
  }

  return structuredClone(await flight) as T
}

export async function gatewayWsUrl(profile?: null | string, id?: null | string) {
  if (id && id !== connectionId) {return { ok: false as const, error: browserMessage('backendTab') }}
  const url = new URL(backendPath('/api/ws'), window.location.origin)
  url.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const selected = profile?.trim() || activeProfile || initialProfile()

  if (selected) {url.searchParams.set('profile', selected)}

  if (window.__HERMES_AUTH_REQUIRED__) {
    try {
      const { ticket } = await requestJson<{ ticket: string }>({ path: '/api/auth/ws-ticket', method: 'POST', body: {} })
      url.searchParams.set('ticket', ticket)
    } catch (error) {
      return { ok: false as const, error: error instanceof Error ? error.message : String(error), needsOauthLogin: (error as { statusCode?: number }).statusCode === 401 }
    }
  } else {
    url.searchParams.set('token', window.__HERMES_SESSION_TOKEN__ || '')
  }

  return { ok: true as const, wsUrl: url.toString() }
}

export function connection(profile?: null | string): HermesConnection {
  const selected = profile?.trim() || activeProfile || initialProfile()
  const url = new URL(backendPath('/api/ws'), window.location.origin)
  url.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'

  if (selected) {url.searchParams.set('profile', selected)}

  return {
    authMode: window.__HERMES_AUTH_REQUIRED__ ? 'oauth' : 'token',
    baseUrl: `${window.location.origin}${window.__HERMES_BASE_PATH__ || ''}`,
    connectionId, registryScoped: true, isFullscreen: false, logs: [], mode: 'remote',
    nativeOverlayWidth: 0, profile: selected, remoteHost: window.location.host,
    remoteIdentity: window.location.origin, remoteKind: 'url', source: 'settings',
    token: window.__HERMES_AUTH_REQUIRED__ ? '' : window.__HERMES_SESSION_TOKEN__ || '',
    windowButtonPosition: null, wsUrl: url.toString()
  }
}
