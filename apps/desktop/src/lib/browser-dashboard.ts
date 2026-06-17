import type { HermesApiRequest, HermesConnection } from '@/global'

const SESSION_HEADER = 'X-Hermes-Session-Token'

function basePath(): string {
  if (typeof window === 'undefined') {
    return ''
  }

  const raw = window.__HERMES_BASE_PATH__ || ''
  if (!raw) {
    return ''
  }

  return (raw.startsWith('/') ? raw : `/${raw}`).replace(/\/+$/, '')
}

function apiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) {
    return path
  }

  return `${basePath()}${path.startsWith('/') ? path : `/${path}`}`
}

async function parseErrorBody(response: Response): Promise<{ error?: string; login_url?: string } | null> {
  try {
    return (await response.clone().json()) as { error?: string; login_url?: string }
  } catch {
    return null
  }
}

export function isBrowserDashboard(): boolean {
  return typeof window !== 'undefined' && !window.hermesDesktop
}

export async function dashboardApi<T>(request: HermesApiRequest): Promise<T> {
  const headers = new Headers()

  if (window.__HERMES_SESSION_TOKEN__) {
    headers.set(SESSION_HEADER, window.__HERMES_SESSION_TOKEN__)
  }

  if (request.body !== undefined) {
    headers.set('Content-Type', 'application/json')
  }

  const response = await fetch(apiUrl(request.path), {
    body: request.body === undefined ? undefined : JSON.stringify(request.body),
    credentials: 'include',
    headers,
    method: request.method || (request.body === undefined ? 'GET' : 'POST')
  })

  if (response.status === 401) {
    const body = await parseErrorBody(response)

    if (
      (body?.error === 'unauthenticated' || body?.error === 'session_expired') &&
      body.login_url &&
      typeof window !== 'undefined'
    ) {
      window.location.assign(body.login_url)
    }
  }

  if (!response.ok) {
    const body = await parseErrorBody(response)
    throw new Error(body?.error || `${request.path}: HTTP ${response.status}`)
  }

  if (response.status === 204) {
    return undefined as T
  }

  return (await response.json()) as T
}

export async function dashboardWsAuthParam(): Promise<[name: string, value: string]> {
  if (window.__HERMES_AUTH_REQUIRED__) {
    const ticket = await dashboardApi<{ ticket: string; ttl_seconds: number }>({
      body: {},
      method: 'POST',
      path: '/api/auth/ws-ticket'
    })

    return ['ticket', ticket.ticket]
  }

  const token = window.__HERMES_SESSION_TOKEN__ || ''

  if (!token) {
    throw new Error('Dashboard session token is unavailable')
  }

  return ['token', token]
}

export async function dashboardGatewayWsUrl(): Promise<string> {
  const [name, value] = await dashboardWsAuthParam()
  const base = basePath()
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = new URL(`${base}/api/ws`, `${wsProtocol}//${window.location.host}`)
  url.searchParams.set(name, value)

  return url.toString()
}

export async function getBrowserDashboardConnection(): Promise<HermesConnection> {
  return {
    authMode: window.__HERMES_AUTH_REQUIRED__ ? 'oauth' : 'token',
    baseUrl: `${window.location.origin}${basePath()}`,
    isFullscreen: false,
    logs: [],
    mode: 'remote',
    nativeOverlayWidth: 0,
    source: 'env',
    token: window.__HERMES_SESSION_TOKEN__ || '',
    windowButtonPosition: null,
    wsUrl: await dashboardGatewayWsUrl()
  }
}
