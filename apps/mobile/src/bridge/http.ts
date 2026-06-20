/**
 * http.ts — the REST half of the bridge.
 *
 * On a device, requests go through CapacitorHttp (the NATIVE HTTP stack), which
 * bypasses the browser CORS the gateway locks to localhost. In browser dev they
 * fall back to `fetch` (CORS-limited — UI work only). Session cookies are
 * captured from responses and replayed via our own cookie jar (see cookie-jar).
 *
 * `api<T>()` implements the contract the vendored desktop `hermes.ts` calls:
 * 4xx → throw `status: text`, empty body → null, and the gateway's structured
 * 401 ({error:'session_expired'|'unauthenticated'}) → drive re-login.
 */

import { Capacitor, CapacitorHttp } from '@capacitor/core'

import { buildApiUrl } from './connection-config'
import { captureSetCookie, cookieHeader } from './cookie-jar'
import { currentTarget, requireReauth } from './state'

export interface RawResponse {
  status: number
  headers: Record<string, string>
  text: string
  /** Parsed JSON when the body was JSON, else null. */
  json: unknown
}

export interface RawRequestOptions {
  baseUrl: string
  path: string
  method?: string
  headers?: Record<string, string>
  body?: unknown
  timeoutMs?: number
}

const isNative = Capacitor.isNativePlatform()

function headerLookup(headers: Record<string, string>, name: string): string | undefined {
  const lower = name.toLowerCase()
  for (const [k, v] of Object.entries(headers)) {
    if (k.toLowerCase() === lower) return v
  }
  return undefined
}

/**
 * Low-level request against an explicit base URL. Used by auth.ts (probe/login,
 * before a target is committed) and by `api<T>()`.
 */
export async function rawRequest(opts: RawRequestOptions): Promise<RawResponse> {
  const url = buildApiUrl(opts.baseUrl, opts.path)
  const method = (opts.method ?? 'GET').toUpperCase()

  const headers: Record<string, string> = { Accept: 'application/json', ...opts.headers }
  const hasJsonBody =
    opts.body !== undefined && opts.body !== null && !(opts.body instanceof FormData)
  if (hasJsonBody && !headerLookup(headers, 'content-type')) {
    headers['Content-Type'] = 'application/json'
  }

  if (isNative) {
    // Replay our jar's cookies explicitly (the native auto-jar may or may not).
    const cookie = cookieHeader(opts.baseUrl)
    if (cookie) headers['Cookie'] = cookie

    const res = await CapacitorHttp.request({
      url,
      method,
      headers,
      data: hasJsonBody ? opts.body : undefined,
      connectTimeout: opts.timeoutMs,
      readTimeout: opts.timeoutMs,
      responseType: 'json',
    })

    const resHeaders = (res.headers ?? {}) as Record<string, string>
    await captureSetCookie(opts.baseUrl, headerLookup(resHeaders, 'set-cookie'))

    const data = res.data
    const text = typeof data === 'string' ? data : data == null ? '' : JSON.stringify(data)
    const json = typeof data === 'string' ? tryParse(data) : (data ?? null)
    return { status: res.status, headers: resHeaders, text, json }
  }

  // Browser dev path.
  const res = await fetch(url, {
    method,
    headers,
    credentials: 'include',
    body: hasJsonBody ? JSON.stringify(opts.body) : (opts.body as BodyInit | undefined),
  })
  const resHeaders: Record<string, string> = {}
  res.headers.forEach((v, k) => {
    resHeaders[k] = v
  })
  const text = await res.text()
  return { status: res.status, headers: resHeaders, text, json: tryParse(text) }
}

function tryParse(text: string): unknown {
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

/**
 * The `window.hermesDesktop.api` implementation. Mirrors the desktop/web
 * `fetchJSON` error semantics so vendored `hermes.ts` runs unmodified.
 */
export async function api<T>(request: {
  path: string
  method?: string
  body?: unknown
  timeoutMs?: number
  profile?: string | null
}): Promise<T> {
  const target = currentTarget()
  if (!target) {
    throw new Error('Not connected to a gateway.')
  }

  const res = await rawRequest({
    baseUrl: target.baseUrl,
    path: request.path,
    method: request.method,
    body: request.body,
    timeoutMs: request.timeoutMs,
  })

  if (res.status === 401) {
    const body = (res.json ?? {}) as { error?: string }
    if (body.error === 'unauthenticated' || body.error === 'session_expired') {
      requireReauth()
    }
    throw new Error(`401: ${res.text || 'unauthorized'}`)
  }

  if (res.status < 200 || res.status >= 300) {
    throw new Error(`${res.status}: ${res.text || 'request failed'}`)
  }

  // Empty body → null (matches web/ behavior for 204s and empty 200s).
  if (!res.text) return null as T
  return (res.json ?? null) as T
}
