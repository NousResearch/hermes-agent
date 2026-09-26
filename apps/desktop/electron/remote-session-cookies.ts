/**
 * remote-session-cookies.ts
 *
 * In-memory per-origin session-cookie mirror for remote gateways (#61457).
 *
 * The `persist:hermes-remote-oauth` partition family is supposed to keep the
 * dashboard `hermes_session*` cookies on disk, but in the field the Chromium
 * jar can drop them (Windows %3A profile folders, lazy hydration, jar flush
 * races), and `electronNet` with `useSessionCookies: true` then intermittently
 * omits the cookie entirely → every authed REST call and WS-ticket mint 401s
 * as `no_cookie` right after a successful sign-in.
 *
 * This module is the belt-and-braces fix: capture every `Set-Cookie` the
 * gateway sends (login window navigations AND authed REST responses) into a
 * process-lifetime map keyed by origin, and let `fetchJsonViaOauthSession`
 * attach them as an explicit `Cookie` header — an explicit header is never
 * subject to the network stack's jar-lookup flakiness.
 *
 * Secrets stay in process memory only (never persisted to disk), and
 * `clear(origin)` drops a stale identity so an old session can never cross
 * into a newly selected one.
 */

export interface ParsedCookie {
  name: string
  value: string
}

/** `Name=Value` from the first attribute pair of a Set-Cookie header; null when unparsable. */
export function parseSetCookie(header: string): ParsedCookie | null {
  if (typeof header !== 'string' || !header.trim()) {
    return null
  }

  const firstPair = header.split(';', 1)[0] ?? ''
  const eq = firstPair.indexOf('=')

  if (eq <= 0) {
    return null
  }

  const name = firstPair.slice(0, eq).trim()
  const value = firstPair.slice(eq + 1).trim()

  if (!name || !value) {
    return null
  }

  return { name, value }
}

/** `protocol//host` origin key for a request URL; null when not http(s). */
export function originKeyFor(url: string): string | null {
  try {
    const parsed = new URL(url)

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return null
    }

    return `${parsed.protocol}//${parsed.host}`
  } catch {
    return null
  }
}

export class RemoteSessionCookieStore {
  private readonly byOrigin = new Map<string, Map<string, string>>()

  /** Record every parsable cookie from a response's `Set-Cookie` header(s). */
  record(url: string, setCookie: string | string[] | undefined | null): void {
    const origin = originKeyFor(url)

    if (!origin || !setCookie) {
      return
    }

    const headers = Array.isArray(setCookie) ? setCookie : [setCookie]
    let jar = this.byOrigin.get(origin)

    for (const header of headers) {
      const parsed = parseSetCookie(header)

      if (!parsed) {
        continue
      }

      if (!jar) {
        jar = new Map()
        this.byOrigin.set(origin, jar)
      }

      jar.set(parsed.name, parsed.value)
    }
  }

  /** Seed the mirror from a session jar read (`sess.cookies.get({url})` results). */
  recordFromJar(url: string, cookies: Array<{ name?: unknown; value?: unknown }> | null | undefined): void {
    const origin = originKeyFor(url)

    if (!origin || !Array.isArray(cookies)) {
      return
    }

    for (const cookie of cookies) {
      if (typeof cookie?.name === 'string' && typeof cookie?.value === 'string' && cookie.name && cookie.value) {
        let jar = this.byOrigin.get(origin)

        if (!jar) {
          jar = new Map()
          this.byOrigin.set(origin, jar)
        }

        jar.set(cookie.name, cookie.value)
      }
    }
  }

  /** Serialize the origin's mirror as a `Cookie` header value; null when empty. */
  cookieHeaderFor(url: string): string | null {
    const origin = originKeyFor(url)
    const jar = origin ? this.byOrigin.get(origin) : undefined

    if (!jar || jar.size === 0) {
      return null
    }

    return [...jar.entries()].map(([name, value]) => `${name}=${value}`).join('; ')
  }

  /** Drop one origin's mirror (stale identity / forced re-login); omit to clear everything. */
  clear(originOrUrl?: string): void {
    if (!originOrUrl) {
      this.byOrigin.clear()

      return
    }

    const origin = originKeyFor(originOrUrl) ?? originOrUrl

    this.byOrigin.delete(origin)
  }
}

/** Process-lifetime mirror used by main.ts. */
export const remoteSessionCookies = new RemoteSessionCookieStore()
