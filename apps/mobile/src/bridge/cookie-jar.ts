/**
 * cookie-jar.ts
 *
 * Explicit, persisted cookie storage — the R2 mitigation. CapacitorHttp's
 * native cookie store usually auto-attaches the gateway's session cookies, but
 * rather than depend on that (and on its cross-launch persistence) we capture
 * `Set-Cookie` from every response ourselves, persist name=value to
 * @capacitor/preferences, and replay them as a `Cookie` header on each request.
 *
 * Notes:
 *  - We store only `name=value` (attributes like Path/HttpOnly/Max-Age are
 *    dropped); the gateway only needs the access/refresh token cookie values.
 *  - On web (dev), browsers forbid setting the `Cookie` header via fetch, so
 *    this is effectively a native-only mechanism — which is exactly where the
 *    native jar's behavior is uncertain. Web dev relies on `credentials:'include'`.
 *  - Keyed by host so switching between saved gateways keeps sessions separate.
 */

import { Preferences } from '@capacitor/preferences'

const KEY_PREFIX = 'hermes.cookies.'

type CookieMap = Record<string, string>

const mem = new Map<string, CookieMap>()

function hostOf(url: string): string {
  try {
    return new URL(url).host
  } catch {
    return url
  }
}

function storageKey(host: string): string {
  return `${KEY_PREFIX}${host}`
}

/** Hydrate the in-memory jar for a host from persisted storage. */
export async function loadCookies(url: string): Promise<void> {
  const host = hostOf(url)
  if (mem.has(host)) return
  const { value } = await Preferences.get({ key: storageKey(host) })
  if (value) {
    try {
      mem.set(host, JSON.parse(value) as CookieMap)
    } catch {
      mem.set(host, {})
    }
  } else {
    mem.set(host, {})
  }
}

async function persist(host: string): Promise<void> {
  const map = mem.get(host) ?? {}
  await Preferences.set({ key: storageKey(host), value: JSON.stringify(map) })
}

/** Parse one or more Set-Cookie header values and store them for `url`'s host. */
export async function captureSetCookie(
  url: string,
  setCookie: string | string[] | undefined | null,
): Promise<void> {
  if (!setCookie) return
  const host = hostOf(url)
  const map = mem.get(host) ?? {}
  const headers = Array.isArray(setCookie) ? setCookie : splitSetCookie(setCookie)

  for (const raw of headers) {
    const first = raw.split(';', 1)[0]?.trim()
    if (!first) continue
    const eq = first.indexOf('=')
    if (eq <= 0) continue
    const name = first.slice(0, eq).trim()
    const value = first.slice(eq + 1).trim()
    // A delete instruction (empty value / expired) clears the cookie.
    if (value === '' || /expires=Thu, 01 Jan 1970/i.test(raw)) {
      delete map[name]
    } else {
      map[name] = value
    }
  }

  mem.set(host, map)
  await persist(host)
}

/** Build a `Cookie` header value for `url`'s host, or '' if none. */
export function cookieHeader(url: string): string {
  const map = mem.get(hostOf(url))
  if (!map) return ''
  return Object.entries(map)
    .map(([k, v]) => `${k}=${v}`)
    .join('; ')
}

/** True if the host has any stored hermes session cookie (live-ish). */
export function hasSession(url: string): boolean {
  const map = mem.get(hostOf(url))
  if (!map) return false
  return Object.keys(map).some((n) => n.includes('hermes_session'))
}

/** Forget all cookies for a host (logout). */
export async function clearCookies(url: string): Promise<void> {
  const host = hostOf(url)
  mem.delete(host)
  await Preferences.remove({ key: storageKey(host) })
}

/**
 * Split a comma-joined Set-Cookie string back into individual cookies. Commas
 * appear inside `Expires=Wed, 21 Oct …` dates, so we only split on a comma that
 * is followed by ` token=` (a new cookie pair), not a date comma.
 */
function splitSetCookie(value: string): string[] {
  const out: string[] = []
  let start = 0
  for (let i = 0; i < value.length; i++) {
    if (value[i] === ',') {
      const rest = value.slice(i + 1)
      // New cookie if what follows looks like `  name=`(letters/digits/_-)…
      if (/^\s*[A-Za-z0-9!#$%&'*+\-.^_`|~]+=/.test(rest)) {
        out.push(value.slice(start, i).trim())
        start = i + 1
      }
    }
  }
  out.push(value.slice(start).trim())
  return out.filter(Boolean)
}
