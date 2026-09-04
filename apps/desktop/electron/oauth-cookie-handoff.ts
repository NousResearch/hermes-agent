/**
 * Move gateway session cookies when a URL acquires a different authoritative
 * OAuth partition (notably: an unsaved registry draft becomes a saved,
 * per-connection remote). Electron's persisted cookie store hydrates lazily,
 * so an empty first read is retried after an explicit warm-up.
 */

export interface CookieLike {
  name: string
  value: string
  domain?: string
  path?: string
  secure?: boolean
  httpOnly?: boolean
  sameSite?: string
  expirationDate?: number
}

export interface CookieSessionLike {
  cookies: {
    get(filter: { url: string }): Promise<CookieLike[]>
    set(details: Record<string, unknown>): Promise<void>
  }
  flushStorageData?: () => void
}

export interface HandoffOauthCookiesOptions {
  source: CookieSessionLike
  target: CookieSessionLike
  url: string
  cookieNames: readonly string[]
  delaysMs?: readonly number[]
  wait?: (ms: number) => Promise<void>
}

export async function handoffOauthCookies({
  source,
  target,
  url,
  cookieNames,
  delaysMs = [0, 30, 60, 90],
  wait = ms => new Promise(resolve => setTimeout(resolve, ms))
}: HandoffOauthCookiesOptions): Promise<number> {
  source.flushStorageData?.()

  let cookies: CookieLike[] = []

  for (const delayMs of delaysMs) {
    if (delayMs > 0) {
      await wait(delayMs)
    }

    cookies = (await source.cookies.get({ url })).filter(
      cookie => cookieNames.includes(cookie.name) && Boolean(cookie.value)
    )

    if (cookies.length > 0) {
      break
    }
  }

  for (const cookie of cookies) {
    const details: Record<string, unknown> = {
      url,
      name: cookie.name,
      value: cookie.value,
      path: cookie.path || '/'
    }

    // Deliberately omit Domain: gateway cookies are host-only, and Chromium
    // rejects __Host- cookies if a Domain attribute is supplied on set().
    if (typeof cookie.secure === 'boolean') details.secure = cookie.secure
    if (typeof cookie.httpOnly === 'boolean') details.httpOnly = cookie.httpOnly
    if (cookie.sameSite) details.sameSite = cookie.sameSite
    if (cookie.expirationDate) details.expirationDate = cookie.expirationDate

    // This is an authoritative handoff. A failed write must reject the save;
    // silently registering the connection would strand the source cookie.
    await target.cookies.set(details)
  }

  target.flushStorageData?.()

  return cookies.length
}
