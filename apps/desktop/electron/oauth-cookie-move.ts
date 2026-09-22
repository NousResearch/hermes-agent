/**
 * oauth-cookie-move.ts (ForgeGuard fork)
 *
 * Carry a cookie-flow gateway's Hermes session cookies from one Electron
 * session partition to another — the effectful half of the repair
 * oauth-partition-moves.ts plans. See that module for why a gateway's jar
 * changes under it.
 *
 * Kept free of `electron` imports: main.ts injects `session.fromPartition`,
 * and the tests inject in-memory jars.
 *
 * Only the gateway's own session cookies move (`hermes_session_at`, `_rt`,
 * `_provider`, `_pkce`, in their bare, `__Secure-` and `__Host-` forms).
 * `hermes_sso_attempt` and anything a portal or IDP set stay where they are.
 *
 * Every cookie is written against the GATEWAY's own origin rather than a
 * scheme guessed from the cookie's `secure` flag: a dashboard behind a TLS
 * terminator in another container writes its cookies in their plain-HTTP
 * shape (no Secure, no prefix — hermes_cli/dashboard_auth/cookies.py) even
 * though the desktop reaches it over https, and Chromium binds a cookie to
 * the scheme that set it. Host-only cookies are re-set WITHOUT a `domain`,
 * which is also what a `__Host-` cookie requires.
 *
 * Best effort throughout: a failure is logged and leaves the user where they
 * would have been without the move — signed out of that gateway, with the
 * usual "sign in" prompt.
 */

import { cookiesHaveLiveSession } from './connection-config'
import type { OauthPartitionMove } from './oauth-partition-moves'

export interface JarCookie {
  name: string
  value: string
  domain?: string
  hostOnly?: boolean
  path?: string
  secure?: boolean
  httpOnly?: boolean
  sameSite?: string
  expirationDate?: number
}

export interface CookieJarLike {
  cookies: {
    get(filter: { url?: string }): Promise<JarCookie[]>
    set(details: Record<string, unknown>): Promise<void>
    remove(url: string, name: string): Promise<void>
    flushStore?(): Promise<void>
  }
}

export interface MoveOauthCookiesDeps {
  fromPartition(partition: string): CookieJarLike | null | undefined
  log?(line: string): void
}

export interface OauthCookieMoveResult {
  url: string
  from: string
  to: string
  moved: number
  failed: number
  skipped?: 'destination-signed-in' | 'nothing-to-move' | 'jar-unavailable' | 'bad-url'
}

const SESSION_COOKIE_BASE = 'hermes_session'

/** True for the gateway's own session cookies, in any prefix form. */
export function isHermesSessionCookieName(name: unknown): boolean {
  if (typeof name !== 'string') {
    return false
  }

  const bare = name.replace(/^__(Host|Secure)-/, '')

  return bare === SESSION_COOKIE_BASE || bare.startsWith(`${SESSION_COOKIE_BASE}_`)
}

function cookieUrl(origin: string, cookie: JarCookie): string {
  const path = typeof cookie.path === 'string' && cookie.path.startsWith('/') ? cookie.path : '/'

  return `${origin}${path}`
}

function setDetails(origin: string, cookie: JarCookie): Record<string, unknown> {
  const details: Record<string, unknown> = {
    url: cookieUrl(origin, cookie),
    name: cookie.name,
    value: cookie.value,
    path: cookie.path || '/',
    secure: cookie.secure === true,
    httpOnly: cookie.httpOnly === true
  }

  if (cookie.sameSite) {
    details.sameSite = cookie.sameSite
  }

  if (typeof cookie.expirationDate === 'number') {
    details.expirationDate = cookie.expirationDate
  }

  // A host-only cookie must stay host-only: passing its domain back would
  // widen it to subdomains, and Chromium refuses a `__Host-` cookie that
  // carries a Domain at all.
  if (cookie.hostOnly === false && cookie.domain) {
    details.domain = cookie.domain
  }

  return details
}

async function moveOne(move: OauthPartitionMove, deps: MoveOauthCookiesDeps): Promise<OauthCookieMoveResult> {
  const log = deps.log || (() => undefined)
  const result: OauthCookieMoveResult = { ...move, moved: 0, failed: 0 }

  let origin: string

  try {
    origin = new URL(move.url).origin
  } catch {
    return { ...result, skipped: 'bad-url' }
  }

  const src = deps.fromPartition(move.from)
  const dst = deps.fromPartition(move.to)

  if (!src || !dst) {
    return { ...result, skipped: 'jar-unavailable' }
  }

  const cookies = (await src.cookies.get({ url: move.url })).filter(
    c => c && isHermesSessionCookieName(c.name) && c.value
  )

  if (!cookies.length) {
    return { ...result, skipped: 'nothing-to-move' }
  }

  // Never clobber a sign-in the destination already holds — it is newer than
  // anything this move could bring, or it belongs to the same session anyway.
  if (cookiesHaveLiveSession(await dst.cookies.get({ url: move.url }))) {
    log(`[oauth-partition] ${move.url} already signed in on ${move.to}; left ${move.from} as it was`)

    return { ...result, skipped: 'destination-signed-in' }
  }

  for (const cookie of cookies) {
    try {
      await dst.cookies.set(setDetails(origin, cookie))
    } catch (error) {
      result.failed += 1
      log(
        `[oauth-partition] could not move ${cookie.name} for ${move.url} (${move.from} -> ${move.to}): ${
          (error as Error)?.message || error
        }`
      )

      continue
    }

    // A move, not a copy: the old jar must not keep a session that now
    // belongs to another jar's gateway (#92183's whole point).
    try {
      await src.cookies.remove(cookieUrl(origin, cookie), cookie.name)
    } catch {
      // The copy landed; a stale source cookie self-expires.
    }

    result.moved += 1
  }

  try {
    await dst.cookies.flushStore?.()
  } catch {
    // Best effort: Chromium flushes on its own schedule too.
  }

  log(
    `[oauth-partition] moved ${result.moved} session cookie(s) for ${move.url} from ${move.from} to ${move.to}` +
      (result.failed ? ` (${result.failed} failed)` : '')
  )

  return result
}

/** Apply every move in order. One move's failure never stops the next. */
export async function moveOauthCookies(
  moves: OauthPartitionMove[],
  deps: MoveOauthCookiesDeps
): Promise<OauthCookieMoveResult[]> {
  const results: OauthCookieMoveResult[] = []

  for (const move of moves) {
    try {
      results.push(await moveOne(move, deps))
    } catch (error) {
      deps.log?.(
        `[oauth-partition] could not move session for ${move.url} (${move.from} -> ${move.to}): ${
          (error as Error)?.message || error
        }`
      )
      results.push({ ...move, moved: 0, failed: 1 })
    }
  }

  return results
}
