/**
 * remote-session-cookie-sync.ts
 *
 * Keeps an OAuth partition's Chromium jar and the in-memory cookie mirror
 * (remote-session-cookies.ts, #61457) on the same session cookies.
 *
 * The gateway rotates its refresh-token cookie on every refresh, and each store
 * only sees the rotations made through it. Whichever store lagged replayed the
 * previous token, and the portal revokes the whole session on a replay. Every
 * jar write is followed into the mirror, and every Set-Cookie the mirror
 * records is written into the jar.
 */

import type { Session } from 'electron'

import { jarWriteFromSetCookie, remoteSessionCookies } from './remote-session-cookies'

type Log = (message: string) => void

/**
 * Follow every write to `sess`'s jar into the mirror. Install once per session
 * object: main.ts caches each OAuth session and never evicts it, and a second
 * listener would only rewrite the same value.
 */
export function keepMirrorInStepWithJar(sess: Session, partition: string): void {
  sess.cookies.on('changed', (_event, cookie, _cause, removed) => {
    if (!removed) {
      remoteSessionCookies.followJarWrite(partition, cookie)
    }
  })
}

async function writeSetCookiesIntoJar(sess: Session, url: string, setCookie: string[], log: Log): Promise<void> {
  for (const header of setCookie) {
    const write = jarWriteFromSetCookie(url, header)

    try {
      if (write?.kind === 'set') {
        await sess.cookies.set(write.details)
      } else if (write?.kind === 'remove') {
        await sess.cookies.remove(url, write.name)
      }
    } catch (error) {
      log(`Remote session cookie jar write failed for ${new URL(url).host}: ${(error as Error)?.message || error}`)
    }
  }
}

/**
 * Record a mirrored response's Set-Cookie header(s) in the mirror and the jar.
 * The jar write is fire-and-forget: the mirror, updated synchronously, is what
 * the next mirrored request sends; the jar catches up for its own readers.
 */
export function recordSetCookies(
  sess: Session,
  partition: string,
  url: string,
  setCookie: string | string[] | undefined,
  log: Log
): void {
  remoteSessionCookies.record(partition, url, setCookie)
  void writeSetCookiesIntoJar(sess, url, [setCookie ?? []].flat(), log)
}

/**
 * #61457: snapshot a partition jar's session cookies for this gateway into the
 * mirror, keyed by the SAME partition the jar belongs to. Best-effort: a failed
 * read just means the next response's Set-Cookie capture populates the mirror.
 */
export async function captureRemoteSessionCookiesIntoMirror(
  sess: Session | null | undefined,
  partition: string,
  baseUrl: string,
  log: Log
): Promise<void> {
  if (!sess || !partition) {
    return
  }

  try {
    remoteSessionCookies.recordFromJar(partition, baseUrl, await sess.cookies.get({ url: baseUrl }))
  } catch (error) {
    log(`Remote session cookie mirror capture failed: ${(error as Error)?.message || error}`)
  }
}
