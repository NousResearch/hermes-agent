/**
 * Which browser tab owns the remembered-chat restore.
 *
 * The native Desktop marks every extra window `?peer=1` and a peer boots on the
 * fresh draft by design: it must not replay the primary window's remembered
 * chat (#74948). A browser tab carries no such mark — every tab is the same URL
 * on the same origin — so the Webapp elects one owner with a Web Lock held for
 * that tab's lifetime: the first tab replays the remembered chat, every later
 * tab boots like a peer.
 *
 * The election resolves `true` outside a browser host (the native primary
 * window keeps its restore) and when the platform has no Web Locks, so a
 * browser that cannot elect keeps today's behaviour instead of losing the
 * restore altogether.
 */

import { isBrowserHostedDesktop } from '@/lib/platform'

const PRIMARY_TAB_LOCK = 'hermes.webapp.primary-tab'

let role: Promise<boolean> | null = null

/**
 * The answer when it is already known without an election: `true` for a native
 * window (its restore is its own, never shared with a tab), `null` for a
 * browser host, where the lock has to be taken first.
 */
export function browserTabNavigationRole(): boolean | null {
  return isBrowserHostedDesktop() ? null : true
}

/** Whether THIS tab replays the origin's remembered navigation. */
export function webappTabReplaysRememberedNavigation(): Promise<boolean> {
  role ??= electPrimaryTab()

  return role
}

async function electPrimaryTab(): Promise<boolean> {
  if (!isBrowserHostedDesktop()) {
    return true
  }

  const locks = typeof navigator === 'undefined' ? undefined : navigator.locks

  if (!locks) {
    return true
  }

  return new Promise<boolean>(resolve => {
    void locks.request(PRIMARY_TAB_LOCK, { ifAvailable: true }, lock => {
      // Another tab already holds it: this tab is a peer.
      if (!lock) {
        resolve(false)

        return
      }

      resolve(true)

      // Held until this tab unloads, so a tab opened later sees it taken.
      return new Promise<void>(() => {})
    })
  })
}

/** @internal Reset the elected role for tests. */
export function _resetWebappTabRoleForTests(): void {
  role = null
}
