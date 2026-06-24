import { computed } from 'nanostores'

import { $nativeNotifyPrefs } from './native-notifications'
import { $attentionSessionIds, $unreadSessionIds } from './session'

// The "needs you" count — sessions with unread output OR a blocking prompt.
// Drives the OS-level dock/taskbar badge + window-title prefix so the state
// survives minimization and alt-tab (the native toast is transient; this is
// the persistent layer). Working sessions are intentionally excluded: a busy
// turn is not actionable, so it doesn't belong in a "needs you" count.
export const $attentionBadgeCount = computed(
  [$unreadSessionIds, $attentionSessionIds],
  (unread, attention) => {
    // Union by id — a session can be both unread and needs-input; count it once.
    const ids = new Set([...unread, ...attention])

    return ids.size
  }
)

// Respects the user's unreadBadge pref (Settings → Notifications). When off,
// the badge is forced to 0 so it clears even if a count was previously pushed.
const $effectiveBadgeCount = computed(
  [$attentionBadgeCount, $nativeNotifyPrefs],
  (count, prefs) => (prefs.unreadBadge ? count : 0)
)

// One subscription drives the OS badge. Set up once on the renderer side
// (called from the app shell); idempotent so repeated calls are safe.
let installed = false

export function installSessionBadgeSync(): () => void {
  if (installed) {
    return () => undefined
  }

  installed = true

  const push = (count: number) => {
    void window.hermesDesktop?.setBadge?.(count)
  }

  // Push the initial value, then keep it in sync.
  push($effectiveBadgeCount.get())

  const off = $effectiveBadgeCount.subscribe(count => push(count))

  return () => {
    off()
    installed = false
    push(0)
  }
}

