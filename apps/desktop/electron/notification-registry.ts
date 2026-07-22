// Keeps shown OS notifications alive until the user is done with them.
//
// Electron's Notification is a JS object wrapping a native toast, and the
// wrapper owns the event emitter. Build one in a local `const`, call show(),
// and return: the toast is on screen but nothing in JS references the wrapper
// anymore, so it becomes eligible for GC. When V8 collects it the emitter dies
// with it, and the user's click lands on a toast whose 'click' handler no
// longer exists — the toast dismisses and nothing else happens.
//
// This bites hardest on Windows, where a toast persists in the Action Center
// for minutes: the gap between show() and the click is wide enough for a GC to
// land almost every time, which is why the symptom reads as "clicking the
// notification does nothing" rather than as a flake. Holding a strong
// reference for the toast's clickable lifetime is the fix.
//
// Electron-free and dependency-injected so it stays unit-testable, mirroring
// how the rest of electron/*.ts splits logic out of the main.ts monolith.

// Lifecycle events after which a toast can no longer be clicked, so the
// reference can be dropped.
const RELEASE_EVENTS = ['click', 'close', 'failed']

// Backstop for the reference held per notification. 'close' is not guaranteed
// on every platform, so without a ceiling a long-running session could retain
// every toast it ever showed. Ten minutes is far longer than a user takes to
// act on a toast, while still bounding the set.
const NOTIFICATION_RETENTION_TTL_MS = 10 * 60 * 1000

function createNotificationRegistry({
  ttlMs = NOTIFICATION_RETENTION_TTL_MS,
  setTimer = setTimeout,
  clearTimer = clearTimeout
}: any = {}) {
  const live = new Set()

  function retain(notification) {
    if (!notification) {
      return null
    }

    live.add(notification)

    let timer: any = null
    const release = () => {
      if (timer !== null) {
        clearTimer(timer)
        timer = null
      }

      live.delete(notification)
    }

    // Additional listeners — never a replacement for the caller's own 'click'
    // handler, which is the whole point of keeping the emitter alive.
    for (const event of RELEASE_EVENTS) {
      notification.on?.(event, release)
    }

    timer = setTimer(release, ttlMs)
    // A pending toast must never hold the process open on quit.
    timer?.unref?.()

    return notification
  }

  return {
    retain,
    has: notification => live.has(notification),
    get size() {
      return live.size
    }
  }
}

export { createNotificationRegistry, NOTIFICATION_RETENTION_TTL_MS, RELEASE_EVENTS }
