import type { EventEmitter } from 'node:events'

interface RetainedNotification extends EventEmitter {
  close: () => void
}

// Windows emits close on banner timeout even while Action Center keeps the
// notification clickable. Only consumption or failed delivery releases it.
const RELEASE_EVENTS = ['click', 'action', 'failed']
const NOTIFICATION_RETENTION_TTL_MS = 10 * 60 * 1000

export function createNotificationRegistry({ ttlMs = NOTIFICATION_RETENTION_TTL_MS } = {}) {
  const live = new Set<RetainedNotification>()

  function retain(notification: RetainedNotification): void {
    live.add(notification)

    const release = () => {
      clearTimeout(timer)
      live.delete(notification)

      for (const event of RELEASE_EVENTS) {
        notification.removeListener(event, release)
      }
    }

    for (const event of RELEASE_EVENTS) {
      notification.on(event, release)
    }

    // Bound retention without leaving an OS notification whose handler is gone.
    const timer = setTimeout(() => {
      notification.close()
      release()
    }, ttlMs)

    timer.unref()
  }

  return { retain, has: (notification: RetainedNotification) => live.has(notification) }
}
