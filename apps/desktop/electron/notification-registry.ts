import type { EventEmitter } from 'node:events'

interface RetainedNotification extends EventEmitter {
  close: () => void
}

/** What a retained notification is about, so seeing that session can withdraw it. */
export interface NotificationSubject {
  kind?: string
  sessionId?: string
}

// Windows emits close on banner timeout even while Action Center keeps the
// notification clickable. Linux's terminal NotificationClosed can opt into release.
const RELEASE_EVENTS = ['click', 'action', 'failed']
const NOTIFICATION_RETENTION_TTL_MS = 10 * 60 * 1000

export function createNotificationRegistry({ ttlMs = NOTIFICATION_RETENTION_TTL_MS, releaseOnClose = false } = {}) {
  const live = new Map<RetainedNotification, { release: () => void; subject?: NotificationSubject }>()
  const releaseEvents = releaseOnClose ? [...RELEASE_EVENTS, 'close'] : RELEASE_EVENTS

  function retain(notification: RetainedNotification, subject?: NotificationSubject): void {
    const release = () => {
      clearTimeout(timer)
      live.delete(notification)

      for (const event of releaseEvents) {
        notification.removeListener(event, release)
      }
    }

    live.set(notification, { release, subject })

    for (const event of releaseEvents) {
      notification.on(event, release)
    }

    // Bound retention without leaving an OS notification whose handler is gone.
    const timer = setTimeout(() => {
      notification.close()
      release()
    }, ttlMs)

    timer.unref()
  }

  /** Withdraw every retained notification about `sessionId` (banner and the
   *  Notification/Action Center entry), optionally only the given kinds. */
  function dismiss(sessionId: string, kinds?: readonly string[]): number {
    let closed = 0

    for (const [notification, { release, subject }] of [...live]) {
      if (!sessionId || subject?.sessionId !== sessionId || (kinds && !kinds.includes(subject.kind ?? ''))) {
        continue
      }

      notification.close()
      release()
      closed++
    }

    return closed
  }

  return { retain, dismiss, has: (notification: RetainedNotification) => live.has(notification) }
}
