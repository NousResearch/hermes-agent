import type { EventEmitter } from 'node:events'

interface NativeNotification extends EventEmitter {
  show: () => void
}

const DELIVERY_TIMEOUT_MS = 5000

// Native show() returns void; delivery (including unsigned macOS failure) is
// reported by events. Stop waiting if the OS never reports either outcome.
export function showNativeNotification(notification: NativeNotification): Promise<boolean> {
  return new Promise(resolve => {
    const finish = (delivered: boolean) => {
      clearTimeout(timer)
      notification.removeListener('show', shown)
      notification.removeListener('failed', failed)
      resolve(delivered)
    }

    const shown = () => finish(true)
    const failed = () => finish(false)
    const timer = setTimeout(failed, DELIVERY_TIMEOUT_MS)
    timer.unref()
    notification.on('show', shown)
    notification.on('failed', failed)

    try {
      notification.show()
    } catch {
      failed()
    }
  })
}
