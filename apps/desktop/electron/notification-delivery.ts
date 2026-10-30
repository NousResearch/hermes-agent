import type { EventEmitter } from 'node:events'

interface DeliveryNotification extends EventEmitter {
  show: () => unknown
}

const WINDOWS_DELIVERY_TIMEOUT_MS = 5000

// Windows show() queues native work and returns void. Its show event confirms
// OS posting, not human attention: Windows notification policy can still mute it.
export function showNotification(notification: DeliveryNotification, platform: NodeJS.Platform): Promise<boolean> {
  if (platform !== 'win32') {
    return Promise.resolve(notification.show()).then(result => result !== false)
  }

  return new Promise(resolve => {
    let settled = false

    const finish = (accepted: boolean) => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      notification.removeListener('show', onShow)
      notification.removeListener('failed', onFailed)
      resolve(accepted)
    }

    const onShow = () => finish(true)
    const onFailed = () => finish(false)
    const timer = setTimeout(onFailed, WINDOWS_DELIVERY_TIMEOUT_MS)
    timer.unref()
    notification.on('show', onShow)
    notification.on('failed', onFailed)

    try {
      notification.show()
    } catch {
      finish(false)
    }
  })
}
